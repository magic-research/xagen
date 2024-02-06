# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""

import math, time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_utils import misc
from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.volumetric_rendering import math_utils
from training.volumetric_rendering.utils import grid_sample
import torch.autograd as autograd

from pytorch3d.ops.knn import knn_points
from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from einops import rearrange
# import pysdf
# torch.autograd.set_detect_anomaly(True)

@misc.profiled_function
def batch_transform(P, v, pad_ones=True):
    if pad_ones:
        homo = torch.ones((*v.shape[:-1], 1), dtype=v.dtype, device=v.device)
    else:
        homo = torch.zeros((*v.shape[:-1], 1), dtype=v.dtype, device=v.device)
    v_homo = torch.cat((v, homo), dim=-1)
    v_homo = torch.matmul(P, v_homo.unsqueeze(-1))
    v_ = v_homo[..., :3, 0]
    return v_

@misc.profiled_function
def batch_index_select(data, inds, device):
    bs, nv = data.shape[:2]
    inds = inds + (torch.arange(bs, dtype=torch.int32, device=device) * nv)[:, None, None]
    data = data.reshape(bs*nv, *data.shape[2:])
    return data[inds.long()]

@misc.profiled_function
def cal_sdf_batch(verts, faces, triangles, points, device):
    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]
    
    Bsize = points.shape[0]
    residues, _, _ = point_to_mesh_distance(points.contiguous(), triangles)
    pts_dist = torch.sqrt(residues.contiguous())
    pts_signs = -2.0 * (check_sign(verts.cuda(), faces[0].cuda(), points.cuda()).float() - 0.5).to(device) # negative outside
    pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)

    return pts_sdf.view(Bsize, -1, 1)

@misc.profiled_function
def cal_sdf_batch_pysdf(verts, faces, triangles, points, device):
    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]
    
    Bsize = points.shape[0]
    mesh_to_sdf = pysdf.SDF
    pts_sdf = []
    for vert, pt in zip(verts, points):
        mesh_to_sdf_f = mesh_to_sdf(vert.detach().cpu().numpy(), faces[0].detach().cpu().numpy())
        pts_sdf.append(torch.from_numpy(-mesh_to_sdf_f(pt.detach().reshape(-1, 3).cpu().numpy())).to(device).reshape(pt.size(0), 1).unsqueeze(0)) # negative outside
    pts_sdf = torch.cat(pts_sdf, 0)

    return pts_sdf.view(Bsize, -1, 1)

def get_eikonal_term(pts, sdf):
    eikonal_term = autograd.grad(outputs=sdf, inputs=pts,
                                 grad_outputs=torch.ones_like(sdf, requires_grad=False, device=pts.device),
                                 create_graph=True,
                                 retain_graph=True,
                                 only_inputs=True)[0]
    return torch.clamp(torch.nan_to_num((eikonal_term.norm(dim=-1) - 1) ** 2, 0.0), 0, 1e6)

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]],
                            [[0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, canonical_bbox=None, mode='bilinear', padding_mode='zeros', box_warp=None, face_bbox=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.reshape(N*n_planes, C, H, W)
    # if canonical_body_bbox is None:
    #     coordinates = (2 / box_warp) * coordinates # TODO: add specific box bounds
    # else:
    #     coordinates = 2 * (coordinates-canonical_body_bbox[:, :1]) / (canonical_body_bbox[:, 1:2]-canonical_body_bbox[:, :1]) - 1
    assert canonical_bbox is not None, "canonical bbox required"
    coordinates = 2 * (coordinates-canonical_bbox[:, :1]) / (canonical_bbox[:, 1:2]-canonical_bbox[:, :1]) - 1
    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = grid_sample(plane_features, projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

# def sample_from_planes(plane_axes, plane_features, coordinates, canonical_bbox=None, mode='bilinear', padding_mode='zeros', box_warp=None, face_bbox=None):
#     assert padding_mode == 'zeros'
#     N, n_planes, C, H, W = plane_features.shape
#     _, M, _ = coordinates.shape
#     plane_features = plane_features.view(N*n_planes, C, H, W)
#     assert canonical_bbox is not None, "canonical bbox required"
#     coordinates = 2 * (coordinates-canonical_bbox[:, :1]) / (canonical_bbox[:, 1:2]-canonical_bbox[:, :1]) - 1
#     projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
#     output_features = grid_sample(plane_features, projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
#     return output_features
    
def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features

def dda(rays_o, rays_d, bbox_3D):
    # bbox_3D 1x2x3
    cam_dist = torch.norm(rays_o, dim=-1, keepdim=True)
    near = cam_dist[:, :1, 0] - 0.75
    far = cam_dist[:, :1, 0] + 0.75
    near = near.unsqueeze(-1) * torch.ones_like(rays_d[..., :1]).float()
    far = far.unsqueeze(-1) * torch.ones_like(rays_d[..., :1]).float()

    # inv_ray_d = 1.0 / (rays_d + 1e-6) # 4x64x64x3
    # t_min = (bbox_3D[:,:1] - rays_o) * inv_ray_d  # 4x64x64x3
    # t_max = (bbox_3D[:,1:] - rays_o) * inv_ray_d
    # t = torch.stack((t_min, t_max))  # 2x4x64x64x3
    # t_min = torch.amax(torch.amin(t, dim=0)[..., 2:], dim=-1, keepdim=True) # 4x64x64x1
    # t_max = torch.amin(torch.amax(t, dim=0)[..., 2:], dim=-1, keepdim=True)
    t_min = torch.ones_like(near) * bbox_3D[:, :1, 2:]
    t_max = torch.ones_like(far) * bbox_3D[:, 1:, 2:]
    mask = t_min >= t_max
    t_min[mask] = near[mask]
    t_max[mask] = far[mask]
    return t_min, t_max
    
class ImportanceRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes()

    def forward(
        self, planes, face_planes, hand_planes, decoder, decoder_face, decoder_hand, ray_origins, ray_directions, 
        rendering_options, canonical_mapping_kwargs, return_sdf=False, return_eikonal=False, smplx_reg=False, 
        offset_network=None, canonical_reg=False, smplx_reg_full_space=False, shrink_face_cano=False):
        self.plane_axes = self.plane_axes.to(ray_origins.device)
        
        # TODO: add these into arugments
        use_cam_dist = rendering_options.get('use_cam_dist', False)
        calc_eikonal_coarse = rendering_options.get('calc_eikonal_coarse', False)
        depth_noise = rendering_options.get('depth_noise', True)

        if use_cam_dist:
            cam_dist = torch.norm(ray_origins, dim=-1, keepdim=True)
            # ray_start = cam_dist - 1.0
            # ray_end = cam_dist + 1.0
            ray_start, ray_end = dda(ray_origins, ray_directions, canonical_mapping_kwargs['obs_bbox_cam'])
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], depth_noise, rendering_options['disparity_space_sampling'])
            rendering_options['box_warp'] = 4.0
        # else:
        #     if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
        #         ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
        #         is_ray_valid = ray_end > ray_start
        #         if torch.any(is_ray_valid).item():
        #             ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
        #             ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
        #         depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        #     else:
        #         # Create stratified depth samples
        #         depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)
        eikonal = None

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(sample_coordinates[0, ::40, 0].cpu(), sample_coordinates[0, ::40, 1].cpu(), sample_coordinates[0, ::40, 2].cpu(), alpha=0.05)
        # ax.scatter(canonical_mapping_kwargs['verts'][0, ::5, 0].cpu(), canonical_mapping_kwargs['verts'][0, ::5, 1].cpu(), canonical_mapping_kwargs['verts'][0, ::5, 2].cpu(), alpha=0.05)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.view_init(0, 0)
        # plt.savefig("tmp_yz_fix.jpg")
        # ax.view_init(90, 90)
        # plt.savefig("tmp_xy_fix.jpg")
        # ax.view_init(0, 90)
        # plt.savefig("tmp_xz_fix.jpg")

        if calc_eikonal_coarse:
            out = self.run_model(planes, face_planes, hand_planes, decoder, decoder_face, decoder_hand, sample_coordinates, sample_directions, rendering_options, return_eikonal=return_eikonal, return_sdf=return_sdf, offset_network=offset_network, smplx_reg_full_space=smplx_reg_full_space, **canonical_mapping_kwargs)
            eikonal_coarse = out['eikonal']
        else:
            out = self.run_model(planes, face_planes, hand_planes, decoder, decoder_face, decoder_hand, sample_coordinates, sample_directions, rendering_options, return_eikonal=False, return_sdf=return_sdf, offset_network=offset_network, smplx_reg_full_space=smplx_reg_full_space, **canonical_mapping_kwargs)
            eikonal_coarse = None
        
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        # always return coarse level sdf for minsurf loss and visualize
        sdf_coarse = out['sdf']
        smplx_sdf = out['smplx_sdf']
        sdf_reg_weights = out['sdf_reg_weights']
        deformation_reg = out['deformation_reg']

        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance, det=(depth_noise==False))

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            out = self.run_model(planes, face_planes, hand_planes, decoder, decoder_face, decoder_hand, sample_coordinates, sample_directions, rendering_options, return_eikonal=return_eikonal, return_sdf=return_sdf, offset_network=offset_network, smplx_reg_full_space=smplx_reg_full_space, **canonical_mapping_kwargs)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            eikonal = out['eikonal']
            overlap_features_fine = out['overlap']
            if out['deformation_reg'] is not None:
                deformation_reg = {k:(deformation_reg[k]+out['deformation_reg'][k])/2 for k in deformation_reg}
            
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(
                depths_coarse, colors_coarse, densities_coarse,
                depths_fine, colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        else:
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)
        
        if eikonal_coarse is not None:
            if eikonal is not None:
                eikonal = torch.cat((eikonal, eikonal_coarse), -1)
            else:
                eikonal = eikonal_coarse
        
        if False: #smplx_reg and smplx_sdf is None and not smplx_reg_full_space:
            # NOTE: replace with verts_observation, different from original AvatarGen
            if canonical_reg:
                verts = canonical_mapping_kwargs['verts_canonical'].clone()
            else:
                verts = canonical_mapping_kwargs['verts'].clone()
            # Sample points in canonical space
            surf_coordinates = verts + (2 * torch.rand_like(verts) - 1) * rendering_options['box_warp'] / 2 * 0.01
            out = self.run_model(
                planes, decoder, surf_coordinates, sample_directions, rendering_options, 
                return_eikonal=False, return_sdf=True, only_density=True, canonical_reg=canonical_reg,
                offset_network=offset_network, **canonical_mapping_kwargs
            )
            smplx_sdf = out['sdf']
            sdf_reg_weights = torch.ones_like(smplx_sdf)

        return rgb_final, depth_final, weights.sum(2), eikonal, smplx_sdf, sdf_reg_weights, sdf_coarse, deformation_reg, overlap_features_fine

    def run_model(
        self, body_planes, face_planes, hand_planes, decoder_body, decoder_face, decoder_hand, sample_coordinates, sample_directions, options, 
        verts, verts_canonical, faces, obs_bbox, canonical_body_bbox, canonical_face_bbox, canonical_hand_bbox, canonical_aux_bboxes, 
        triangles, ober2cano_transform, lbs_weights, return_sdf=False, return_eikonal=False,
        only_density=False, canonical_reg=False, offset_network=None, smplx_canonical_sdf=None,
        smplx_params=None, ws=None, ws_geo=None, smplx_reg_full_space=False, joints=None, **kwargs):
        
        if return_eikonal and not only_density:
            sample_coordinates.requires_grad_(True)
        else:
            sample_coordinates.requires_grad_(False)

        num_batch, num_pts = sample_coordinates.shape[:2]
        device = sample_coordinates.device

        obs_mask, obs_coords_mask, obs_max_length, \
        cano_body_mask, cano_body_coords_mask, cano_body_max_length, cano_body_coords, \
        cano_face_mask, cano_face_coords_mask, cano_face_max_length, cano_face_coords, \
        cano_right_hand_mask, cano_right_hand_coords_mask, cano_right_hand_max_length, cano_right_hand_coords, \
        cano_left_hand_mask, cano_left_hand_coords_mask, cano_left_hand_max_length, cano_left_hand_coords, \
        cano_overlap_mask, cano_overlap_coords_mask, cano_overlap_max_length, cano_overlap_coords, \
        cano_overlap_rh_mask, cano_overlap_rh_coords_mask, cano_overlap_rh_max_length, cano_overlap_rh_coords, \
        cano_overlap_lh_mask, cano_overlap_lh_coords_mask, cano_overlap_lh_max_length, cano_overlap_lh_coords, \
        base_sdf, deformation_reg = self.canonical_mapping(
            sample_coordinates, verts, verts_canonical, faces, obs_bbox, canonical_body_bbox, canonical_face_bbox, canonical_hand_bbox, canonical_aux_bboxes, 
            triangles, ober2cano_transform, lbs_weights, options.get('use_deformation', False), smplx_canonical_sdf=smplx_canonical_sdf, 
            offset_network=offset_network, canonical_reg=canonical_reg, smplx_params=smplx_params, ws=ws, ws_geo=ws_geo)

        sampled_body_features = sample_from_planes(self.plane_axes, body_planes, cano_body_coords, canonical_body_bbox, padding_mode='zeros')
        sampled_body_features = sampled_body_features.reshape(num_batch, 3, cano_body_max_length, -1)
        
        sampled_face_features = sample_from_planes(self.plane_axes, face_planes, cano_face_coords, canonical_face_bbox, padding_mode='zeros')
        sampled_face_features = sampled_face_features.reshape(num_batch, 3, cano_face_max_length, -1)
        
        sampled_right_hand_features = sample_from_planes(self.plane_axes, hand_planes, cano_right_hand_coords, canonical_hand_bbox, padding_mode='zeros')
        sampled_right_hand_features = sampled_right_hand_features.reshape(num_batch, 3, cano_right_hand_max_length, -1)
        
        # mirror left hand to right hand
        mirror_cano_left_hand_coords = cano_left_hand_coords.clone()
        mirror_cano_left_hand_coords[..., 0] = -1 * mirror_cano_left_hand_coords[..., 0]
        sampled_left_hand_features = sample_from_planes(self.plane_axes, hand_planes, mirror_cano_left_hand_coords, canonical_hand_bbox, padding_mode='zeros')
        sampled_left_hand_features = sampled_left_hand_features.reshape(num_batch, 3, cano_left_hand_max_length, -1)

        out = decoder_body(sampled_body_features, sample_directions, base_sdf['body'])
        out_face = decoder_face(sampled_face_features, sample_directions, base_sdf['face'])
        out_right_hand = decoder_hand(sampled_right_hand_features, sample_directions, base_sdf['right_hand'])
        out_left_hand = decoder_hand(sampled_left_hand_features, sample_directions, base_sdf['left_hand'])
        
        out['sigma'] = decoder_body.sdf2sigma(-out['sdf'])
        
        if cano_overlap_max_length > 0:
            out_body_overlap = self.gather_overlap_features(cano_overlap_coords, cano_overlap_coords_mask, out, cano_overlap_mask)
        if cano_overlap_rh_max_length > 0:
            out_body_overlap_rh = self.gather_overlap_features(cano_overlap_rh_coords, cano_overlap_rh_coords_mask, out, cano_overlap_rh_mask)
        if cano_overlap_lh_max_length > 0:
            out_body_overlap_lh = self.gather_overlap_features(cano_overlap_lh_coords, cano_overlap_lh_coords_mask, out, cano_overlap_lh_mask)
            
        out_face['sigma'] = decoder_face.sdf2sigma(-out_face['sdf'])
        out_right_hand['sigma'] = decoder_hand.sdf2sigma(-out_right_hand['sdf'])
        out_left_hand['sigma'] = decoder_hand.sdf2sigma(-out_left_hand['sdf'])
        
        out['rgb'][cano_face_mask] = out_face['rgb'][cano_face_coords_mask]
        out['sdf'][cano_face_mask] = out_face['sdf'][cano_face_coords_mask]
        out['sigma'][cano_face_mask] = out_face['sigma'][cano_face_coords_mask]
        
        out['rgb'][cano_right_hand_mask] = out_right_hand['rgb'][cano_right_hand_coords_mask]
        out['sdf'][cano_right_hand_mask] = out_right_hand['sdf'][cano_right_hand_coords_mask]
        out['sigma'][cano_right_hand_mask] = out_right_hand['sigma'][cano_right_hand_coords_mask]
        
        out['rgb'][cano_left_hand_mask] = out_left_hand['rgb'][cano_left_hand_coords_mask]
        out['sdf'][cano_left_hand_mask] = out_left_hand['sdf'][cano_left_hand_coords_mask]
        out['sigma'][cano_left_hand_mask] = out_left_hand['sigma'][cano_left_hand_coords_mask]
        
        if cano_overlap_max_length > 0:
            out_face_overlap = self.gather_overlap_features(cano_overlap_coords, cano_overlap_coords_mask, out, cano_overlap_mask)
            out_overlap, overlap = self.feature_composition(out_body_overlap, out_face_overlap, cano_overlap_coords, cano_overlap_coords_mask, canonical_aux_bboxes['canonical_face_filter_bbox'], canonical_aux_bboxes['canonical_upper_spine_bbox'])
            for k in out:
                out[k][cano_overlap_mask] = out_overlap[k][cano_overlap_coords_mask]
        
        if cano_overlap_rh_max_length > 0:
            out_hand_overlap_rh = self.gather_overlap_features(cano_overlap_rh_coords, cano_overlap_rh_coords_mask, out, cano_overlap_rh_mask)
            out_overlap_rh, overlap_rh = self.feature_composition(out_body_overlap_rh, out_hand_overlap_rh, cano_overlap_rh_coords, cano_overlap_rh_coords_mask, canonical_aux_bboxes['canonical_right_hand_filter_bbox'], canonical_aux_bboxes['canonical_right_arm_bbox'])
            for k in out:
                out[k][cano_overlap_rh_mask] = out_overlap_rh[k][cano_overlap_rh_coords_mask]
                        
        if cano_overlap_lh_max_length > 0:
            out_hand_overlap_lh = self.gather_overlap_features(cano_overlap_lh_coords, cano_overlap_lh_coords_mask, out, cano_overlap_lh_mask)
            out_overlap_lh, overlap_lh = self.feature_composition(out_body_overlap_lh, out_hand_overlap_lh, cano_overlap_lh_coords, cano_overlap_lh_coords_mask, canonical_aux_bboxes['canonical_left_hand_filter_bbox'], canonical_aux_bboxes['canonical_left_arm_bbox'])
            for k in out:
                out[k][cano_overlap_lh_mask] = out_overlap_lh[k][cano_overlap_lh_coords_mask]

        rgb_obs = torch.zeros((num_batch, obs_max_length, out['rgb'].shape[-1]), device=device)
        sdf_obs = torch.zeros((num_batch, obs_max_length, out['sdf'].shape[-1]), device=device) + 30  # TODO: set a bg sdf for bg (30?)
        sigma_obs = torch.zeros((num_batch, obs_max_length, out['sigma'].shape[-1]), device=device)
        rgb_obs[cano_body_mask] = out['rgb'][cano_body_coords_mask]
        sdf_obs[cano_body_mask] = out['sdf'][cano_body_coords_mask]
        sigma_obs[cano_body_mask] = out['sigma'][cano_body_coords_mask]

        rgb_final = torch.zeros((num_batch, num_pts, out['rgb'].shape[-1]), device=device)
        sdf_final = torch.zeros((num_batch, num_pts, out['sdf'].shape[-1]), device=device) + 30  # TODO: set a bg sdf for bg (30?)
        sigma_final = torch.zeros((num_batch, num_pts, out['sigma'].shape[-1]), device=device)
        rgb_final[obs_mask] = rgb_obs[obs_coords_mask]
        sdf_final[obs_mask] = sdf_obs[obs_coords_mask]
        sigma_final[obs_mask] = sigma_obs[obs_coords_mask]

        smplx_sdf = None
        sdf_reg_weights = None
        if smplx_reg_full_space:
            verts_dis = torch.clamp(torch.sqrt(base_sdf['body'].reshape(*out['sdf'].shape)**2), 1e-2, 1e3)
            sdf_reg_weights = torch.exp(-verts_dis**2/2*100)
            smplx_sdf = out['sdf']
        
        out['smplx_sdf'] = smplx_sdf
        out['sdf_reg_weights'] = sdf_reg_weights
        out['deformation_reg'] = deformation_reg
        out['rgb'] = rgb_final
        out['sdf'] = sdf_final
        out['sigma'] = sigma_final
        out['overlap'] = None
        out['overlap_rh'] = None
        out['overlap_lh'] = None
        
        out['eikonal'] = None
        if return_eikonal and not only_density:
            out['eikonal'] = get_eikonal_term(sample_coordinates, out['sdf'])
        
        if not return_sdf and not only_density:
            out['sdf'] = None

        if options.get('density_noise', 0) > 0:
            out['sigma'] = out['sigma'] + torch.randn_like(out['sigma']) * options['density_noise']
        return out

    def deformation(self, offset_network, coords_cano, coords, ws=None, ws_geo=None, smpl_params=None, canonical_aux_bboxes=None):
        input_coords = torch.cat([coords_cano, coords],dim=-1)
        coords_cano_offset = offset_network(input_coords, ws, ws_geo, smpl_params)
        deformed_coords_cano = coords_cano + coords_cano_offset
        # replace with original pts if deformed out of canonical bbox
        bs, num_pts = coords.shape[:2]
        deform_mask = torch.zeros([bs, num_pts], dtype=torch.bool, device=coords.device)
        for key, bbox in canonical_aux_bboxes.items():
            if 'filter' in key or 'overlap' in key:
                if isinstance(bbox, list):
                    continue
                boxed_coords = 2 * (coords_cano-bbox[:,0:1]) / (bbox[:,1:2]-bbox[:,0:1]) - 1
                boxed_coords_deformed = 2 * (deformed_coords_cano-bbox[:,0:1]) / (bbox[:,1:2]-bbox[:,0:1]) - 1
                if 'body' in key:
                    boxed_mask = boxed_coords_deformed.abs().amax(-1) >= 1
                else:
                    boxed_mask = torch.logical_and(boxed_coords.abs().amax(-1) < 1, boxed_coords_deformed.abs().amax(-1) >= 1)
                deform_mask = torch.logical_or(deform_mask, boxed_mask)
        deformed_coords_cano[deform_mask] = coords_cano[deform_mask]
        return deformed_coords_cano, coords_cano_offset
    
    def gather_overlap_features(self, coords, coords_mask, out, orig_mask):
        bs, num_pts = coords.shape[:2]
        rgb = torch.zeros([bs, num_pts, 32], dtype=coords.dtype, device=coords.device)
        rgb[coords_mask] = out['rgb'][orig_mask]
        sdf = torch.zeros([bs, num_pts, 1], dtype=coords.dtype, device=coords.device)
        sdf[coords_mask] = out['sdf'][orig_mask]
        sigma = torch.zeros([bs, num_pts, 1], dtype=coords.dtype, device=coords.device)
        sigma[coords_mask] = out['sigma'][orig_mask]
        return {'rgb':rgb, 'sdf':sdf, 'sigma':sigma}
    
    def feature_composition(self, out_body, out_face, coord, coord_overlap_mask, canonical_face_bbox, canonical_body_bbox, alpha=2, beta=6):
        points = coord[coord_overlap_mask]
        body_points = 2 * (points-canonical_body_bbox[0, :1]) / (canonical_body_bbox[0, 1:2]-canonical_body_bbox[0, :1]) - 1
        face_points = 2 * (points-canonical_face_bbox[0, :1]) / (canonical_face_bbox[0, 1:2]-canonical_face_bbox[0, :1]) - 1
        
        weights_body = (-alpha * torch.pow(body_points, beta).sum(1, keepdim=True)).exp()
        weights_face = (-alpha * torch.pow(face_points, beta).sum(1, keepdim=True)).exp()
        weights_body_norm = weights_body / (weights_body + weights_face + 1e-8)
        weights_face_norm = weights_face / (weights_body + weights_face + 1e-8)
        
        overlap_body_rgb, overlap_body_sdf, overlap_body_sigma = out_body['rgb'][coord_overlap_mask], out_body['sdf'][coord_overlap_mask], out_body['sigma'][coord_overlap_mask]
        overlap_face_rgb, overlap_face_sdf, overlap_face_sigma = out_face['rgb'][coord_overlap_mask], out_face['sdf'][coord_overlap_mask], out_face['sigma'][coord_overlap_mask]
        
        out_face['rgb'][coord_overlap_mask] = weights_face_norm * overlap_face_rgb + weights_body_norm * overlap_body_rgb
        out_face['sdf'][coord_overlap_mask] = weights_face_norm * overlap_face_sdf + weights_body_norm * overlap_body_sdf
        out_face['sigma'][coord_overlap_mask] = weights_face_norm * overlap_face_sigma + weights_body_norm * overlap_body_sigma
        
        overlap = {
            'overlap_face_rgb': overlap_face_rgb,
            'overlap_face_sdf': overlap_face_sdf,
            'overlap_face_sigma': overlap_face_sigma,
            'overlap_body_rgb': overlap_body_rgb,
            'overlap_body_sdf': overlap_body_sdf,
            'overlap_body_sigma': overlap_body_sigma,
        }
        
        return out_face, overlap

    def canonical_mapping(
        self, coords, verts, verts_canonical, smplx_faces, obs_bbox, 
        canonical_body_bbox, canonical_face_bbox, canonical_hand_bbox, canonical_aux_bboxes,
        triangles, ober2cano_transform, lbs_weights, use_deformation=False, 
        offset_network=None, canonical_reg=False, body_sdf_from_obs=False, 
        smplx_params=None, smplx_canonical_sdf=None, ws=None, ws_geo=None):
        """
        coordinates: N x Npts x 3
        """

        device = coords.device
        deformation_reg = None

        if not canonical_reg:
            # filter and mask points using observation bbox
            obs_mask, obs_coords_mask, obs_max_length, obs_coords, _ = self.filter_and_pad(coords, obs_bbox)

            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(filtered_coords[0, ::40, 0].detach().cpu().numpy(), filtered_coords[0, ::40, 1].detach().cpu().numpy(), filtered_coords[0, ::40, 2].detach().cpu().numpy(), alpha=0.05)
            # ax.scatter(verts[0, ::5, 0].detach().cpu().numpy(), verts[0, ::5, 1].detach().cpu().numpy(), verts[0, ::5, 2].detach().cpu().numpy(), alpha=0.05)
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            # ax.view_init(0, 0)
            # plt.savefig("tmp_yz_fix.jpg")
            # ax.view_init(90, 90)
            # plt.savefig("tmp_xy_fix.jpg")
            # ax.view_init(0, 90)
            # plt.savefig("tmp_xz_fix.jpg")            
            
            cano_coords, valid = self.unpose(obs_coords, verts, ober2cano_transform, lbs_weights, device)

            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(filtered_coords_cano[0, ::40, 0].detach().cpu().numpy(), filtered_coords_cano[0, ::40, 1].detach().cpu().numpy(), filtered_coords_cano[0, ::40, 2].detach().cpu().numpy(), alpha=0.05)
            # ax.scatter(verts_canonical[0, ::5, 0].detach().cpu().numpy(), verts_canonical[0, ::5, 1].detach().cpu().numpy(), verts_canonical[0, ::5, 2].detach().cpu().numpy(), alpha=0.05)
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            # ax.view_init(0, 0)
            # plt.savefig("tmp_yz_fix.jpg")
            # ax.view_init(90, 90)
            # plt.savefig("tmp_xy_fix.jpg")
            # ax.view_init(0, 90)
            # plt.savefig("tmp_xz_fix.jpg")

            cano_body_mask, cano_body_coords_mask, cano_body_max_length, \
            cano_body_coords, _ = self.filter_and_pad(
                                    cano_coords, 
                                    canonical_aux_bboxes['canonical_body_fine_grained_filter_bboxes'], 
                                    aux_bbox=[
                                        canonical_aux_bboxes['canonical_face_filter_bbox'], 
                                        canonical_aux_bboxes['canonical_right_hand_filter_bbox'], 
                                        canonical_aux_bboxes['canonical_left_hand_filter_bbox']
                                    ])
        
            if use_deformation:
                assert offset_network is not None
                _obs_coords = torch.zeros_like(cano_body_coords)
                _obs_coords[cano_body_coords_mask] = obs_coords[cano_body_mask].clone()
                cano_body_coords, coords_cano_offset = self.deformation(offset_network, cano_body_coords, _obs_coords, ws, ws_geo, smplx_params, canonical_aux_bboxes)
        else:
            cano_coords = coords

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(filtered_coords[0, ::40, 0].detach().cpu().numpy(), filtered_coords[0, ::40, 1].detach().cpu().numpy(), filtered_coords[0, ::40, 2].detach().cpu().numpy(), alpha=0.05)
        # ax.scatter(verts_canonical[0, ::5, 0].detach().cpu().numpy(), verts_canonical[0, ::5, 1].detach().cpu().numpy(), verts_canonical[0, ::5, 2].detach().cpu().numpy(), alpha=0.05)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.view_init(0, 0)
        # plt.savefig("tmp_yz_fix.jpg")
        # ax.view_init(90, 90)
        # plt.savefig("tmp_xy_fix.jpg")
        # ax.view_init(0, 90)
        # plt.savefig("tmp_xz_fix.jpg")
        
        cano_face_mask, cano_face_coords_mask, cano_face_max_length, cano_face_coords, _ = self.filter_and_pad(cano_body_coords, canonical_aux_bboxes['canonical_face_filter_bbox'])

        cano_right_hand_mask, cano_right_hand_coords_mask, cano_right_hand_max_length, cano_right_hand_coords, _ = self.filter_and_pad(cano_body_coords, canonical_aux_bboxes['canonical_right_hand_filter_bbox'])
        cano_left_hand_mask, cano_left_hand_coords_mask, cano_left_hand_max_length, cano_left_hand_coords, _ = self.filter_and_pad(cano_body_coords, canonical_aux_bboxes['canonical_left_hand_filter_bbox'])

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(filtered_hand_coords[0, ::40, 0].detach().cpu().numpy(), filtered_hand_coords[0, ::40, 1].detach().cpu().numpy(), filtered_hand_coords[0, ::40, 2].detach().cpu().numpy(), alpha=0.05)
        # ax.scatter(verts_canonical[0, ::5, 0].detach().cpu().numpy(), verts_canonical[0, ::5, 1].detach().cpu().numpy(), verts_canonical[0, ::5, 2].detach().cpu().numpy(), alpha=0.05)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.view_init(0, 0)
        # plt.savefig("tmp_yz_fix.jpg")
        # ax.view_init(90, 90)
        # plt.savefig("tmp_xy_fix.jpg")
        # ax.view_init(0, 90)
        # plt.savefig("tmp_xz_fix.jpg")   
        
        cano_overlap_mask, cano_overlap_coords_mask, cano_overlap_max_length, cano_overlap_coords, _ = self.filter_and_pad(cano_body_coords, canonical_aux_bboxes['overlap_head_bbox'])
        
        cano_overlap_rh_mask, cano_overlap_rh_coords_mask, cano_overlap_rh_max_length, cano_overlap_rh_coords, _ = self.filter_and_pad(cano_body_coords, canonical_aux_bboxes['overlap_right_hand_bbox'])
        cano_overlap_lh_mask, cano_overlap_lh_coords_mask, cano_overlap_lh_max_length, cano_overlap_lh_coords, _ = self.filter_and_pad(cano_body_coords, canonical_aux_bboxes['overlap_left_hand_bbox'])
        
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(cano_overlap_coords[0, ::40, 0].detach().cpu().numpy(), cano_overlap_coords[0, ::40, 1].detach().cpu().numpy(), cano_overlap_coords[0, ::40, 2].detach().cpu().numpy(), alpha=0.05)
        # ax.scatter(verts_canonical[0, ::5, 0].detach().cpu().numpy(), verts_canonical[0, ::5, 1].detach().cpu().numpy(), verts_canonical[0, ::5, 2].detach().cpu().numpy(), alpha=0.05)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.view_init(0, 0)
        # plt.savefig("tmp_yz_fix.jpg")
        # ax.view_init(90, 90)
        # plt.savefig("tmp_xy_fix.jpg")
        # ax.view_init(0, 90)
        # plt.savefig("tmp_xz_fix.jpg")   
        
        # Query SMPL SDF
        # start = time.time()
        # t1 = time.time() - start
        # start = time.time()
        # kaolin_body_sdf = cal_sdf_batch(
        #     verts_canonical,
        #     smplx_faces.unsqueeze(0).repeat(verts_canonical.size(0), 1, 1), 
        #     triangles, 
        #     cano_body_coords,
        #     device
        # )
        # t2 = time.time() - start
        # start = time.time()
        # body_sdf = cal_sdf_batch(
        #     verts,
        #     smplx_faces.unsqueeze(0).repeat(verts.size(0), 1, 1), 
        #     triangles, 
        #     cano_body_coords,
        #     device
        # )
        # t3 = time.time() - start
        # print(f"inter:{t1:.6f}\tkaolin cano:{t2:.6f}\tkaolin obs:{t2:.6f}")
        body_sdf = self.query_canonical_sdf(smplx_canonical_sdf, canonical_body_bbox, cano_body_coords)
        # base_sdf = {'body':body_sdf}
        # face_sdf = self.query_canonical_sdf(smplx_canonical_sdf, canonical_body_bbox, cano_face_coords)
        # right_hand_sdf = self.query_canonical_sdf(smplx_canonical_sdf, canonical_body_bbox, cano_right_hand_coords)
        # left_hand_sdf = self.query_canonical_sdf(smplx_canonical_sdf, canonical_body_bbox, cano_left_hand_coords)
        
        bs, num_pts = cano_face_coords.shape[:2]
        face_sdf = torch.zeros([bs, num_pts, 1], device=device)
        face_sdf[cano_face_coords_mask] = body_sdf[cano_face_mask]
        
        bs, num_pts = cano_right_hand_coords.shape[:2]
        right_hand_sdf = torch.zeros([bs, num_pts, 1], device=device)
        right_hand_sdf[cano_right_hand_coords_mask] = body_sdf[cano_right_hand_mask]
        
        bs, num_pts = cano_left_hand_coords.shape[:2]
        left_hand_sdf = torch.zeros([bs, num_pts, 1], device=device)
        left_hand_sdf[cano_left_hand_coords_mask] = body_sdf[cano_left_hand_mask]
        
        base_sdf = {'body':body_sdf, 'face':face_sdf, 'right_hand':right_hand_sdf, 'left_hand':left_hand_sdf}
        
        if use_deformation:
            deformation_reg = {
                'body': coords_cano_offset[cano_body_coords_mask&(~cano_face_mask)&(~cano_right_hand_mask)&(~cano_left_hand_mask)].norm(dim=-1).mean(),
                'face': coords_cano_offset[cano_face_mask].norm(dim=-1).mean(),
                'hand': coords_cano_offset[cano_right_hand_mask|cano_left_hand_mask].norm(dim=-1).mean(),
                }
        else:
            deformation_reg = None
        
        return obs_mask, obs_coords_mask, obs_max_length, \
               cano_body_mask, cano_body_coords_mask, cano_body_max_length, cano_body_coords, \
               cano_face_mask, cano_face_coords_mask, cano_face_max_length, cano_face_coords, \
               cano_right_hand_mask, cano_right_hand_coords_mask, cano_right_hand_max_length, cano_right_hand_coords, \
               cano_left_hand_mask, cano_left_hand_coords_mask, cano_left_hand_max_length, cano_left_hand_coords, \
               cano_overlap_mask, cano_overlap_coords_mask, cano_overlap_max_length, cano_overlap_coords, \
               cano_overlap_rh_mask, cano_overlap_rh_coords_mask, cano_overlap_rh_max_length, cano_overlap_rh_coords, \
               cano_overlap_lh_mask, cano_overlap_lh_coords_mask, cano_overlap_lh_max_length, cano_overlap_lh_coords, \
               base_sdf, deformation_reg

    @staticmethod
    def filter_and_pad(coords, bbox, aux_bbox=None, coords_ori=None, bbox_ori=None):
        # filter out coords that are out of the bbox and pad the batch to same length
        device = coords.device
        batch_size, num_pts, _ = coords.shape
        if isinstance(bbox, list):
            mask = torch.zeros([batch_size, num_pts], dtype=torch.bool, device=coords.device)
            for _bbox in bbox:
                boxed_coords = 2 * (coords-_bbox[:, 0:1]) / (_bbox[:, 1:2]-_bbox[:, 0:1]) - 1
                mask = torch.logical_or(mask, boxed_coords.abs().amax(-1) <= 1)
        else:
            boxed_coords = 2 * (coords-bbox[:, 0:1]) / (bbox[:, 1:2]-bbox[:, 0:1]) - 1
            mask = boxed_coords.abs().amax(-1) <= 1
        
        if aux_bbox is not None:
            for _aux_bbox in aux_bbox:
                boxed_coords = 2 * (coords-_aux_bbox[:, 0:1]) / (_aux_bbox[:, 1:2]-_aux_bbox[:, 0:1]) - 1
                mask = torch.logical_or(mask, boxed_coords.abs().amax(-1) <= 1)

        max_length = torch.max(torch.sum(mask, 1))
        filtered_coords_mask = torch.arange(max_length, device=device).unsqueeze(0).repeat(batch_size, 1) < torch.sum(mask, 1, keepdim=True)
        filtered_coords = torch.zeros([batch_size, max_length, 3], dtype=torch.float32, device=device)
        filtered_coords[filtered_coords_mask] = coords[mask]
        filtered_coords_ori = None

        if coords_ori is not None and bbox_ori is not None:
            filtered_coords_ori = bbox_ori[:, 0:1].repeat(1, max_length, 1)
            filtered_coords_ori[filtered_coords_mask] = coords_ori[mask]

        return mask, filtered_coords_mask, max_length, filtered_coords, filtered_coords_ori

    def query_canonical_sdf(self, smplx_canonical_sdf, bbox, coords):
        coords_bbox = 2 * (coords-bbox[:, 0:1]) / (bbox[:, 1:2]-bbox[:, 0:1]) - 1
        sdf = torch.nn.functional.grid_sample(smplx_canonical_sdf, coords_bbox.unsqueeze(1).unsqueeze(1), padding_mode='border', align_corners=False)
        return sdf[:, 0, 0].permute(0, 2, 1).detach()

    def roi_detach(self, filtered_coords, filtered_face_coords, overlap_bbox):
        
        boxed_coords = 2 * (filtered_coords-overlap_bbox[:, 0:1]) / (overlap_bbox[:, 1:2]-overlap_bbox[:, 0:1]) - 1
        detach_mask = boxed_coords.abs().amax(-1) > 1
        
        boxed_coords = 2 * (filtered_face_coords-overlap_bbox[:, 0:1]) / (overlap_bbox[:, 1:2]-overlap_bbox[:, 0:1]) - 1
        detach_mask_face = boxed_coords.abs().amax(-1) > 1
                
        return detach_mask, detach_mask_face
    
    def unpose(self, coords, verts, ober2cano_transform, lbs_weights, device, dis_threshold=0.1):
        bs, nv = coords.shape[:2]
        coords_dist, coords_transform_inv = self.get_neighbs(
            coords, verts, ober2cano_transform.clone(), lbs_weights.clone(), device
        )
        coords_valid = torch.lt(coords_dist, dis_threshold).float()
        coords_unposed = batch_transform(coords_transform_inv, coords)
        return coords_unposed, coords_valid

    def get_neighbs(self, coords, verts, verts_transform_inv, lbs_weights, device, k_neigh=1, weight_std=0.1):
        bs, nv = verts.shape[:2]

        with torch.no_grad():
            # try:
            neighbs_dist, neighbs, _ = knn_points(coords, verts, K=k_neigh)

            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(verts[0, :, 0].cpu(), verts[0, :, 1].cpu(), verts[0, :, 2].cpu())
            # plt.savefig("tmp.jpg")

            neighbs_dist = torch.sqrt(neighbs_dist)
            # except:
                # # will cause OOM issue
                # coords_v = coords.unsqueeze(2) - verts.unsqueeze(1)
                # dist_ = torch.norm(coords_v, dim=-1, p=2)
                # neighbs_dist, neighbs = dist.topk(k_neigh, largest=False, dim=-1)
        
        weight_std2 = 2. * weight_std ** 2
        coords_neighbs_lbs_weight = lbs_weights[neighbs] # (bs, n_rays*K, k_neigh, 24)
        # (bs, n_rays*K, k_neigh)
        coords_neighbs_weight_conf = torch.exp(-torch.sum(torch.abs(coords_neighbs_lbs_weight - coords_neighbs_lbs_weight[..., 0:1, :]), dim=-1) / weight_std2)
        coords_neighbs_weight_conf = torch.gt(coords_neighbs_weight_conf, 0.9).float()  # why 0.9?
        coords_neighbs_weight = torch.exp(-neighbs_dist) # (bs, n_rays*K, k_neigh)
        coords_neighbs_weight = coords_neighbs_weight * coords_neighbs_weight_conf
        coords_neighbs_weight = coords_neighbs_weight / coords_neighbs_weight.sum(-1, keepdim=True) # (bs, n_rays*K, k_neigh)

        coords_neighbs_transform_inv = batch_index_select(verts_transform_inv, neighbs, device) # (bs, n_rays*K, k_neigh, 4, 4)
        coords_transform_inv = torch.sum(coords_neighbs_weight.unsqueeze(-1).unsqueeze(-1)*coords_neighbs_transform_inv, dim=2) # (bs, n_rays*K, 4, 4)

        coords_dist = torch.sum(coords_neighbs_weight*neighbs_dist, dim=2, keepdim=True) # (bs, n_rays*K, 1)
        return coords_dist, coords_transform_inv

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, depth_noise=False, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            if depth_noise:
                depth_delta = 1/(depth_resolution - 1)
                depths_coarse = depths_coarse + torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                if depth_noise:
                    depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                    depths_coarse = depths_coarse + torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                if depth_noise:
                    depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                    depths_coarse = depths_coarse + torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance, det=False):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance, det=det).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples