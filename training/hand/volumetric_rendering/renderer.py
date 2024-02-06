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

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
from pytorch3d.ops.knn import knn_points
from torch_utils import misc
from training.hand.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.hand.volumetric_rendering import math_utils
from training.hand.volumetric_rendering.utils import grid_sample
from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance

def get_eikonal_term(pts, sdf):
    eikonal_term = autograd.grad(outputs=sdf, inputs=pts,
                                 grad_outputs=torch.ones_like(sdf, requires_grad=False, device=pts.device),
                                 create_graph=True,
                                 retain_graph=True,
                                 only_inputs=True)[0]
    return torch.clamp(torch.nan_to_num((eikonal_term.norm(dim=-1) - 1) ** 2, 0.0), 0, 1e6)

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

def sample_from_planes(plane_axes, plane_features, coordinates, canonical_bbox=None, mode='bilinear', padding_mode='zeros', box_warp=None, obs_bbox=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.reshape(N*n_planes, C, H, W)

    # canonical_bbox = torch.tensor([[-0.2, 0.032, -0.15], [0.2, 0.4833, 0.15]]).reshape(1, 2, 3).float().to(plane_features.device)
    # coordinates = 2 * (coordinates-canonical_bbox[:, :1]) / (canonical_bbox[:, 1:2]-canonical_bbox[:, :1]) - 1
    # coordinates[..., 2] /= 3.0
    if canonical_bbox is not None:
        coordinates = 2 * (coordinates-canonical_bbox[:, :1]) / (canonical_bbox[:, 1:2]-canonical_bbox[:, :1]) - 1
        # # coordinates[..., 2] /= 3.0
        # coordinates *= 2.0
        # coordinates[..., 1] = (coordinates[..., 1] * 2 - 1)
    else:
        # coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds
        pass

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = grid_sample(plane_features, projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features
 
def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                   coordinates.reshape(batch_size, 1, 1, -1, n_dims))
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features

def dda(rays_o, rays_d, bbox_3D):
    # bbox_3D 1x2x3
    cam_dist = torch.norm(rays_o, dim=-1, keepdim=True)
    near = cam_dist[:, :1, 0] - 0.5
    far = cam_dist[:, :1, 0] + 0.5
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
    inds = inds + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    data = data.reshape(bs*nv, *data.shape[2:])
    return data[inds.long()]

class ImportanceRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes()

    def forward(self, planes, decoder, ray_origins, ray_directions, rendering_options,
        canonical_mapping_kwargs, return_sdf, return_eikonal, smplx_reg, 
        offset_network=None, density_reg=False, smplx_reg_full_space=False):

        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            if True:
                # cam_dist = torch.norm(ray_origins, dim=-1, keepdim=True)
                # ray_start = cam_dist - 0.5
                # ray_end = cam_dist + 0.5
                # depths_coarse_fix = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
            # else:
                ray_start, ray_end = dda(ray_origins, ray_directions, canonical_mapping_kwargs['obs_bbox_cam'])
                depths_coarse_dda = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse_dda.shape

        # Coarse Pass
        # sample_coordinates_fix = (ray_origins.unsqueeze(-2) + depths_coarse_fix * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_coordinates_dda = (ray_origins.unsqueeze(-2) + depths_coarse_dda * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(sample_coordinates_dda[0, ::40, 0].cpu(), sample_coordinates_dda[0, ::40, 1].cpu(), sample_coordinates_dda[0, ::40, 2].cpu(), alpha=0.5)
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

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(sample_coordinates_dda[0, :, 0].cpu(), sample_coordinates_dda[0, :, 1].cpu(), sample_coordinates_dda[0, :, 2].cpu(), alpha=0.5)
        # # zeros = torch.zeros_like(ray_origins)
        # # ax.plot(np.array([ray_origins[0, 0, 0].item(), zeros[0, 0, 0].item()]), np.array([ray_origins[0, 0, 1].item(), zeros[0, 0, 1].item()]), np.array([ray_origins[0, 0, 2].item(), zeros[0, 0, 2].item()]))
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.view_init(0, 0)
        # plt.savefig("tmp_yz_dda.jpg")
        # ax.view_init(90, 90)
        # plt.savefig("tmp_xy_dda.jpg")
        # ax.view_init(0, 90)
        # plt.savefig("tmp_xz_dda.jpg")

        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)
        
        sample_coordinates = sample_coordinates_dda
        depths_coarse = depths_coarse_dda

        out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options, offset_network=offset_network, return_sdf=return_sdf, return_eikonal=return_eikonal, smplx_reg_full_space=smplx_reg_full_space, **canonical_mapping_kwargs)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        sdf_coarse = out['sdf']
        eikonal_coarse = out["eikonal"]
        smplx_sdf = out['smplx_sdf']
        sdf_reg_weights = out['sdf_reg_weights']
        deformation_reg = out['deformation_reg']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options, offset_network=offset_network, return_sdf=return_sdf, return_eikonal=return_eikonal, smplx_reg_full_space=smplx_reg_full_space, **canonical_mapping_kwargs)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            eikonal_fine = out['eikonal']
            sdf_fine = out['sdf']
            if out['deformation_reg'] is not None:
                deformation_reg = (deformation_reg+out['deformation_reg'])/2
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        else:
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

        eikonal = None
        if eikonal_coarse is not None:
            if eikonal_fine is not None:
                eikonal = torch.cat((eikonal_fine, eikonal_coarse), -1)
            else:
                eikonal = eikonal_coarse
        # feature_samples, depth_samples, weights_samples, eikonal, smplx_sdf, sdf_reg_weights, coarse_sdf, deformation_reg
        return rgb_final, depth_final, weights.sum(2), eikonal, smplx_sdf, sdf_reg_weights, sdf_coarse, deformation_reg

    def run_model(self, planes, decoder, sample_coordinates, sample_directions, options,
        verts=None, verts_canonical=None, faces=None, obs_bbox=None, canonical_hand_bbox=None, canonical_body_bbox=None, triangles=None, 
        ober2cano_transform=None, lbs_weights=None, return_sdf=False, return_eikonal=False, smplx_canonical_sdf=None,
        only_density=False, density_reg=False, ws=None, ws_geo=None, offset_network=None, smplx_params=None, **kwargs):

        if return_eikonal and not only_density:
            sample_coordinates.requires_grad_(True)
        else:
            sample_coordinates.requires_grad_(False)

        num_batch, num_pts = sample_coordinates.shape[:2]
        device = sample_coordinates.device

        # TODO: compute face sdf to clamp decoder outputs
        sample_coordinates_cano, base_sdf, mask, coordinates_mask, max_length, deformation_reg = self.canonical_mapping(
            sample_coordinates, verts, verts_canonical, faces, obs_bbox, canonical_hand_bbox, canonical_body_bbox, 
            triangles, ober2cano_transform, lbs_weights, options.get('use_deformation', False), smplx_canonical_sdf=smplx_canonical_sdf, 
            offset_network=offset_network, density_reg=density_reg, smplx_params=smplx_params, ws=ws, ws_geo=ws_geo)
        # sample_coordinates_cano = sample_coordinates.clone()

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(sample_coordinates_cano[0, ::40, 0].detach().cpu().numpy(), sample_coordinates_cano[0, ::40, 1].detach().cpu().numpy(), sample_coordinates_cano[0, ::40, 2].detach().cpu().numpy(), alpha=0.5)
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

        sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates_cano, canonical_hand_bbox, padding_mode='zeros', box_warp=options['box_warp'], obs_bbox=obs_bbox)
        # sampled_features = sampled_features.reshape(sampled_features.shape[0], 3, max_length, -1)
        sampled_features = sampled_features.reshape(sampled_features.shape[0], 3, max_length, -1)

        out = decoder(sampled_features, sample_directions, base_sdf)

        rgb_final = torch.zeros((num_batch, num_pts, out['rgb'].shape[-1]), device=device)
        sdf_final = torch.zeros((num_batch, num_pts, out['sdf'].shape[-1]), device=device)
        rgb_final[mask] = out['rgb'][coordinates_mask]
        sdf_final[mask] = out['sdf'][coordinates_mask]
        out['rgb'] = rgb_final
        out['sdf'] = sdf_final
        sample_sdf = out['sdf']
        out['sigma'] = decoder.sdf2sigma(-sample_sdf)

        out['eikonal'] = None
        if return_eikonal and not only_density:
            out['eikonal'] = get_eikonal_term(sample_coordinates, sample_sdf)
        
        if not return_sdf and not only_density:
            out['sdf'] = None

        smplx_sdf = None
        sdf_reg_weights = None
        # if smplx_reg_full_space:
        #     verts_dis = torch.clamp(torch.sqrt(face_sdf.reshape(*out['sdf'].shape)**2), 1e-2, 1e3)
        #     sdf_reg_weights = torch.exp(-verts_dis**2/2*100)
        #     smplx_sdf = out['sdf']
        
        out['smplx_sdf'] = smplx_sdf
        out['sdf_reg_weights'] = sdf_reg_weights

        out['deformation_reg'] = None
        if options.get('use_deformation', False):
            out['deformation_reg'] = deformation_reg

        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        
        return out

    def canonical_mapping(
        self, coords, verts, verts_canonical, faces, obs_bbox, canonical_hand_bbox, canonical_body_bbox,
        triangles, ober2cano_transform, lbs_weights, use_deformation=False, 
        offset_network=None, density_reg=False, face_sdf_from_obs=True, 
        smplx_params=None, smplx_canonical_sdf=None, ws=None, ws_geo=None):
        """
        coordinates: N x Npts x 3
        """

        device = coords.device
        deformation_reg = None

        if not density_reg:
            # filter and mask points using observation bbox
            # mask, filtered_coords_mask, max_length, filtered_coords, _ = self.filter_and_pad(coords, obs_bbox)
            # filtered_coords_ori = filtered_coords.clone()
            filtered_coords = coords
            filtered_coords_cano, valid = self.unpose(filtered_coords, verts, ober2cano_transform, lbs_weights, device)

            if use_deformation:
                assert offset_network is not None
                filtered_coords_cano, coords_cano_offset = self.deformation(offset_network, filtered_coords_cano, filtered_coords, ws, ws_geo, smplx_params, canonical_hand_bbox)
                deformation_reg = coords_cano_offset.norm(dim=-1).mean()
            
            filtered_coords = filtered_coords_cano
            
            # mask, canonical_filtered_coords_mask, max_length, filtered_coords, filtered_coords_ori = self.filter_and_pad(filtered_coords, canonical_hand_bbox.repeat(coords.shape[0], 1, 1))    
                
        else:
            filtered_coords = coords
            filtered_coords_ori = filtered_coords.clone()
            
        canonical_filtered_coords_mask = torch.ones_like(filtered_coords[..., 0]).bool()
        mask = torch.ones_like(filtered_coords[..., 0]).bool()
        max_length = mask.shape[1]
    
        # if max_length == 0:
        #     print("Dry run...")
        #     filtered_coords = coords
        #     filtered_coords_ori = filtered_coords.clone()
        #     canonical_filtered_coords_mask = torch.ones_like(filtered_coords[..., 0]).bool()
        #     mask = torch.ones_like(filtered_coords[..., 0]).bool()
        #     max_length = mask.shape[1]

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(filtered_coords_cano[0, :, 0].detach().cpu(), filtered_coords_cano[0, :, 1].detach().cpu(), filtered_coords_cano[0, :, 2].detach().cpu())
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.view_init(0, 0)
        # plt.savefig("tmp_0.jpg")
        # ax.view_init(60, 90)
        # plt.savefig("tmp_90.jpg")

        # filter and mask pts using canonical global bbox (sample as triplane bbox warp)
        # if not density_reg:
        #     # canonical_mask = mask.clone()
        #     # canonical_filtered_coords_mask = filtered_coords_mask.clone()
        #     canonical_mask, canonical_filtered_coords_mask, max_length, filtered_coords, filtered_coords_ori = \
        #         self.filter_and_pad(filtered_coords, canonical_hand_bbox.repeat(coords.shape[0], 1, 1), filtered_coords_ori, obs_bbox)
        # else:
        #     canonical_mask = canonical_filtered_coords_mask = torch.ones_like(filtered_coords[..., 0]).bool()
        #     max_length = canonical_mask.shape[1]

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(filtered_coords[0, :, 0].detach().cpu(), filtered_coords[0, :, 1].detach().cpu(), filtered_coords[0, :, 2].detach().cpu())
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.view_init(0, 0)
        # plt.savefig("tmp_0.jpg")
        # ax.view_init(90, 90)
        # plt.savefig("tmp_90.jpg")
        
        # Combine two masks (make clone of the masks to avoid inplace replacement)
        # if not density_reg:
        #     new_mask = mask.clone()
        #     new_mask[mask] = canonical_mask[filtered_coords_mask]
        #     mask = new_mask

        #     new_canonical_mask = canonical_filtered_coords_mask.clone()
        #     new_canonical_mask[canonical_filtered_coords_mask] = filtered_coords_mask[canonical_mask]
        #     canonical_filtered_coords_mask = new_canonical_mask
        
        # else:
        #     mask = canonical_mask

        face_sdf = self.query_canonical_sdf(smplx_canonical_sdf, canonical_body_bbox, filtered_coords)
        # if not density_reg:
        #     canonical_filtered_coords_mask = filtered_coords_mask.clone() 
        # else:

        # Query SMPL SDF
        # if True:
        #     face_sdf = cal_sdf_batch(
        #         verts,
        #         faces.unsqueeze(0).repeat(verts.size(0), 1, 1), 
        #         triangles, 
        #         coords,
        #         device
        #     )[:, :max_length]
        # else:
        #     face_sdf = cal_sdf_batch(
        #         verts_canonical,
        #         faces.unsqueeze(0).repeat(verts_canonical.size(0), 1, 1), 
        #         triangles, 
        #         filtered_coords,
        #         device
        #     )[:, :max_length]

        return filtered_coords, face_sdf, mask, canonical_filtered_coords_mask, max_length, deformation_reg

    @staticmethod
    def filter_and_pad(coords, bbox, coords_ori=None, bbox_ori=None):
        # filter out coords that are out of the bbox and pad the batch to same length
        device = coords.device
        batch_size, num_pts, _ = coords.shape
        boxed_coords = 2 * (coords-bbox[:, 0:1]) / (bbox[:, 1:2]-bbox[:, 0:1]) - 1
        mask = boxed_coords.abs().amax(-1) <= 1
        max_length = torch.max(torch.sum(mask, 1))
        filtered_coords_mask = torch.arange(max_length, device=device).unsqueeze(0).repeat(batch_size, 1) < torch.sum(mask, 1, keepdim=True)
        filtered_coords = bbox[:, 0:1].repeat(1, max_length, 1)
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

    def deformation(self, offset_network, coords_cano, coords, ws=None, ws_geo=None, smpl_params=None, bbox=None):
        input_coords = torch.cat([coords_cano, coords],dim=-1)
        coords_cano_offset = offset_network(input_coords, ws, ws_geo, smpl_params)
        deformed_coords_cano = coords_cano + coords_cano_offset
        # replace with original pts if deformed out of canonical bbox
        bs, num_pts = coords.shape[:2]
        deform_mask = torch.zeros([bs, num_pts], dtype=torch.bool, device=coords.device)
        boxed_coords_deformed = 2 * (deformed_coords_cano-bbox[:,0:1]) / (bbox[:,1:2]-bbox[:,0:1]) - 1
        deform_mask = boxed_coords_deformed.abs().amax(-1) >= 1
        deformed_coords_cano[deform_mask] = coords_cano[deform_mask]
        return deformed_coords_cano, coords_cano_offset

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

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            # depth_delta = 1/(depth_resolution - 1)
            # depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                # depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                # depths_coarse += torch.rand_like(depths_coarse) * depth_delta.unsqueeze(-1)
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                # depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                # depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
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
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
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