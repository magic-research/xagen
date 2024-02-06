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
import torch
import torch.nn as nn
import numpy as np

from torch_utils import misc
from training.hand.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.hand.volumetric_rendering import math_utils
from training.hand.volumetric_rendering.utils import grid_sample, grid_sample_3d
import torch.autograd as autograd

from pytorch3d.ops.knn import knn_points
from pytorch3d.ops import add_points_features_to_volume_densities_features
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

def generate_meshgrids(canonical_bbox, grid_stride):
    i, j, k = torch.meshgrid(torch.linspace(canonical_bbox[0,0,0], canonical_bbox[0,1,0], int((canonical_bbox[0,1,0]-canonical_bbox[0,0,0])/grid_stride)),
                        torch.linspace(canonical_bbox[0,0,1], canonical_bbox[0,1,1], int((canonical_bbox[0,1,1]-canonical_bbox[0,0,1])/grid_stride)),
                        torch.linspace(canonical_bbox[0,0,2], canonical_bbox[0,1,2], int((canonical_bbox[0,1,2]-canonical_bbox[0,0,2])/grid_stride)),)
    meshgrids = torch.cat([i.reshape(-1,1),j.reshape(-1,1),k.reshape(-1,1)],dim=-1).unsqueeze(0)
    return meshgrids

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
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
    # return projections[..., :2]
    return projections[..., :3]

def sample_from_planes(plane_axes, plane_features, coordinates, canonical_bbox=None, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)
    if canonical_bbox is None:
        coordinates = (2 / box_warp) * coordinates # TODO: add specific box bounds
    else:
        coordinates = 2 * (coordinates-canonical_bbox[:, :1]) / (canonical_bbox[:, 1:2]-canonical_bbox[:, :1]) - 1
    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = grid_sample(plane_features, projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

def sample_from_volumes(plane_axes, plane_features, coordinates, canonical_bbox=None, mode='bilinear', padding_mode='zeros', box_warp=None, depth=4):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    C_0=C//depth

    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C_0, depth, H, W)
    if canonical_bbox is None:
        coordinates = (2 / box_warp) * coordinates # TODO: add specific box bounds
    else:
        coordinates = 2 * (coordinates-canonical_bbox[:, :1]) / (canonical_bbox[:, 1:2]-canonical_bbox[:, :1]) - 1
    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1).unsqueeze(1) # N,1,1,M,3
    output_features = grid_sample_3d(plane_features, projected_coordinates.float()).permute(0, 4, 3, 2, 1).reshape(N, n_planes, M, C_0)
    return output_features

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
    near = cam_dist[:, :1, 0] - 1.0
    far = cam_dist[:, :1, 0] + 1.0
    near = near.unsqueeze(-1) * torch.ones_like(rays_d[..., :1]).float()
    far = far.unsqueeze(-1) * torch.ones_like(rays_d[..., :1]).float()

    inv_ray_d = 1.0 / (rays_d + 1e-6) # 4x64x64x3
    t_min = (bbox_3D[:,:1] - rays_o) * inv_ray_d  # 4x64x64x3
    t_max = (bbox_3D[:,1:] - rays_o) * inv_ray_d
    t = torch.stack((t_min, t_max))  # 2x4x64x64x3
    t_min = torch.amax(torch.amin(t, dim=0), dim=-1, keepdim=True) # 4x64x64x1
    t_max = torch.amin(torch.amax(t, dim=0), dim=-1, keepdim=True)
    mask = t_min >= t_max
    t_min[mask] = near[mask]
    t_max[mask] = far[mask]
    return t_min, t_max
    
class ImportanceRenderer(torch.nn.Module):
    def __init__(self, canonical_bbox, grid_stride=0.02):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes()
        meshgrids = generate_meshgrids(canonical_bbox, grid_stride)
        self.register_buffer('meshgrids', meshgrids)
    def forward(
        self, planes, decoder, ray_origins, ray_directions, rendering_options, 
        canonical_mapping_kwargs, return_sdf, return_eikonal, smpl_reg, 
        offset_network=None, canonical_reg=False):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        # TODO: add these into arugments
        use_cam_dist = rendering_options.get('use_cam_dist', False)
        calc_eikonal_coarse = rendering_options.get('calc_eikonal_coarse', False)

        if use_cam_dist:
            cam_dist = torch.norm(ray_origins, dim=-1, keepdim=True)
            # ray_start = cam_dist - 1.0
            # ray_end = cam_dist + 1.0
            ray_start, ray_end = dda(ray_origins, ray_directions, canonical_mapping_kwargs['body_bbox'])
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
            rendering_options['box_warp'] = 4.0
        else:
            if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
                ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
                is_ray_valid = ray_end > ray_start
                if torch.any(is_ray_valid).item():
                    ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                    ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
                depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
            else:
                # Create stratified depth samples
                depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)
        eikonal = None
        if calc_eikonal_coarse:
            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options, return_eikonal=return_eikonal, return_sdf=return_sdf, offset_network=offset_network, **canonical_mapping_kwargs)
            eikonal_coarse = out['eikonal']
        else:
            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options, offset_network=offset_network, **canonical_mapping_kwargs)
            eikonal_coarse = None
        
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        # always return coarse level sdf for minsurf loss and visualize
        sdf_coarse = out['sdf']

        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options, return_eikonal=return_eikonal, return_sdf=return_sdf, offset_network=offset_network, **canonical_mapping_kwargs)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            eikonal = out['eikonal']
            
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
        
        smpl_sdf = None
        # FIXME: if we need a flag to set verts in obs or can spaces?
        if smpl_reg:
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
            smpl_sdf = out['sdf']

        return rgb_final, depth_final, weights.sum(2), eikonal, smpl_sdf, sdf_coarse

    def run_model(
        self, planes, decoder, sample_coordinates, sample_directions, options, 
        verts, verts_canonical, faces, body_bbox, canonical_bbox, triangles, 
        ober2cano_transform, cano2ober_transform, lbs_weights, joints_canonical, return_sdf=False, return_eikonal=False,
        only_density=False, canonical_reg=False, offset_network=None, use_observation_mapping=True):
        

        assert canonical_reg==False
        if return_eikonal and not only_density:
            sample_coordinates.requires_grad_(True)
        else:
            sample_coordinates.requires_grad_(False)

        num_batch, num_pts = sample_coordinates.shape[:2]
        device = sample_coordinates.device

        sample_coordinates_canonical, body_sdf, mask, coordinates_mask, max_length, filtered_sample_coordinates_obs = self.canonical_mapping(
            sample_coordinates, verts, verts_canonical, faces, body_bbox, canonical_bbox,
            triangles, ober2cano_transform, lbs_weights, options.get('use_deformation', False), 
            offset_network=offset_network, canonical_reg=canonical_reg)



        sampled_features = sample_from_volumes(self.plane_axes, planes, sample_coordinates_canonical, canonical_bbox, padding_mode='zeros', box_warp=options['box_warp'])
        sampled_features = sampled_features.reshape(sampled_features.shape[0], 3, max_length, -1)

        if use_observation_mapping:
            # FIXME: add more args here for grid_stride, boxwarp, dist_thresh
            sampled_features_ob, body_sdf_ob, mask_ob, coordinates_mask_ob = self.observation_mapping(
                planes, filtered_sample_coordinates_obs, verts, verts_canonical, faces, body_bbox, canonical_bbox,
                triangles, cano2ober_transform, lbs_weights, joints_canonical, device)
            body_sdf = body_sdf_ob
            sampled_features = (sampled_features+sampled_features_ob)/2


        out = decoder(sampled_features, sample_directions, body_sdf)
        rgb_final = torch.zeros((num_batch, num_pts, out['rgb'].shape[-1]), device=device)
        sdf_final = torch.zeros((num_batch, num_pts, out['sdf'].shape[-1]), device=device) + 30  # TODO: set a bg sdf for bg (30?)
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

        if options.get('density_noise', 0) > 0:
            out['sigma'] = out['sigma'] + torch.randn_like(out['sigma']) * options['density_noise']
        return out

    def sample_frostum_grid(self, ray_origins, ray_directions, rendering_options):

        cam_dist = torch.norm(ray_origins, dim=-1, keepdim=True)
        ray_start = cam_dist - 1.0
        ray_end = cam_dist + 1.0
        # ray_start, ray_end = dda(ray_origins, ray_directions, canonical_mapping_kwargs['body_bbox'])
        depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)
        return sample_coordinates, sample_directions

    def deformation(self, offset_network, coords):
        deformed_coords = offset_network(coords)
        return deformed_coords

    def observation_mapping(self, canonical_volume_feat ,coords, verts, verts_canonical,
        faces, body_bbox, canonical_bbox, triangles, cano2ober_transform, lbs_weights, joints_canonical,
        device, dist_thresh=0.25, box_warp=1.5, depth=4):
        # warp canonical volume to observation volume based on forward skinning

        N, n_planes, C, H, W = canonical_volume_feat.shape
        C_0=C//4
        
        # filter grids
        dist = torch.cdist(self.meshgrids, joints_canonical[:1,:23])
        mask = dist.min(2)[0]<dist_thresh
        filtered_grids = self.meshgrids[mask].unsqueeze(0).expand(N,-1,-1)
        
        grids_canonical = 2 * (filtered_grids-canonical_bbox[:, :1]) / (canonical_bbox[:, 1:2]-canonical_bbox[:, :1]) - 1
        
        # plane_features = torch.rand(1*3,32,4,256,256).to(device) # B*n_planes, C_0, depth, H,W

        projected_grids_canonical = project_onto_planes(self.plane_axes, grids_canonical).unsqueeze(1).unsqueeze(1)
        output_features = grid_sample_3d(canonical_volume_feat.reshape(N*n_planes, C_0, depth, H, W), projected_grids_canonical).permute(0, 4, 3, 2, 1).reshape(N*n_planes, -1, C_0)

        _, grids_transform_inv = self.get_neighbs(
            filtered_grids, verts_canonical, cano2ober_transform.clone(), lbs_weights, device
        )

        grids_posed = batch_transform(grids_transform_inv, filtered_grids)/box_warp
        projected_grids_posed = project_onto_planes(self.plane_axes, grids_posed)

        volume_density = torch.zeros(N*n_planes,1,depth, H,W).to(device)
        volume_feature = torch.zeros(N*n_planes,C_0,depth, H,W).to(device)
        obs_plane_features,obs_plane_density = add_points_features_to_volume_densities_features(projected_grids_posed, output_features, volume_density,volume_feature, "trilinear", 1, _python=True)
        


        mask, filtered_coords_mask, max_length, filtered_coords = self.filter_and_pad(coords, body_bbox)
        body_sdf_ober = cal_sdf_batch(
            verts,
            faces.unsqueeze(0).repeat(verts_canonical.size(0), 1, 1), 
            triangles, 
            filtered_coords,
            device
        )[:, :max_length].reshape(-1, 1)

        sampled_features = sample_from_volumes(self.plane_axes, obs_plane_features.reshape(N, n_planes, C, H, W), filtered_coords, None, padding_mode='zeros', box_warp=2*box_warp)
        sampled_features = sampled_features.reshape(sampled_features.shape[0], 3, max_length, -1)
        return sampled_features, body_sdf_ober, mask, filtered_coords_mask


    def canonical_mapping(
        self, coords, verts, verts_canonical, faces, body_bbox, canonical_bbox, 
        triangles, ober2cano_transform, lbs_weights, use_deformation=False, 
        offset_network=None, canonical_reg=False):
        """
        coordinates: N x Npts x 3
        """

        device = coords.device

        if not canonical_reg:
            # filter and mask points using observation bbox
            mask, filtered_coords_mask, max_length, filtered_coords = self.filter_and_pad(coords, body_bbox)
            filtered_coords_cano, valid = self.unpose(filtered_coords, verts, ober2cano_transform, lbs_weights, device)

            if use_deformation:
                assert offset_network is not None
                filtered_coords_offset = self.deformation(offset_network, filtered_coords)
                filtered_coords = filtered_coords_cano + filtered_coords_offset
            else:
                filtered_coords = filtered_coords_cano
        else:
            filtered_coords = coords

        # filter and mask pts using canonical global bbox (sample as triplane bbox warp)
        canonical_mask, canonical_filtered_coords_mask, max_length, filtered_coords = \
            self.filter_and_pad(filtered_coords, canonical_bbox.repeat(coords.shape[0], 1, 1))
        

        # Combine two masks (make clone of the masks to avoid inplace replacement)
        if not canonical_reg:
            new_mask = mask.clone()
            new_mask[mask] = canonical_mask[filtered_coords_mask]
            mask = new_mask

            new_canonical_mask = canonical_filtered_coords_mask.clone()
            new_canonical_mask[canonical_filtered_coords_mask] = filtered_coords_mask[canonical_mask]
            canonical_filtered_coords_mask = new_canonical_mask


        else:
            mask = canonical_mask

        filtered_coords_obs = body_bbox[:, 0:1].repeat(1, max_length, 1)
        filtered_coords_obs[canonical_filtered_coords_mask] = coords[mask]
        # Query SMPL SDF
        # with torch.no_grad():
            # get SDF in canonical space
            # try:
        body_sdf = cal_sdf_batch(
            verts_canonical,
            faces.unsqueeze(0).repeat(verts_canonical.size(0), 1, 1), 
            triangles, 
            filtered_coords,
            device
        )[:, :max_length].reshape(-1, 1)

        return filtered_coords, body_sdf, mask, canonical_filtered_coords_mask, max_length, filtered_coords_obs

    @staticmethod
    def filter_and_pad(coords, bbox):
        # filter out coords that are out of the bbox and pad the batch to same length
        device = coords.device
        batch_size, num_pts, _ = coords.shape
        boxed_coords = 2 * (coords-bbox[:, 0:1]) / (bbox[:, 1:2]-bbox[:, 0:1]) - 1
        mask = boxed_coords.abs().amax(-1) <= 1
        max_length = torch.max(torch.sum(mask, 1))
        filtered_coords_mask = torch.arange(max_length, device=device).unsqueeze(0).repeat(batch_size, 1) < torch.sum(mask, 1, keepdim=True)
        filtered_coords = bbox[:, 0:1].repeat(1, max_length, 1)
        filtered_coords[filtered_coords_mask] = coords[mask]
        return mask, filtered_coords_mask, max_length, filtered_coords

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
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse = depths_coarse + torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse = depths_coarse + torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                depths_coarse = depths_coarse + torch.rand_like(depths_coarse) * depth_delta

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