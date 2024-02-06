# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import einops
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from torchvision.ops import roi_align

import dnnlib
# import legacy

from torch_utils import persistence, misc
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.face.networks_stylegan2 import Generator as StyleGAN2BackboneFace
from training.hand.networks_stylegan2 import Generator as StyleGAN2BackboneHand
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.face.volumetric_rendering.renderer import ImportanceRenderer as ImportanceRendererFace
from training.hand.volumetric_rendering.renderer import ImportanceRenderer as ImportanceRendererHand
from training.face.superresolution import SuperresolutionHybrid4XDC as SuperresolutionFace
from training.hand.superresolution import SuperresolutionHybrid4XDC as SuperresolutionHand
from training.volumetric_rendering.ray_sampler import RaySampler, AvatarGenRaySampler, camera2world_dry_run
from training.volumetric_rendering.utils import get_canonical_pose, world2cam_transform, project
from smplx.SMPLX import SMPLX, LEFT_HAND_INDEX, RIGHT_HAND_INDEX
# from deprec_smplxx import create

@misc.profiled_function
def face_vertices(vertices, faces, device):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """

    bs, nv = vertices.shape[:2]
    _, nf = faces.shape[:2]
    faces = faces + (torch.arange(bs, dtype=torch.int32, device=device) * nv)[:, None, None]
    vertices = vertices.reshape((bs*nv, vertices.shape[-1]))
    return vertices[faces.long()]

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        is_ema = False,
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer_body = ImportanceRenderer()
        self.renderer_face = ImportanceRendererFace()
        self.renderer_hand = ImportanceRendererHand()
        self.ray_sampler = AvatarGenRaySampler()
        
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=[512, 512], img_channels=48*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        # self.backbone_face = StyleGAN2BackboneFace(z_dim, c_dim, w_dim, img_resolution=128, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        # self.backbone_hand = StyleGAN2BackboneHand(z_dim, c_dim, w_dim, img_resolution=128, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32}, sigmoid_beta=rendering_kwargs['sigmoid_beta'])
        # self.decoder_face = AvatarGenOSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32}, sigmoid_beta=0.1)
        # self.decoder_hand = AvatarGenOSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32}, sigmoid_beta=0.1)
        
        self.superresolution_body = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.superresolution_face = SuperresolutionFace(channels=32, img_resolution=[256, 256], sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.superresolution_hand = SuperresolutionHand(channels=32, img_resolution=[256, 256], sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)

        self.neural_rendering_resolution = 256
        self.neural_rendering_resolution_face = 32
        self.neural_rendering_resolution_hand = 32
        self.rendering_kwargs = rendering_kwargs
        self.cam_dim = 25
        self.use_deformation = rendering_kwargs['use_deformation']
        self.canonical_reg = rendering_kwargs['canonical_reg']
        self.part_disc = rendering_kwargs.get('part_disc', False)
        self.backbone_face_pkl = rendering_kwargs.get('backbone_face_pkl', None)
        self.backbone_hand_pkl = rendering_kwargs.get('backbone_hand_pkl', None)
        self.backbone_pkl = rendering_kwargs.get('backbone_body_pkl', None)
        self.smplx_reg_full_space = rendering_kwargs.get('smplx_reg_full_space', False)
        
        if self.use_deformation:
            self.offset_network = OffsetNetwork(depth=6, width=64, freqs_xyz=8, scale=5e-2, use_pose_cond=False, scale_type='clamp')
        else:
            self.offset_network = None

        self.body_model = SMPLX()
        self.FLAME_INDEX = np.load("smplx/assets/SMPL-X__FLAME_vertex_ids.npy").tolist()
        self.RIGHT_HAND_INDEX = joblib.load('smplx/assets/MANO_SMPLX_vertex_ids.pkl')['right_hand']
        # Define Da-Pose
        pose_canonical = self.body_model.get_canonical_pose(use_da_pose=True)
        lbs_weights = self.body_model.lbs_weights
        faces = self.body_model.faces_tensor
        smplx_canonical_sdf = joblib.load('smplx/assets/smplx_canonical_body_sdf.pkl')
        
        # NOTE: shrinked canonical body and and bboxes
        canonical_body_bbox = torch.tensor([[-0.9554, -1.5118, -0.2120], [ 0.9554, 0.7406, 0.2011]]).reshape(1,2,3).float()
        canonical_face_bbox = torch.tensor([[-0.2, 0.032, -0.18], [0.2, 0.4833, 0.18]]).reshape(1, 2, 3).float()
        canonical_hand_bbox = torch.tensor([[-0.9554, -0.0285, -0.2266], [-0.5797,  0.0921,  0.1172]]).reshape(1, 2, 3).float()
        canonical_body_filter_bbox = torch.tensor([[-0.72, -1.5118, -0.212], [ 0.72, 0.2, 0.2011]]).reshape(1,2,3).float()
        canonical_face_filter_bbox = torch.tensor([[-0.15, 0.032, -0.212], [0.15, 0.45, 0.2011]]).reshape(1, 2, 3).float()
        canonical_right_hand_filter_bbox = torch.tensor([[-0.9554, -0.0285, -0.212], [-0.5797, 0.0921, 0.2011]]).reshape(1, 2, 3).float()
        canonical_left_hand_filter_bbox = torch.tensor([[0.9554, -0.0285, -0.212], [0.5797, 0.0921, 0.2011]]).reshape(1, 2, 3).float()
        overlap_head_bbox = torch.tensor([[-0.15, 0.032, -0.212], [0.15, 0.16, 0.2011]]).reshape(1, 2, 3).float()
        overlap_right_hand_bbox = torch.tensor([[-0.72, -0.0285, -0.212], [-0.5797, 0.0921, 0.2011]]).reshape(1, 2, 3).float()
        overlap_left_hand_bbox = torch.tensor([[0.72, -0.0285, -0.212], [0.5797, 0.0921, 0.2011]]).reshape(1, 2, 3).float()
        canonical_upper_spine_bbox = torch.tensor([[-0.15, -0.1, -0.212], [0.15, 0.16, 0.2011]]).reshape(1, 2, 3).float()
        canonical_right_arm_bbox = torch.tensor([[-0.72, -0.0285, -0.212], [-0.45, 0.0921, 0.2011]]).reshape(1, 2, 3).float()
        canonical_left_arm_bbox = torch.tensor([[0.72, -0.0285, -0.212], [0.45, 0.0921, 0.2011]]).reshape(1, 2, 3).float()
        
        self.register_buffer('canonical_body_bbox', canonical_body_bbox)
        self.register_buffer('canonical_face_bbox', canonical_face_bbox)
        self.register_buffer('canonical_hand_bbox', canonical_hand_bbox)
        self.register_buffer('canonical_body_filter_bbox', canonical_body_filter_bbox)
        self.register_buffer('canonical_face_filter_bbox', canonical_face_filter_bbox)
        self.register_buffer('canonical_right_hand_filter_bbox', canonical_right_hand_filter_bbox)
        self.register_buffer('canonical_left_hand_filter_bbox', canonical_left_hand_filter_bbox)
        self.register_buffer('overlap_head_bbox', overlap_head_bbox)
        self.register_buffer('overlap_right_hand_bbox', overlap_right_hand_bbox)
        self.register_buffer('overlap_left_hand_bbox', overlap_left_hand_bbox)
        self.register_buffer('canonical_upper_spine_bbox', canonical_upper_spine_bbox)
        self.register_buffer('canonical_right_arm_bbox', canonical_right_arm_bbox)
        self.register_buffer('canonical_left_arm_bbox', canonical_left_arm_bbox)
        self.register_buffer('pose_canonical', pose_canonical)
        self.register_buffer('lbs_weights', lbs_weights)
        self.register_buffer('faces', faces)
        self.register_buffer('smplx_canonical_sdf', smplx_canonical_sdf)
        
        self.canonical_body_fine_grained_filter_bboxes = [
            torch.tensor([[-0.72, -0.08, -0.2], [ 0.72, 0.2, 0.2]]).reshape(1,2,3).float(),
            
            torch.tensor([[-0.25, -0.5, -0.2], [ 0.25, 0, 0.2]]).reshape(1,2,3).float(),
            
            torch.tensor([[-0.45, -0.75, -0.2], [ 0, -0.45, 0.2]]).reshape(1,2,3).float(),
            torch.tensor([[0, -0.75, -0.2], [ 0.45, -0.45, 0.2]]).reshape(1,2,3).float(),
            
            torch.tensor([[-0.52, -1.0, -0.2], [-0.1, -0.7, 0.2]]).reshape(1,2,3).float(),
            torch.tensor([[0.1, -1.0, -0.2], [0.52, -0.7, 0.2]]).reshape(1,2,3).float(),
            
            torch.tensor([[-0.72, -1.35, -0.2], [0.25, -0.95, 0.2]]).reshape(1,2,3).float(),
            torch.tensor([[-0.25, -1.35, -0.2], [0.72, -0.95, 0.2]]).reshape(1,2,3).float()
        ]

        with torch.no_grad():
            # Canonical Space
            # TODO: check if we need to use a fixed shape for canonical space
            body_model_params_canonical = {'full_pose': self.pose_canonical.unsqueeze(0).repeat(1, 1, 1, 1)}
            verts_canonical, _, _, verts_transform_canonical, shape_offsets_canonical, pose_offsets_canonical = self.body_model(
                                            **body_model_params_canonical, return_transform=True, return_offsets=True)
        self.register_buffer('verts_canonical', verts_canonical)
        self.register_buffer('verts_transform_canonical', verts_transform_canonical)
        self.register_buffer('shape_offsets_canonical', shape_offsets_canonical)
        self.register_buffer('pose_offsets_canonical', pose_offsets_canonical)

        self._last_body_planes = None
        self._last_face_planes = None
        self._last_hand_planes = None
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)  

    def mapping_geo(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping_geo(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)  

    # def mapping_face(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
    #     if self.rendering_kwargs['c_gen_conditioning_zero']:
    #         c = torch.zeros_like(c)
    #     return self.backbone_face.mapping(z, c[:, :self.cam_dim] * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    # def mapping_hand(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
    #     if self.rendering_kwargs['c_gen_conditioning_zero']:
    #         c = torch.zeros_like(c)
    #     return self.backbone_hand.mapping(z, c[:, :self.cam_dim] * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def calc_ober2cano_transform(
        self, verts_transform, verts_transform_canonical, 
        shape_offsets, shape_offsets_canonical,
        pose_offsets, pose_offsets_canonical):

        ober2cano_transform = torch.inverse(verts_transform).clone()
        ober2cano_transform[..., :3, 3] = ober2cano_transform[..., :3, 3] + (shape_offsets_canonical - shape_offsets)
        ober2cano_transform[..., :3, 3] = ober2cano_transform[..., :3, 3] + (pose_offsets_canonical - pose_offsets)
        ober2cano_transform = torch.matmul(verts_transform_canonical, ober2cano_transform)
        return ober2cano_transform

    def get_canonical_mapping_joints(self, c):
        c_smplx = c[:, self.cam_dim:]
        c_betas = c_smplx[:, :200]
        c_expressions = c_smplx[:, 200:250]
        c_poses = c_smplx[:, 250:]

        with torch.no_grad():
            model_params = {'body_pose':c_poses[:, 3:66], 'jaw_pose':c_poses[:, 66:69], 'eye_pose':torch.zeros_like(c_poses[:, 0:6]),
                            'left_hand_pose':c_poses[:, 69:114], 'right_hand_pose': c_poses[:, 114:], 'shape_params': c_betas, 'expression_params':c_expressions}
            verts, _, joints, verts_transform, shape_offsets, pose_offsets = self.body_model(**model_params, pose2rot=True, return_transform=True, return_offsets=True)
            canonical_mapping_kwargs = {
                'joints': joints
            }
            return canonical_mapping_kwargs

    def get_canonical_mapping_info(self, c, mode):
        assert mode in ["body", "face", "hand"]
        c_cam = c[:, :self.cam_dim]
        c_smplx = c[:, self.cam_dim:]

        c_betas = c_smplx[:, :200]
        c_expressions = c_smplx[:, 200:250]
        c_poses = c_smplx[:, 250:]
        device = c_cam.device
        batch_size = c_cam.shape[0]

        with torch.no_grad():

            # Observation Space
            model_params = {'body_pose':c_poses[:, 3:66], 'jaw_pose':c_poses[:, 66:69], 'eye_pose':torch.zeros_like(c_poses[:, 0:6]),
                            'left_hand_pose':c_poses[:, 69:114], 'right_hand_pose': c_poses[:, 114:], 'shape_params': c_betas, 'expression_params':c_expressions}
            if mode in ["face", "hand"]:
                model_params['body_pose'] = torch.zeros_like(c_poses[:, 3:66])
            verts, _, joints, verts_transform, shape_offsets, pose_offsets = self.body_model(**model_params, pose2rot=True, return_transform=True, return_offsets=True)
            triangles = face_vertices(self.verts_canonical.repeat(batch_size, 1, 1), self.faces, device)
            
            # Compute obs bbox
            if mode == "body":
                vert_indices = torch.arange(verts.shape[1])
            elif mode == "face":
                vert_indices = self.FLAME_INDEX 
            elif mode == "hand":
                vert_indices = self.RIGHT_HAND_INDEX

            # Compute transformation from observation to canonical space
            ober2cano_transform = self.calc_ober2cano_transform(
                verts_transform[:, vert_indices], self.verts_transform_canonical.repeat(c_poses.shape[0], 1, 1, 1)[:, vert_indices],
                shape_offsets[:, vert_indices], self.shape_offsets_canonical.repeat(c_poses.shape[0], 1, 1)[:, vert_indices],
                pose_offsets[:, vert_indices], self.pose_offsets_canonical.repeat(c_poses.shape[0], 1, 1)[:, vert_indices]
            )
            
            extrinsics = torch.linalg.inv(c_cam[:, :16].reshape(-1, 4, 4))
            R = extrinsics[:, :3, :3]
            t = extrinsics[:, :3, 3].unsqueeze(1)
            verts_cam = world2cam_transform(verts, R, t)
            vert_min = torch.amin(verts_cam[:, vert_indices], dim=1)
            vert_max = torch.amax(verts_cam[:, vert_indices], dim=1)
            # extend box by a pre-defined ratio
            ratio = 0.2  # TODO: set as an arugment
            obs_bbox_cam = torch.stack([(1 + ratio) * vert_min - ratio * vert_max, (1 + ratio) * vert_max - ratio * vert_min], dim=1)  # Bx2x3

            vert_min = torch.amin(verts[:, vert_indices], dim=1)
            vert_max = torch.amax(verts[:, vert_indices], dim=1)
            # extend box by a pre-defined ratio
            ratio = 0.2  # TODO: set as an arugment
            obs_bbox = torch.stack([(1 + ratio) * vert_min - ratio * vert_max, (1 + ratio) * vert_max - ratio * vert_min], dim=1)  # Bx2x3

            canonical_aux_bboxes = {
                'canonical_body_filter_bbox': self.canonical_body_filter_bbox, 
                'canonical_face_filter_bbox': self.canonical_face_filter_bbox, 
                'canonical_right_hand_filter_bbox': self.canonical_right_hand_filter_bbox, 
                'canonical_left_hand_filter_bbox': self.canonical_left_hand_filter_bbox, 
                'overlap_head_bbox': self.overlap_head_bbox, 
                'overlap_right_hand_bbox': self.overlap_right_hand_bbox, 
                'overlap_left_hand_bbox': self.overlap_left_hand_bbox, 
                'canonical_upper_spine_bbox': self.canonical_upper_spine_bbox, 
                'canonical_right_arm_bbox': self.canonical_right_arm_bbox, 
                'canonical_left_arm_bbox': self.canonical_left_arm_bbox,
            }
            canonical_aux_bboxes = {k:v.repeat(batch_size, 1, 1) for k,v in canonical_aux_bboxes.items()}
            canonical_aux_bboxes['canonical_body_fine_grained_filter_bboxes'] = [box.repeat(batch_size, 1, 1).to(self.canonical_body_filter_bbox.device) for box in self.canonical_body_fine_grained_filter_bboxes]
            
            canonical_mapping_kwargs = {
                'obs_bbox': obs_bbox,
                'obs_bbox_cam': obs_bbox_cam,
                'ober2cano_transform': ober2cano_transform,
                'verts': verts[:, vert_indices],
                'verts_canonical': self.verts_canonical.repeat(batch_size, 1, 1)[:, vert_indices],
                'faces': self.faces,
                'triangles': triangles,
                'joints': joints, 
                'lbs_weights': self.lbs_weights[vert_indices],
                'canonical_body_bbox': self.canonical_body_bbox,
                'canonical_face_bbox': self.canonical_face_bbox,
                'canonical_hand_bbox': self.canonical_hand_bbox,
                'canonical_aux_bboxes': canonical_aux_bboxes, 
                'smplx_params': torch.cat((c_betas, c_expressions, c_poses), dim=-1),
                'smplx_canonical_sdf': self.smplx_canonical_sdf.repeat(batch_size, 1, 1, 1, 1)
            }

            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(self.verts_canonical[0, :, 0].detach().cpu(), self.verts_canonical[0, :, 1].detach().cpu(), self.verts_canonical[0, :, 2].detach().cpu())
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            # ax.view_init(0, 0)
            # plt.savefig("tmp_yz_fix.jpg")
            # ax.view_init(90, 90)
            # plt.savefig("tmp_xy_fix.jpg")
            # ax.view_init(0, 90)
            # plt.savefig("tmp_xz_fix.jpg")

            return canonical_mapping_kwargs

    def synthesis(self, ws, ws_geo, c_body, c_face, c_hand, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, return_eikonal=False, smplx_reg=False, **synthesis_kwargs):
        if neural_rendering_resolution is not None:
            self.neural_rendering_resolution = neural_rendering_resolution
            self.neural_rendering_resolution_face = int(neural_rendering_resolution / 8)
            self.neural_rendering_resolution_hand = int(neural_rendering_resolution / 8)
        
        canonical_mapping_kwargs_body = self.get_canonical_mapping_info(c_body, mode="body")
        canonical_mapping_kwargs_face = self.get_canonical_mapping_info(c_face, mode="face")
        canonical_mapping_kwargs_hand = self.get_canonical_mapping_info(c_hand, mode="hand")
        
        H = self.neural_rendering_resolution
        W = self.neural_rendering_resolution // 2
        
        canonical_mapping_kwargs_body['ws'] = ws
        canonical_mapping_kwargs_face['ws'] = ws
        canonical_mapping_kwargs_hand['ws'] = ws
        canonical_mapping_kwargs_body['ws_geo'] = ws_geo
        canonical_mapping_kwargs_face['ws_geo'] = ws_geo
        canonical_mapping_kwargs_hand['ws_geo'] = ws_geo
        
        # Create a batch of rays for volume rendering
        ray_origins_body, ray_directions_body = self.ray_sampler(c_body[:, :16].view(-1, 4, 4), c_body[:, 16:25].view(-1, 3, 3), [H, W])
        ray_origins_face, ray_directions_face = self.ray_sampler(c_face[:, :16].view(-1, 4, 4), c_face[:, 16:25].view(-1, 3, 3), [self.neural_rendering_resolution_face, self.neural_rendering_resolution_face])
        ray_origins_hand, ray_directions_hand = self.ray_sampler(c_hand[:, :16].view(-1, 4, 4), c_hand[:, 16:25].view(-1, 3, 3), [self.neural_rendering_resolution_hand, self.neural_rendering_resolution_hand])

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins_body.shape
        if use_cached_backbone and self._last_body_planes is not None:
            body_planes = self._last_body_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
            body_planes = planes[:, :32*3].reshape(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
            face_planes = planes[:, 32*3:40*3].reshape(len(planes), 3, 32, planes.shape[-2]//2, planes.shape[-1]//2)
            hand_planes = planes[:, 40*3:].reshape(len(planes), 3, 32, planes.shape[-2]//2, planes.shape[-1]//2)
        if cache_backbone:
            self._last_body_planes = body_planes
            self._last_face_planes = face_planes
            self._last_hand_planes = hand_planes
        
        # Perform volume rendering
        feature_samples, depth_samples, weights_samples, eikonal, smplx_sdf, \
        sdf_reg_weights, coarse_sdf, deformation_reg, overlap_features_fine = self.renderer_body(body_planes, face_planes, hand_planes, \
            self.decoder, self.decoder, self.decoder, ray_origins_body, ray_directions_body, self.rendering_kwargs, canonical_mapping_kwargs_body, \
            return_eikonal=return_eikonal, return_sdf=True, smplx_reg=smplx_reg, offset_network=self.offset_network, \
            canonical_reg=self.canonical_reg, smplx_reg_full_space=self.smplx_reg_full_space)
        
        feature_samples_face, depth_samples_face, weights_samples_face, eikonal_face, _, _, coarse_sdf_face, deformation_reg_face = self.renderer_face(face_planes, self.decoder, ray_origins_face, ray_directions_face, self.rendering_kwargs, canonical_mapping_kwargs_face, 
                                                                                                                        return_eikonal=False, return_sdf=True, smplx_reg=False, offset_network=self.offset_network, 
                                                                                                                        density_reg=None, smplx_reg_full_space=None)
        
        feature_samples_hand, depth_samples_hand, weights_samples_hand, eikonal_hand, _, _, coarse_sdf_hand, deformation_reg_hand = self.renderer_hand(hand_planes, self.decoder, ray_origins_hand, ray_directions_hand, self.rendering_kwargs, canonical_mapping_kwargs_hand, 
                                                                                                                        return_eikonal=False, return_sdf=True, smplx_reg=False, offset_network=self.offset_network, 
                                                                                                                        density_reg=None, smplx_reg_full_space=None)

        # Reshape into 'raw' neural-rendered image
        C = feature_samples.shape[-1]
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, C, H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        feature_image_face = feature_samples_face.permute(0, 2, 1).reshape(N, C, self.neural_rendering_resolution_face, self.neural_rendering_resolution_face).contiguous()
        depth_image_face = depth_samples_face.permute(0, 2, 1).reshape(N, 1, self.neural_rendering_resolution_face, self.neural_rendering_resolution_face)
        feature_image_hand = feature_samples_hand.permute(0, 2, 1).reshape(N, C, self.neural_rendering_resolution_hand, self.neural_rendering_resolution_hand).contiguous()
        depth_image_hand = depth_samples_hand.permute(0, 2, 1).reshape(N, 1, self.neural_rendering_resolution_hand, self.neural_rendering_resolution_hand)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution_body(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        rgb_image_face = feature_image_face[:, :3]
        sr_image_face = self.superresolution_face(rgb_image_face, feature_image_face, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        rgb_image_hand = feature_image_hand[:, :3]
        sr_image_hand = self.superresolution_hand(rgb_image_hand, feature_image_hand, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})    

        if deformation_reg is not None:
            deformation_reg['face'] = 0.5 * (deformation_reg['face'] + deformation_reg_face)
            deformation_reg['hand'] = 0.5 * (deformation_reg['hand'] + deformation_reg_hand)
            
        out = {
            'image': sr_image, 
            'image_raw': rgb_image, 
            'image_depth': depth_image, 
            'image_face': sr_image_face, 
            'image_raw_face': rgb_image_face, 
            'image_depth_face': depth_image_face,
            'image_hand': sr_image_hand, 
            'image_raw_hand': rgb_image_hand, 
            'image_depth_hand': depth_image_hand,
            'eikonal': eikonal, 
            'eikonal_face': eikonal_face, 
            'eikonal_hand': eikonal_hand, 
            'smplx_sdf': smplx_sdf, 
            'coarse_sdf': coarse_sdf, 
            'coarse_sdf_face': coarse_sdf_face, 
            'coarse_sdf_hand': coarse_sdf_hand, 
            'sdf_reg_weights': sdf_reg_weights, 
            'deformation_reg': deformation_reg,
            'overlap_features_fine': None,
        }

        if self.part_disc:
            image_part_face, image_part_left_hand, image_part_right_hand = self.get_part(sr_image, c_body, canonical_mapping_kwargs_body)
            out['image_part_face'] = image_part_face
            out['image_part_left_hand'] = image_part_left_hand
            out['image_part_right_hand'] = image_part_right_hand

        return out

    def get_part_for_metrics(self, image, cam, canonical_mapping_kwargs):
        # Find head center and project to image.
        device = image.device
        batch_size = image.shape[0]
        c = cam.clone() # NOTE: avoid inplace operator
        
        if c.shape[-1] > self.cam_dim:
            c = c[:, :self.cam_dim]

        # For dry run compatibility
        if torch.all(c==0).item():
            c = camera2world_dry_run.repeat(batch_size, 1).to(device)[:, :self.cam_dim]
            
        # face size ratio set as 20/256
        box_size_face = (32/512) * self.rendering_kwargs['image_resolution'][0]
        box_size_hand = (24/512) * self.rendering_kwargs['image_resolution'][0]
        target_size_face = int(2 * box_size_face)
        target_size_hand = int(2 * box_size_hand)
        padding=False

        with torch.no_grad():
            
            K = c[:, 16:25].reshape(-1, 3, 3)
            K[:, :2] *= self.rendering_kwargs['image_resolution'][0]
            extrinsics = torch.linalg.inv(c[:, :16].reshape(-1, 4, 4))
            R = extrinsics[:, :3, :3]
            T = extrinsics[:, :3, 3].unsqueeze(1)
            
            joints = canonical_mapping_kwargs['joints']
            joints_2d = project(joints, K, R, T)
            head_center_2D = joints_2d[:, 68] # nose
            left_hand_center_2D = joints_2d[:, LEFT_HAND_INDEX].mean(1) # left hand
            right_hand_center_2D = joints_2d[:, RIGHT_HAND_INDEX].mean(1) # right hand
            
            # Face
            if torch.min(head_center_2D) < box_size_face:
                # pad image upper part with white
                padding = True
                head_center_2D[:, 1] += box_size_face
            bboxs_face = torch.cat([head_center_2D - box_size_face, head_center_2D + box_size_face], dim=-1)
            
            _image = F.pad(image, (0, 0, int(box_size_face),0), "constant", 1.0) if padding else image

            bboxs_face = torch.cat([torch.arange(batch_size, dtype=torch.float32, device=device).reshape(batch_size, 1), bboxs_face.to(torch.float32) - 0.5], 1)
            image_part_face = roi_align(input=_image,
                                boxes=bboxs_face,
                                output_size=(target_size_face, target_size_face))
            
            # Left hand
            if torch.min(left_hand_center_2D[:, 0]) < box_size_hand:
                padding = True
                left_hand_center_2D[:, 0] += box_size_hand
            bboxs_left_hand = torch.cat([left_hand_center_2D - box_size_hand, left_hand_center_2D + box_size_hand], dim=-1)
            _image = F.pad(image, (int(box_size_hand), 0, 0, 0), "constant", 1.0) if padding else image

            bboxs_left_hand = torch.cat([torch.arange(batch_size, dtype=torch.float32, device=device).reshape(batch_size, 1), bboxs_left_hand.to(torch.float32) - 0.5], 1)
            image_part_left_hand = roi_align(input=_image,
                                boxes=bboxs_left_hand,
                                output_size=(target_size_hand, target_size_hand))
            
            # Right hand
            if torch.max(right_hand_center_2D[:, 0]) > (self.rendering_kwargs['image_resolution'][1] - box_size_hand):
                padding = True
                # right_hand_center_2D[:, 0] += box_size_hand
            bboxs_right_hand = torch.cat([right_hand_center_2D - box_size_hand, right_hand_center_2D + box_size_hand], dim=-1)
            _image = F.pad(image, (0, int(box_size_hand), 0, 0), "constant", 1.0) if padding else image

            bboxs_right_hand = torch.cat([torch.arange(batch_size, dtype=torch.float32, device=device).reshape(batch_size, 1), bboxs_right_hand.to(torch.float32) - 0.5], 1)
            image_part_right_hand = roi_align(input=_image,
                                boxes=bboxs_right_hand,
                                output_size=(target_size_hand, target_size_hand))

        return image_part_face, image_part_left_hand, image_part_right_hand

    def get_part(self, image, cam, canonical_mapping_kwargs):
        # Find head center and project to image.
        device = image.device
        batch_size = image.shape[0]
        c = cam.clone() # NOTE: avoid inplace operator
        
        if c.shape[-1] > self.cam_dim:
            c = c[:, :self.cam_dim]

        # For dry run compatibility
        if torch.all(c==0).item():
            c = camera2world_dry_run.repeat(batch_size, 1).to(device)[:, :self.cam_dim]
            
        # face size ratio set as 20/256
        box_size_face = (1.0*self.neural_rendering_resolution_face/512) * self.rendering_kwargs['image_resolution'][0]
        box_size_hand = (1.0*self.neural_rendering_resolution_hand/512) * self.rendering_kwargs['image_resolution'][0]
        target_size_face = int(2 * box_size_face)
        target_size_hand = int(2 * box_size_hand)
        padding=False

        with torch.no_grad():
            
            K = c[:, 16:25].reshape(-1, 3, 3)
            K[:2, :2] *= self.rendering_kwargs['image_resolution'][0]
            extrinsics = torch.linalg.inv(c[:, :16].reshape(-1, 4, 4))
            R = extrinsics[:, :3, :3]
            T = extrinsics[:, :3, 3].unsqueeze(1)
            
            joints = canonical_mapping_kwargs['joints']
            joints_2d = project(joints, K, R, T)
            head_center_2D = joints_2d[:, 68] # nose
            left_hand_center_2D = joints_2d[:, LEFT_HAND_INDEX].mean(1) # left hand
            right_hand_center_2D = joints_2d[:, RIGHT_HAND_INDEX].mean(1) # right hand
            
            # Face
            if torch.min(head_center_2D) < box_size_face:
                # pad image upper part with white
                padding = True
                head_center_2D[:, 1] += box_size_face
            bboxs_face = torch.cat([head_center_2D - box_size_face, head_center_2D + box_size_face], dim=-1)
            
            _image = F.pad(image, (0, 0, int(box_size_face),0), "constant", 1.0) if padding else image

            bboxs_face = torch.cat([torch.arange(batch_size, dtype=torch.float32, device=device).reshape(batch_size, 1), bboxs_face.to(torch.float32) - 0.5], 1)
            image_part_face = roi_align(input=_image,
                                boxes=bboxs_face,
                                output_size=(target_size_face, target_size_face))
            
            # Left hand
            if torch.min(left_hand_center_2D[:, 0]) < box_size_hand:
                padding = True
                left_hand_center_2D[:, 0] += box_size_hand
            bboxs_left_hand = torch.cat([left_hand_center_2D - box_size_hand, left_hand_center_2D + box_size_hand], dim=-1)
            _image = F.pad(image, (int(box_size_hand), 0, 0, 0), "constant", 1.0) if padding else image

            bboxs_left_hand = torch.cat([torch.arange(batch_size, dtype=torch.float32, device=device).reshape(batch_size, 1), bboxs_left_hand.to(torch.float32) - 0.5], 1)
            image_part_left_hand = roi_align(input=_image,
                                boxes=bboxs_left_hand,
                                output_size=(target_size_hand, target_size_hand))
            
            # Right hand
            if torch.max(right_hand_center_2D[:, 0]) > (self.rendering_kwargs['image_resolution'][1] - box_size_hand):
                padding = True
                # right_hand_center_2D[:, 0] += box_size_hand
            bboxs_right_hand = torch.cat([right_hand_center_2D - box_size_hand, right_hand_center_2D + box_size_hand], dim=-1)
            _image = F.pad(image, (0, int(box_size_hand), 0, 0), "constant", 1.0) if padding else image

            bboxs_right_hand = torch.cat([torch.arange(batch_size, dtype=torch.float32, device=device).reshape(batch_size, 1), bboxs_right_hand.to(torch.float32) - 0.5], 1)
            image_part_right_hand = roi_align(input=_image,
                                boxes=bboxs_right_hand,
                                output_size=(target_size_hand, target_size_hand))

        return image_part_face, image_part_left_hand, image_part_right_hand

    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        canonical_mapping_kwargs = self.get_canonical_mapping_info(c)
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs, return_sdf=True, return_eikonal=False, **canonical_mapping_kwargs)

    def sample_mixed(self, coordinates, directions, ws_face, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws_face' instead of Gaussian noise 'z'
        planes = self.backbone_face.synthesis(ws_face, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        canonical_mapping_kwargs = self.get_canonical_mapping_info(c, mode='face')
        return self.renderer_face.run_model(planes, self.decoder_face, coordinates, directions, self.rendering_kwargs, return_sdf=True, return_eikonal=False, **canonical_mapping_kwargs)

    def get_parameters(self):
        '''
        Freeze selected submodules
        '''
        # TODO: set different learning rates for each submodule
        frozen_modules = []
        # frozen_modules = ["backbone_face", "decoder_face", "backbone_hand", "decoder_hand"]
        for n, m in self.named_parameters():
            if n.split(".")[0] in frozen_modules:
                continue
            # if not n.startswith("backbone.mapping"): # or n.startswith("backbone.mapping"):
            #     continue
            yield m

    def forward(self, z, c, c_face, c_hand, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        # ws_face = self.mapping_face(z, c_face[:, :self.cam_dim], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        # ws_hand = self.mapping_hand(z, c_hand[:, :self.cam_dim], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        ws = self.mapping(z, c[:, :self.cam_dim], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        ws_geo = self.mapping_geo(z, c[:, self.cam_dim:], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        # ws_face = ws[:, :1].repeat(1, self.backbone_face.num_ws, 1)
        # ws_hand = ws[:, :1].repeat(1, self.backbone_hand.num_ws, 1)
        return self.synthesis(ws, ws_geo, c, c_face, c_hand, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options, sigmoid_beta=3.0):
        super().__init__()
        self.hidden_dim = 64

        # TODO: adapt our networks here
        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
        )
        self.net_rgb = FullyConnectedLayer(self.hidden_dim, options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        # self.net_sdf = FullyConnectedLayer(self.hidden_dim, 1, lr_multiplier=options['decoder_lr_mul'])
        self.net_sdf = FullyConnectedLayer(self.hidden_dim+self.hidden_dim//2, 1, lr_multiplier=options['decoder_lr_mul'])
        self.sdf_mapper = FullyConnectedLayer(1, self.hidden_dim//2, lr_multiplier=options['decoder_lr_mul'], activation='lrelu')
        
        torch.nn.init.xavier_uniform_(self.net_sdf.weight)
        self.sigmoid_beta = nn.Parameter(sigmoid_beta * torch.ones(1)) # 0.0041 face

    ### Eikonal
    def get_beta(self):
        return self.sigmoid_beta.abs()+0.0001

    def sdf2sigma(self, x):
        # stylesdf activation
        sigma=(1/self.get_beta()) * torch.sigmoid(x / self.get_beta())

        # objectsdf & anisdf activation
        # sigma = (1/self.get_beta())  * (0.5 - 0.5 * x.sign() * torch.expm1(-x.abs() / self.get_beta()))
        sigma[x<=-20]=0
        return sigma

    def forward(self, sampled_features, ray_directions, base_sdf=None):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)
        base_sdf = base_sdf.view(N*M, 1)
        
        x = self.net(x)
        rgb = self.net_rgb(x)
        rgb = rgb.view(N, M, -1)
        rgb = torch.sigmoid(rgb)*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF

        sdf_code = self.sdf_mapper(base_sdf)
        sdf = self.net_sdf(torch.cat((x, sdf_code), dim=-1)) # TODO: check if need another activation
        # sdf = self.net_sdf(x)
        sdf = sdf + base_sdf
        sdf = sdf.view(N, M, -1)
        base_sdf = base_sdf.view(N, M, -1)
        
        # clamp sdf to avoid collapse
        sdf = torch.clamp(sdf, torch.min(base_sdf)-0.5, torch.max(base_sdf)+0.5)

        return {'rgb': rgb, 'sdf': sdf}


class AvatarGenOSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options, sigmoid_beta=0.1):
        super().__init__()
        self.hidden_dim = 64

        # TODO: adapt our networks here
        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
        )
        self.net_rgb = FullyConnectedLayer(self.hidden_dim, options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        self.net_sdf = FullyConnectedLayer(self.hidden_dim, 1, lr_multiplier=options['decoder_lr_mul'])
        # self.sdf_mapper = FullyConnectedLayer(1, self.hidden_dim//2, lr_multiplier=options['decoder_lr_mul'], activation='lrelu')
        
        torch.nn.init.xavier_uniform_(self.net_sdf.weight)
        self.sigmoid_beta = torch.nn.Parameter(sigmoid_beta * torch.ones(1))

    ### Eikonal
    def get_beta(self):
        return self.sigmoid_beta.abs()+0.0001

    def sdf2sigma(self, x):
        # stylesdf activation
        sigma=(1/self.get_beta()) * torch.sigmoid(x / self.get_beta())

        # objectsdf & anisdf activation
        # sigma = (1/self.get_beta())  * (0.5 - 0.5 * x.sign() * torch.expm1(-x.abs() / self.get_beta()))
        sigma[x<=-20]=0
        return sigma

    def forward(self, sampled_features, ray_directions, face_sdf=None):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)
        # face_sdf = face_sdf.view(N*M, 1)

        x = self.net(x)
        rgb = self.net_rgb(x)
        rgb = rgb.view(N, M, -1)
        rgb = torch.sigmoid(rgb)*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF

        # sdf_code = self.sdf_mapper(face_sdf)
        # sdf = self.net_sdf(torch.cat((x, sdf_code), dim=-1)) # TODO: check if need another activation
        sdf = self.net_sdf(x) # TODO: check if need another activation
        # sdf = sdf + face_sdf
        sdf = sdf.view(N, M, -1)
        
        # clamp sdf to avoid collapse
        # sdf = torch.clamp(sdf, torch.min(face_sdf)-0.5, torch.max(face_sdf)+0.5)

        return {'rgb': rgb, 'sdf': sdf}


class OffsetNetwork(torch.nn.Module):
    def __init__(self,
                 depth=6, 
                 width=128,
                 freqs_xyz=10,
                 out_channels=3,
                 use_pose_cond=False,
                 freqs_pose=2,
                 scale=0.1,
                 scale_type="clamp",
                 skips=[4],
                 activation="lrelu"):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels: number of input channels for xyz (3+3*10*2=63 by default)
        skips: add skip connection in the Dth layer
        """
        super().__init__()  
        self.depth = depth
        self.width = width
        self.freqs_xyz = freqs_xyz
        self.skips = skips
        self.scale = scale
        self.scale_type = scale_type
        self.use_pose_cond = use_pose_cond

        self.in_channels = 6 + 6*freqs_xyz*2
        code_dim=512*2


        self.out_channels = out_channels
        self.encoding_xyz = Embedder(6, freqs_xyz)


        self.pts_linears = []
        for i in range(depth):
            if i == 0:
                layer = FullyConnectedLayer(self.in_channels+code_dim, width, activation=activation)
            elif i in skips:
                layer = FullyConnectedLayer(width + self.in_channels, width, activation=activation)
            else:
                layer = FullyConnectedLayer(width, width)
            self.pts_linears.append(layer)
        self.pts_linears = nn.ModuleList(self.pts_linears)        

        self.output_linear = FullyConnectedLayer(width, out_channels, activation=activation)
        
    def forward(self, xyz, ws=None, ws_geo=None, pose=None):
        w = ws[:,:1]
        w_geo = ws_geo[:,:1]
        xyz_encoded = self.encoding_xyz(xyz)
        h = torch.cat([xyz_encoded, w.repeat(1, xyz_encoded.size(1), 1), w_geo.repeat(1, xyz_encoded.size(1), 1)], -1)
        B, N, C = h.shape
        h = h.reshape(B*N, C)
        xyz_encoded = xyz_encoded.reshape(B*N,-1)
        for i, l in enumerate(self.pts_linears):
            if i in self.skips:
                h = torch.cat([xyz_encoded, h], -1)
            h = self.pts_linears[i](h)

        outputs = self.output_linear(h)*self.scale
        return outputs.reshape(B,N, -1)


class Embedder(torch.nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)
