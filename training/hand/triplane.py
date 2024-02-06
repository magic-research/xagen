# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import joblib
import dnnlib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import persistence, misc
from training.hand.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.hand.volumetric_rendering.renderer import ImportanceRenderer
from training.hand.volumetric_rendering.ray_sampler import RaySampler, AvatarGenRaySampler
from training.hand.volumetric_rendering.utils import get_canonical_pose, world2cam_transform
from smplx.SMPLX import SMPLX


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
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = AvatarGenRaySampler()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        # self.to_rgb = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)   
        # )
        self.decoder = AvatarGenOSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32}, sigmoid_beta=rendering_kwargs['sigmoid_beta'])
        self.neural_rendering_resolution = 32
        self.cam_dim = 25
        self.part_disc = False
        self.rendering_kwargs = rendering_kwargs
        self.use_deformation = rendering_kwargs['use_deformation']
        
        self.body_model = SMPLX()
        self.FLAME_INDEX = np.load('smplx/assets/SMPL-X__FLAME_vertex_ids.npy').tolist()
        self.RIGHT_HAND_INDEX = joblib.load('smplx/assets/MANO_SMPLX_vertex_ids.pkl')['right_hand']

        if self.use_deformation:
            self.offset_network = OffsetNetwork(depth=6, width=128, freqs_xyz=10, scale=0.1, scale_type='clamp', use_pose_cond=rendering_kwargs['use_pose_cond'])
        else:
            self.offset_network = None

        # Define Da-Pose
        pose_canonical = self.body_model.get_canonical_pose(use_da_pose=True)
        lbs_weights = self.body_model.lbs_weights
        faces = self.body_model.faces_tensor
        # body:  [[-1.2061, -1.5118, -0.2120], [ 1.2058,  0.7406,  0.2011]]
        # canonical_bbox = torch.tensor([[-1.1040029525756836, -1.5, -0.1670],[1.0613611936569214, 1.0272386074066162, 0.2306]]).reshape(1,2,3).float()
        # face: [[-0.1389,  0.0320, -0.1719], [ 0.1393,  0.4833,  0.1628]]
        # canonical_bbox = torch.tensor([[-0.1389, 0.032, -0.1719], [0.1393, 0.4833, 0.1628]]).reshape(1,2,3).float()
        canonical_bbox = torch.tensor([[-0.9554, -0.0285, -0.2266], [-0.5797,  0.0921,  0.1172]]).reshape(1, 2, 3).float()

        self.register_buffer('canonical_bbox', canonical_bbox)
        self.register_buffer('pose_canonical', pose_canonical)
        self.register_buffer('lbs_weights', lbs_weights)
        self.register_buffer('faces', faces)

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

        self._last_planes = None

    def calc_ober2cano_transform(
        self, verts_transform, verts_transform_canonical, 
        shape_offsets, shape_offsets_canonical,
        pose_offsets, pose_offsets_canonical):

        ober2cano_transform = torch.inverse(verts_transform).clone()
        ober2cano_transform[..., :3, 3] = ober2cano_transform[..., :3, 3] + (shape_offsets_canonical - shape_offsets)
        ober2cano_transform[..., :3, 3] = ober2cano_transform[..., :3, 3] + (pose_offsets_canonical - pose_offsets)
        ober2cano_transform = torch.matmul(verts_transform_canonical, ober2cano_transform)
        return ober2cano_transform

    def get_canonical_mapping_info(self, c):
        c_cam = c[:, :self.cam_dim]
        c_smplx = c[:, self.cam_dim:]

        c_betas = c_smplx[:, :200]
        c_expressions = c_smplx[:, 200:250]
        c_poses = c_smplx[:, 250:]
        device = c.device
        batch_size = c.shape[0]

        with torch.no_grad():

            # Observation Space
            # NOTE: use zero body pose for hand generation
            body_model_params = {'body_pose':torch.zeros_like(c_poses[:, 3:66]), 'jaw_pose':c_poses[:, 66:69], 'eye_pose':torch.zeros_like(c_poses[:, 0:6]),
                                 'left_hand_pose':c_poses[:, 69:114], 'right_hand_pose': c_poses[:, 114:], 'shape_params': c_betas, 'expression_params':c_expressions}
            verts, _, joints, verts_transform, shape_offsets, pose_offsets = self.body_model(**body_model_params, pose2rot=True, return_transform=True, return_offsets=True)

            triangles = face_vertices(verts, self.faces, device)

            # Compute transformation from observation to canonical space
            ober2cano_transform = self.calc_ober2cano_transform(
                verts_transform, self.verts_transform_canonical.repeat(c_poses.shape[0], 1, 1, 1),
                shape_offsets, self.shape_offsets_canonical.repeat(c_poses.shape[0], 1, 1),
                pose_offsets, self.pose_offsets_canonical.repeat(c_poses.shape[0], 1, 1)
            )

            # Compute body bbox
            try:
                extrinsics = torch.linalg.inv(c_cam[:, :16].reshape(-1, 4, 4))
            except:
                # For Dry-run compatibility
                extrinsics = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            R = extrinsics[:, :3, :3]
            t = extrinsics[:, :3, 3].unsqueeze(1)
            verts_cam = world2cam_transform(verts, R, t)
            # import cv2
            # intrinsics = (c_cam[:, 16:25]).reshape(1, 3, 3)
            # intrinsics[:, 0, :] *= 512
            # intrinsics[:, 1, :] *= 512
            # xy = verts_cam / verts_cam[:, :, 2:]
            # xy = torch.einsum('bij,bkj->bki', intrinsics//4, xy)
            # _img = np.ones((512,512,3), dtype=np.uint8)*255
            # for i, point in enumerate(xy[0, self.RIGHT_HAND_INDEX]):
            #     _img = cv2.circle(_img, (int(point[0])+256, int(point[1])+256), 1, (255, 0, 0), 1)
            #     # img = cv2.putText(img, text=str(i), org=(int(point[0]), int(point[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 0, 0), thickness=1)
            # cv2.imwrite("kpts.jpg", _img[:,:,::-1])

            face_min = torch.amin(verts_cam[:, self.RIGHT_HAND_INDEX], dim=1)
            face_max = torch.amax(verts_cam[:, self.RIGHT_HAND_INDEX], dim=1)
            # extend box by a pre-defined ratio
            ratio = 0.2  # TODO: set as an arugment
            face_bbox = torch.stack([(1+ratio)*face_min-ratio*face_max, (1+ratio)*face_max-ratio*face_min], dim=1)  # Bx2x3

            # canonical_mapping_kwargs = {
            #     'body_bbox': body_bbox,
            #     'ober2cano_transform': ober2cano_transform[:, self.RIGHT_HAND_INDEX],
            #     'verts': verts[:, self.RIGHT_HAND_INDEX],
            #     'verts_canonical': verts_canonical[:, self.RIGHT_HAND_INDEX],
            #     'faces': self.faces,
            #     'triangles': triangles,
            #     'lbs_weights': self.lbs_weights[self.RIGHT_HAND_INDEX],
            #     'canonical_bbox': self.canonical_bbox,
            #     'smplx_params': torch.cat((c_betas, c_expressions, c_poses), dim=-1)
            # }

            canonical_mapping_kwargs = {
                'face_bbox': face_bbox,
                'ober2cano_transform': ober2cano_transform,
                'verts': verts,
                'verts_canonical': self.verts_canonical.repeat(c_poses.shape[0], 1, 1),
                'faces': self.faces,
                'triangles': triangles,
                'lbs_weights': self.lbs_weights,
                'canonical_bbox': self.canonical_bbox,
                'smplx_params': torch.cat((c_betas, c_expressions, c_poses), dim=-1)
            }

            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(verts_cam[0, self.RIGHT_HAND_INDEX, 0].detach().cpu(), verts_cam[0, self.RIGHT_HAND_INDEX, 1].detach().cpu(), verts_cam[0, self.RIGHT_HAND_INDEX, 2].detach().cpu(), alpha=0.05)
            # ax.scatter(verts_cam[0, :, 0].detach().cpu(), verts_cam[0, :, 1].detach().cpu(), verts_cam[0, :, 2].detach().cpu(), alpha=0.05)
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            # ax.view_init(0, 90)
            # plt.savefig("debug/tmp_0.jpg")
            # ax.view_init(90, 90)
            # plt.savefig("debug/tmp_90.jpg")

            return canonical_mapping_kwargs
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c[:, :self.cam_dim] * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def mapping_geo(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping_geo(z, c[:, self.cam_dim:] * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)    

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, return_eikonal=False, return_sdf=True, smplx_reg=False, smplx_reg_full_space=False, **synthesis_kwargs):

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution
        
        canonical_mapping_kwargs = self.get_canonical_mapping_info(c)

        # Create a batch of rays for volume rendering
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples, eikonal, smplx_sdf, sdf_reg_weights, coarse_sdf, deformation_reg = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs, canonical_mapping_kwargs, 
                                                                                                                        return_eikonal=return_eikonal, return_sdf=return_sdf, smplx_reg=smplx_reg, offset_network=self.offset_network, 
                                                                                                                        density_reg=None, smplx_reg_full_space=None) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        # sr_image = self.to_rgb(feature_image) #
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        out = {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'eikonal': eikonal, 'smplx_sdf': smplx_sdf, 'coarse_sdf': coarse_sdf, 'sdf_reg_weights': sdf_reg_weights, 'deformation_reg': deformation_reg}
        
        return out
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c[:, :self.cam_dim], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        canonical_mapping_kwargs = self.get_canonical_mapping_info(c)
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs, **canonical_mapping_kwargs)

    def sample_mixed(self, coordinates, directions, ws, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        canonical_mapping_kwargs = self.get_canonical_mapping_info(c)
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs, return_sdf=True, return_eikonal=False, density_reg=True, **canonical_mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, smplx_reg_full_space=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c[:, :self.cam_dim], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        ws_geo = None #self.mapping_geo(torch.rand_like(z), c[:, self.cam_dim:], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, smplx_reg_full_space=smplx_reg_full_space, **synthesis_kwargs)


from training.hand.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}

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
                 skips=[4]):
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

        self.in_channels = 3 + 3*freqs_xyz*2
        self.out_channels = out_channels
        self.encoding_xyz = Embedder(3, freqs_xyz)

        if use_pose_cond:
            self.encoding_pose = Embedder(409, freqs_pose)
            self.pose_linear = nn.Linear(self.encoding_pose.out_channels, width//2)
            self.combine_linear = nn.Linear(width+width//2, width)
            self.combine_linear.bias.data[:] = 0.0

        self.pts_linears = []
        for i in range(depth):
            if i == 0:
                layer = nn.Linear(self.in_channels, width)
            elif i in skips:
                layer = nn.Linear(width + self.in_channels, width)
            else:
                layer = nn.Linear(width, width)
            layer.bias.data[:] = 0.0
            self.pts_linears.append(layer)
        self.pts_linears = nn.ModuleList(self.pts_linears)        

        self.output_linear = nn.Linear(width, out_channels)
        self.output_linear.bias.data[:] = 0.0
        
    def forward(self, xyz, pose=None, bbox_norm=None):
        xyz_encoded = self.encoding_xyz(xyz)
        h = xyz_encoded
        for i, l in enumerate(self.pts_linears):
            if i in self.skips:
                h = torch.cat([xyz_encoded, h], -1)
            h = self.pts_linears[i](h)
            h = F.relu(h)

        if self.use_pose_cond:
            assert pose is not None
            pose_encoded = self.pose_linear(self.encoding_pose(pose)).unsqueeze(1).repeat(1, h.size(1), 1)
            h = self.combine_linear(torch.cat((h, pose_encoded), dim=-1))
            h = F.relu(h)
        
        outputs = self.output_linear(h)

        if self.scale_type == 'no':
            return outputs
        elif self.scale_type == 'linear':
            return outputs * self.scale
        elif self.scale_type == 'tanh':
            return torch.tanh(outputs) * self.scale
        elif self.scale_type == 'clamp':
            assert bbox_norm is not None
            return torch.clamp(outputs, -self.scale*bbox_norm[:, None, None], self.scale*bbox_norm[:, None, None])


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
