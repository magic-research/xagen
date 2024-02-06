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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import persistence, misc
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.volumetric_rendering.hyper_renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler, AvatarGenRaySampler
from training.volumetric_rendering.utils import get_canonical_pose
import dnnlib

# from deprec_smplx import create


@misc.profiled_function
def face_vertices(vertices, faces, device):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """

    bs, nv = vertices.shape[:2]
    _, nf = faces.shape[:2]
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs*nv, vertices.shape[-1]))
    return vertices[faces.long()]


@persistence.persistent_class
class HyperPlaneGenerator(torch.nn.Module):
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
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*6, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32}, sigmoid_beta=rendering_kwargs['sigmoid_beta'])
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
        self.cam_dim=20
        self._last_planes = None
        self.use_deformation = rendering_kwargs['use_deformation']
        self.canonical_reg = rendering_kwargs['canonical_reg']
        
        if self.use_deformation:
            self.offset_network = OffsetNetwork(depth=6, width=128, freqs_xyz=10, scale=1.0, scale_type='linear')
        else:
            self.offset_network = None

        # Define SMPL BODY MODEL
        self.body_model = create(model_path='./smplx/models', model_type='smpl', gender='neutral')
        # Define X-Pose
        pose_canonical = get_canonical_pose()
        lbs_weights = self.body_model.lbs_weights
        faces = torch.from_numpy(self.body_model.faces.astype(np.int64))
        # TODO: how to determine canonical bbox for different datasets
        # canonical_bbox = torch.tensor([[-1.2764, -1.4695, -0.2149], [1.2788, 0.9363, 0.2949]]).reshape(1,2,3).float()
        canonical_bbox = torch.tensor([[-1.0878, -1.3004, -0.1670, -0.7],[1.0900, 0.7663, 0.2306, 1.0]]).reshape(1,2,4).float()
        self.register_buffer('canonical_bbox', canonical_bbox)
        self.register_buffer('pose_canonical', pose_canonical)
        self.register_buffer('lbs_weights', lbs_weights)
        self.register_buffer('faces', faces)
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def mapping_geo(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping_geo(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def calc_ober2cano_transform(
        self, verts_transform, verts_transform_canonical, 
        shape_offsets, shape_offsets_canonical,
        pose_offsets, pose_offsets_canonical):

        ober2cano_transform = torch.inverse(verts_transform).clone()
        ober2cano_transform[..., :3, 3] = ober2cano_transform[..., :3, 3] + (shape_offsets_canonical - shape_offsets)
        ober2cano_transform[..., :3, 3] = ober2cano_transform[..., :3, 3] + (pose_offsets_canonical - pose_offsets)
        ober2cano_transform = torch.matmul(verts_transform_canonical, ober2cano_transform)
        return ober2cano_transform

    def get_camera_params(self, c, fixed_focal=5000.0):

        device = c.device
        focal = c[:, :2].reshape(-1, 1, 2).detach() * fixed_focal         # un-normalize focal
        focal = focal * (self.neural_rendering_resolution / self.rendering_kwargs['image_resolution'])    # normalize focal based on current size
        center = c[:, 2:4].reshape(-1, 1, 2).detach() * fixed_focal       # un-normalize center
        center = center * (self.neural_rendering_resolution / self.rendering_kwargs['image_resolution'])  # normalize center based on current size
        P = c[:, 4:20].reshape(-1, 4, 4).detach()
        M = torch.eye(4).to(device)
        M[1, 1] *= -1
        M[2, 2] *= -1
        if torch.all(c==0).item():
            P = M.unsqueeze(0).repeat(P.size(0), 1, 1)
            focal = focal + fixed_focal
            center = center + self.neural_rendering_resolution / 2
        else:
            P = torch.linalg.inv(P)
            P = torch.bmm(P, M.unsqueeze(0).repeat(P.size(0), 1, 1))
        return focal, center, P

    def get_canonical_mapping_info(self, c):
        c_cam = c[:, :self.cam_dim]
        c_smpl = c[:, self.cam_dim:]

        c_betas = c_smpl[:, :10]
        c_poses = c_smpl[:, 10:]
        device = c_cam.device

        with torch.no_grad():
            # Canonical Space
            # TODO: check if we need to use a fixed shape for canonical space
            body_model_params_canonical = {'body_pose': self.pose_canonical.unsqueeze(0).repeat(c_poses.size(0), 1), 'betas': c_betas}
            body_model_out_canonical = self.body_model(**body_model_params_canonical, return_verts=True)
            verts_canonical = body_model_out_canonical['vertices']
            triangles = face_vertices(verts_canonical, self.faces, device)
            verts_transform_canonical = body_model_out_canonical['vertices_transform']
            shape_offsets_canonical = body_model_out_canonical['shape_offsets']
            pose_offsets_canonical = body_model_out_canonical['pose_offsets']
            del body_model_out_canonical

            # Observation Space
            body_model_params = {'body_pose': c_poses, 'betas': c_betas}
            body_model_out = self.body_model(**body_model_params, return_verts=True)
            verts = body_model_out['vertices']
            verts_transform = body_model_out['vertices_transform']
            shape_offsets = body_model_out['shape_offsets']
            pose_offsets = body_model_out['pose_offsets']
            del body_model_out

            # Compute transformation from observation to canonical space
            ober2cano_transform = self.calc_ober2cano_transform(
                verts_transform, verts_transform_canonical,
                shape_offsets, shape_offsets_canonical,
                pose_offsets, pose_offsets_canonical
            )

            # Compute body bbox
            body_min = torch.amin(verts, dim=1)
            body_max = torch.amax(verts, dim=1)
            # extend box by a pre-defined ratio
            ratio = 0.2  # TODO: set as an arugment
            body_bbox = torch.stack([(1+ratio)*body_min-ratio*body_max, (1+ratio)*body_max-ratio*body_min], dim=1)  # Bx2x3

            canonical_mapping_kwargs = {
                'body_bbox': body_bbox,
                'ober2cano_transform': ober2cano_transform,
                'verts': verts,
                'verts_canonical': verts_canonical,
                'faces': self.faces,
                'triangles': triangles,
                'lbs_weights': self.lbs_weights,
                'canonical_bbox': self.canonical_bbox,
            }
            return canonical_mapping_kwargs

    def synthesis(self, ws, ws_geo, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, return_eikonal=False, smpl_reg=False, **synthesis_kwargs):

        c_cam = c[:, :self.cam_dim]
        c_smpl = c[:, self.cam_dim:]
        focal, center, cam2world_matrix = self.get_camera_params(c_cam)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution
        
        canonical_mapping_kwargs = self.get_canonical_mapping_info(c)

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, focal, center, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into six 32-channel planes
        planes = planes.view(len(planes), 6, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples, eikonal, smpl_sdf, coarse_sdf = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs, canonical_mapping_kwargs, return_eikonal=return_eikonal, return_sdf=True, smpl_reg=smpl_reg, offset_network=self.offset_network, canonical_reg=self.canonical_reg) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'eikonal': eikonal, 'smpl_sdf': smpl_sdf, 'coarse_sdf':coarse_sdf}

    def get_geometry(self, ws, ws_geo, c, **synthesis_kwargs):
        
        c_cam = c[:, :self.cam_dim]
        focal, center, cam2world_matrix = self.get_camera_params(c_cam)
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, focal, center, self.neural_rendering_resolution)
        coordinates, directions = self.renderer.sample_frostum_grid(ray_origins, ray_directions, self.rendering_kwargs)
        return self.sample_mixed(coordinates, directions, ws, ws_geo, c, **synthesis_kwargs)

    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 6, 32, planes.shape[-2], planes.shape[-1])
        canonical_mapping_kwargs = self.get_canonical_mapping_info(c)
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs, return_sdf=True, return_eikonal=False, **canonical_mapping_kwargs)

    def sample_mixed(self, coordinates, directions, ws, ws_geo, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 6, 32, planes.shape[-2], planes.shape[-1])
        canonical_mapping_kwargs = self.get_canonical_mapping_info(c)
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs, return_sdf=True, return_eikonal=False, **canonical_mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c[:, :self.cam_dim], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        ws_geo = self.mapping_geo(torch.rand_like(z), c[:, self.cam_dim:], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, ws_geo, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


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
        self.net_sdf = FullyConnectedLayer(self.hidden_dim+self.hidden_dim//2, 1, lr_multiplier=options['decoder_lr_mul'])
        self.sdf_mapper = FullyConnectedLayer(1, self.hidden_dim//2, lr_multiplier=options['decoder_lr_mul'], activation='lrelu')
        
        torch.nn.init.xavier_uniform_(self.net_sdf.weight)
        self.sigmoid_beta = nn.Parameter(sigmoid_beta * torch.ones(1))

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

    def forward(self, sampled_features, ray_directions, body_sdf):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        rgb = self.net_rgb(x)
        rgb = rgb.view(N, M, -1)
        rgb = torch.sigmoid(rgb)*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF

        sdf_code = self.sdf_mapper(body_sdf)
        sdf = self.net_sdf(torch.cat((x, sdf_code), dim=-1)) # TODO: check if need another activation
        sdf = sdf + body_sdf
        sdf = sdf.view(N, M, -1)
        
        # clamp sdf to avoid collapse
        sdf = torch.clamp(sdf, torch.min(body_sdf)-0.5, torch.max(body_sdf)+0.5)

        return {'rgb': rgb, 'sdf': sdf}


class OffsetNetwork(torch.nn.Module):
    def __init__(self,
                 depth=6, 
                 width=128,
                 freqs_xyz=10,
                 out_channels=3,
                 scale=1.0,
                 scale_type="no",
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

        self.in_channels = 3 + 3*freqs_xyz*2
        self.out_channels = out_channels
        self.encoding_xyz = Embedder(3, freqs_xyz)

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.in_channels, width)] + [nn.Linear(width, width) if i not in self.skips else nn.Linear(width + self.in_channels, width) for i in range(depth-1)])
        self.output_linear = nn.Linear(width, out_channels)
        
    def forward(self, xyz):
        xyz_encoded = self.encoding_xyz(xyz)
        h = xyz_encoded
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([xyz_encoded, h], -1)
        outputs = self.output_linear(h)

        if self.scale_type == 'no':
            return outputs
        elif self.scale_type == 'linear':
            return outputs * self.scale
        elif self.scale_type == 'tanh':
            return torch.tanh(outputs) * self.scale


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
