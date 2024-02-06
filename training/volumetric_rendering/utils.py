import torch
import random, os
import trimesh
import numpy as np
from math import atan2, asin, cos
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from scipy.spatial import Delaunay
from skimage.measure import marching_cubes
from torch_utils import persistence, misc
import pytorch3d.io
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    HardPhongShader
)
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.renderer.blending import (
    BlendParams,
    softmax_rgb_blend
)

try:
    import mcubes
except:
    os.system("pip3 install --upgrade PyMCubes")
    import mcubes

@misc.profiled_function
def world2cam_transform(xyz, R, T):
    # batch operator
    # camera coord
    xyz = torch.einsum('bij,bkj->bki', R, xyz)
    xyz = xyz + T
    return xyz

@misc.profiled_function
def project(xyz, K, R, T):
    # camera coord
    xyz = torch.einsum('bij,bkj->bki', R, xyz)
    xyz = xyz + T
    # camera coord -> image coord
    xyz = xyz / xyz[:, :, 2:]
    xy = torch.einsum('bij,bkj->bki', K, xyz)
    # xy[:, :, 1] = 2*112 - xy[:, :, 1]
    return xy[:, :, :2]

######################### Dataset util functions ###########################
# Get data sampler
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

# Get data minibatch
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

############################## Model weights util functions #################
# Turn model gradients on/off
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

# Exponential moving average for generator weights
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

################### Latent code (Z) sampling util functions ####################
# Sample Z space latent codes for the generator
def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)
    else:
        return [make_noise(batch, latent_dim, 1, device)]

################# Camera parameters sampling ####################
def generate_camera_params(resolution, device, batch=1, locations=None, sweep=False,
                           uniform=False, azim_range=0.3, elev_range=0.15,
                           fov=12, dist_radius=0.12):
    if locations != None:
        azim = locations[:,0].view(-1,1)
        elev = locations[:,1].view(-1,1)

        # generate intrinsic parameters
        # fix distance to 1
        dist = torch.ones(azim.shape[0], 1, device=device)
        near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
        fov_angle = fov / 2 * torch.ones(azim.shape[0], 1, device=device).view(-1,1) * np.pi / 180
        focal = 0.5 * resolution / torch.tan(fov_angle).unsqueeze(-1)
    elif sweep:
        # generate camera locations on the unit sphere
        azim = (-azim_range + (2 * azim_range / 7) * torch.arange(8, device=device)).view(-1,1).repeat(batch,1)
        elev = (-elev_range + 2 * elev_range * torch.rand(batch, 1, device=device).repeat(1,8).view(-1,1))

        # generate intrinsic parameters
        dist = (torch.ones(batch, 1, device=device)).repeat(1,8).view(-1,1)
        near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
        fov_angle = fov / 2 * torch.ones(batch, 1, device=device).repeat(1,8).view(-1,1) * np.pi / 180
        focal = 0.5 * resolution / torch.tan(fov_angle).unsqueeze(-1)
    else:
        # sample camera locations on the unit sphere
        if uniform:
            azim = (-azim_range + 2 * azim_range * torch.rand(batch, 1, device=device))
            elev = (-elev_range + 2 * elev_range * torch.rand(batch, 1, device=device))
        else:
            azim = (azim_range * torch.randn(batch, 1, device=device))
            elev = (elev_range * torch.randn(batch, 1, device=device))

        # generate intrinsic parameters
        dist = torch.ones(batch, 1, device=device) # restrict camera position to be on the unit sphere
        near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
        fov_angle = fov / 2 * torch.ones(batch, 1, device=device) * np.pi / 180 # full fov is 12 degrees
        focal = 0.5 * resolution / torch.tan(fov_angle).unsqueeze(-1)

    viewpoint = torch.cat([azim, elev], 1)

    #### Generate camera extrinsic matrix ##########

    # convert angles to xyz coordinates
    x = torch.cos(elev) * torch.sin(azim)
    y = torch.sin(elev)
    z = torch.cos(elev) * torch.cos(azim)
    camera_dir = torch.stack([x, y, z], dim=1).view(-1,3)
    camera_loc = dist * camera_dir

    # get rotation matrices (assume object is at the world coordinates origin)
    up = torch.tensor([[0,1,0]]).float().to(device) * torch.ones_like(dist)
    z_axis = F.normalize(camera_dir, eps=1e-5) # the -z direction points into the screen
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    T = camera_loc[:, :, None]
    extrinsics = torch.cat((R.transpose(1, 2), T), -1)
    homo_coord = torch.Tensor([0., 0., 0., 1.]).to(device).unsqueeze(0).unsqueeze(0).repeat(batch, 1, 1)
    extrinsics = torch.cat((extrinsics, homo_coord), dim=1)
    focal = focal.repeat(1, 1, 2)
    return extrinsics, focal, near, far, viewpoint

#################### Mesh generation util functions ########################
# Reshape sampling volume to camera frostum
def align_volume(volume, near=0.88, far=1.12):
    b, h, w, d, c = volume.shape
    yy, xx, zz = torch.meshgrid(torch.linspace(-1, 1, h),
                                torch.linspace(-1, 1, w),
                                torch.linspace(-1, 1, d))

    grid = torch.stack([xx, yy, zz], -1).to(volume.device)

    frostum_adjustment_coeffs = torch.linspace(far / near, 1, d).view(1,1,1,-1,1).to(volume.device)
    frostum_grid = grid.unsqueeze(0)
    frostum_grid[...,:2] = frostum_grid[...,:2] * frostum_adjustment_coeffs
    out_of_boundary = torch.any((frostum_grid.lt(-1).logical_or(frostum_grid.gt(1))), -1, keepdim=True)
    frostum_grid = frostum_grid.permute(0,3,1,2,4).contiguous()
    permuted_volume = volume.permute(0,4,3,1,2).contiguous()
    final_volume = F.grid_sample(permuted_volume, frostum_grid, padding_mode="border", align_corners=True)
    final_volume = final_volume.permute(0,3,4,2,1).contiguous()
    # set a non-zero value to grid locations outside of the frostum to avoid marching cubes distortions.
    # It happens because pytorch grid_sample uses zeros padding.
    final_volume[out_of_boundary] = 1

    return final_volume


# Reshape sampling volume to camera frostum
def align_density_volume(volume, near=0.88, far=1.12):
    b, h, w, d, c = volume.shape
    yy, xx, zz = torch.meshgrid(torch.linspace(-1, 1, h),
                                torch.linspace(-1, 1, w),
                                torch.linspace(-1, 1, d))

    grid = torch.stack([xx, yy, zz], -1).to(volume.device)

    frostum_adjustment_coeffs = torch.linspace(far / near, 1, d).view(1,1,1,-1,1).to(volume.device)
    frostum_grid = grid.unsqueeze(0)
    frostum_grid[...,:2] = frostum_grid[...,:2] * frostum_adjustment_coeffs
    out_of_boundary = torch.any((frostum_grid.lt(-1).logical_or(frostum_grid.gt(1))), -1, keepdim=True)
    frostum_grid = frostum_grid.permute(0,3,1,2,4).contiguous()
    permuted_volume = volume.permute(0,4,3,1,2).contiguous()
    final_volume = F.grid_sample(permuted_volume, frostum_grid, padding_mode="border", align_corners=True)
    final_volume = final_volume.permute(0,3,4,2,1).contiguous()
    # set a non-zero value to grid locations outside of the frostum to avoid marching cubes distortions.
    # It happens because pytorch grid_sample uses zeros padding.
    final_volume[out_of_boundary] = 0.0

    return final_volume

# Extract mesh with marching cubes
def extract_mesh_with_marching_cubes(sdf, level=0):
    b, h, w, d, _ = sdf.shape

    # change coordinate order from (y,x,z) to (x,y,z)
    sdf_vol = sdf[0,...,0].permute(1,0,2).cpu().numpy()

    # scale vertices
    verts, faces, _, _ = marching_cubes(sdf_vol, level)
    verts[:,0] = (verts[:,0]/float(w)-0.5)*0.24
    verts[:,1] = (verts[:,1]/float(h)-0.5)*0.24
    verts[:,2] = (verts[:,2]/float(d)-0.5)*0.24

    # fix normal direction
    verts[:,2] *= -1; verts[:,1] *= -1
    mesh = trimesh.Trimesh(verts, faces)

    return mesh

# Generate mesh from xyz point cloud
def xyz2mesh(xyz):
    b, _, h, w = xyz.shape
    x, y = np.meshgrid(np.arange(h), np.arange(w))

    # Extract mesh faces from xyz maps
    tri = Delaunay(np.concatenate((x.reshape((h*w, 1)), y.reshape((h*w, 1))), 1))
    faces = tri.simplices

    # invert normals
    faces[:,[0, 1]] = faces[:,[1, 0]]

    # generate_meshes
    mesh = trimesh.Trimesh(xyz.squeeze(0).permute(1,2,0).view(h*w,3).cpu().numpy(), faces)

    return mesh


################# Mesh rendering util functions #############################
def add_textures(meshes:Meshes, vertex_colors=None) -> Meshes:
    verts = meshes.verts_padded()
    if vertex_colors is None:
        vertex_colors = torch.ones_like(verts) # (N, V, 3)
    textures = TexturesVertex(verts_features=vertex_colors)
    meshes_t = Meshes(
        verts=verts,
        faces=meshes.faces_padded(),
        textures=textures,
        verts_normals=meshes.verts_normals_padded(),
    )
    return meshes_t


def create_cameras(
    R=None, T=None,
    azim=0, elev=0., dist=1.,
    fov=12., znear=0.01,
    device="cuda") -> FoVPerspectiveCameras:
    """
    all the camera parameters can be a single number, a list, or a torch tensor.
    """
    if R is None or T is None:
        R, T = look_at_view_transform(dist=dist, azim=azim, elev=elev, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear, fov=fov)
    return cameras


def NormalCalcuate(meshes, fragments):
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals #  torch.ones_like()
    )
    return pixel_normals


class NormalShader(nn.Module):
    def __init__(self, device="cuda", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()
    def forward(self, fragments, meshes, **kwargs):
        blend_params = kwargs.get("blend_params", self.blend_params)
        normals = NormalCalcuate(meshes, fragments)
        images = softmax_rgb_blend(normals, fragments, blend_params)[:,:,:,:3]

        images = F.normalize(images, dim=3)
        return images


def create_mesh_renderer(
        cameras: FoVPerspectiveCameras,
        image_size: int = 256,
        blur_radius: float = 1e-6,
        light_location=((-0.5, 1., 5.0),),
        device="cuda",
        **light_kwargs,
):
    """
    If don't want to show direct texture color without shading, set the light_kwargs as
    ambient_color=((1, 1, 1), ), diffuse_color=((0, 0, 0), ), specular_color=((0, 0, 0), )
    """
    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=5,
    )
    # We can add a point light in front of the object.
    lights = PointLights(
        device=device, location=light_location, **light_kwargs
    )
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )

    return phong_renderer


def create_mesh_normal_renderer(
        cameras: FoVPerspectiveCameras,
        image_size: int = 256,
        blur_radius: float = 1e-6,
        light_location=((-0.5, 1., 5.0),),
        device="cuda",
        **light_kwargs,
):
    """
    If don't want to show direct texture color without shading, set the light_kwargs as
    ambient_color=((1, 1, 1), ), diffuse_color=((0, 0, 0), ), specular_color=((0, 0, 0), )
    """
    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=5,
    )
    # We can add a point light in front of the object.
    lights = PointLights(
        device=device, location=light_location, **light_kwargs
    )
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=NormalShader(device=device)
    )

    return phong_renderer


## custom renderer
class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


def create_depth_mesh_renderer(
        cameras: FoVPerspectiveCameras,
        image_size: int = 256,
        blur_radius: float = 1e-6,
        light_location=((-0.5, 1., 5.0),),
        device="cuda",
        **light_kwargs,
):
    """
    If don't want to show direct texture color without shading, set the light_kwargs as
    ambient_color=((1, 1, 1), ), diffuse_color=((0, 0, 0), ), specular_color=((0, 0, 0), )
    """
    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    # raster_settings = RasterizationSettings(
    #     image_size=image_size,
    #     blur_radius=blur_radius,
    #     faces_per_pixel=17,
    # )
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    # We can add a point light in front of the object.
    lights = PointLights(
        device=device, location=light_location, **light_kwargs
    )
    renderer = MeshRendererWithDepth(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),
        # shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )

    return renderer


def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]


def tv_loss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)/batch_size

def P2sRt(P):
    """ decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    """
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d

def matrix2angle(R):
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    todo: check and debug
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
     """
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    return x, y, z

def calc_pose(param):
    P = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(P)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle(R)
    pose = [p * 180 / np.pi for p in pose]

    return P, pose

def grid_sample(image, optical):

    # Ref: https://github.com/pytorch/pytorch/issues/34704#issuecomment-878940122

    #print(image.shape, optical.shape)
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    ix = torch.nan_to_num(ix, float('inf'))
    iy = torch.nan_to_num(iy, float('inf'))

    ix = ((ix + 1) / 2) * (IW-1)
    iy = ((iy + 1) / 2) * (IH-1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)
    
    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

def grid_sample_3d(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1);
    iy = ((iy + 1) / 2) * (IH - 1);
    iz = ((iz + 1) / 2) * (ID - 1);
    with torch.no_grad():
        
        ix_tnw = torch.floor(ix);
        iy_tnw = torch.floor(iy);
        iz_tnw = torch.floor(iz);

        ix_tne = ix_tnw + 1;
        iy_tne = iy_tnw;
        iz_tne = iz_tnw;

        ix_tsw = ix_tnw;
        iy_tsw = iy_tnw + 1;
        iz_tsw = iz_tnw;

        ix_tse = ix_tnw + 1;
        iy_tse = iy_tnw + 1;
        iz_tse = iz_tnw;

        ix_bnw = ix_tnw;
        iy_bnw = iy_tnw;
        iz_bnw = iz_tnw + 1;

        ix_bne = ix_tnw + 1;
        iy_bne = iy_tnw;
        iz_bne = iz_tnw + 1;

        ix_bsw = ix_tnw;
        iy_bsw = iy_tnw + 1;
        iz_bsw = iz_tnw + 1;

        ix_bse = ix_tnw + 1;
        iy_bse = iy_tnw + 1;
        iz_bse = iz_tnw + 1;

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);


    with torch.no_grad():

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val


# Extract mesh from density with marching cubes
def extract_mesh_from_density_with_marching_cubes(sigma, sigma_threshold):
    b, h, w, d, _ = sigma.shape

    # change coordinate order from (y,x,z) to (x,y,z)
    sigma_vol = sigma[0,...,0].permute(1,0,2).cpu().numpy()

    # scale vertices
    verts, faces = mcubes.marching_cubes(sigma_vol, sigma_threshold)
    verts[:,0] = (verts[:,0]/float(w)-0.5)*0.24
    verts[:,1] = (verts[:,1]/float(h)-0.5)*0.24
    verts[:,2] = (verts[:,2]/float(d)-0.5)*0.24

    # fix normal direction
    verts[:,2] *= -1
    verts[:,1] *= -1
    mesh = trimesh.Trimesh(verts, faces)

    return mesh


def get_canonical_pose():
    # Define Da-pose
    pose_canonical = torch.from_numpy(np.array([ 
        0.0000,  0.0000,  0.5000,  0.0000,  0.0000, -0.5000,  0.0000,  0.0000,
        0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000,  0.0000,  0.0000,  0.0000,  0.0000])).float()
    
    # Define T-pose
    # pose_canonical = torch.from_numpy(np.array([ 
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000])).float()

    # Define NeuMen-pose
    # pose_canonical = torch.from_numpy(np.array([ 
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000])).float()

    # Define A-pose
    # pose_canonical = torch.from_numpy(np.array([ 
    #   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    #   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    #   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    #   0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
    #   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])).float()

    # Define mpv avg-pose # NOTE: need to re-compute based on pare estimation
    # pose_canonical = torch.from_numpy(np.array([
    #  -8.6796e-02,  1.7853e-02,  3.4279e-02, -1.1798e-01, -2.5379e-02,
    #  -3.1753e-02,  1.8608e-01,  3.3509e-03, -7.0832e-03,  3.6292e-01,
    #  -7.6810e-02, -4.3281e-02,  4.0448e-01,  6.1843e-02,  3.4722e-02,
    #  -2.4436e-03, -1.6713e-02,  1.1871e-02,  3.0730e-02,  1.8517e-01,
    #  -9.5188e-02,  1.6661e-02, -2.1843e-01,  1.5958e-01,  5.7923e-02,
    #  -2.0970e-02,  9.1754e-04, -2.1429e-01,  9.9760e-02,  1.5397e-01,
    #  -9.4491e-02,  9.0547e-02, -2.3808e-01, -6.0367e-02, -4.3389e-02,
    #   3.3936e-02,  4.1307e-02, -7.0583e-02, -3.9037e-01,  1.7194e-02,
    #   1.1326e-01,  4.0724e-01,  1.1830e-01, -7.4102e-02,  2.4201e-03,
    #   1.6863e-01, -2.4240e-01, -9.6669e-01,  2.2869e-01,  2.4341e-01,
    #   9.5056e-01,  1.7584e-01, -6.6856e-01,  1.3077e-01,  6.8550e-02,
    #   6.4382e-01, -1.3920e-01, -1.6693e-02, -5.1269e-02,  5.0651e-02,
    #   1.2155e-02,  5.6685e-02, -6.2308e-02, -1.6776e-01, -8.7323e-02,
    #  -2.0540e-01, -1.2860e-01,  1.1626e-01,  2.2188e-01])).float()

    return pose_canonical