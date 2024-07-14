import numpy as np
import imageio
import os
import torch
import argparse
from tqdm import tqdm
from .renderer import get_renderer
import utils.rotation_conversions as geometry
from utils.misc import to_torch
from model.rotation2xyz import Rotation2xyz, Rotation2xyz_x
from scipy.ndimage import gaussian_filter1d

def get_rotation(theta=np.pi/3):
    axis = torch.tensor([0, 1, 0], dtype=torch.float)
    axisangle = theta*axis
    matrix = geometry.axis_angle_to_matrix(axisangle)
    return matrix.numpy()


def render_video(meshes, key, action, renderer, savepath, background, num_person, cam=(0.75, 0.75, 0, 0.10), color=[0.11, 0.53, 0.8]):
    writer = imageio.get_writer(savepath, fps=30)
    # center the first frame
    print(meshes.shape, meshes[0].shape, meshes[0].mean(axis=0).shape)
    mean_value = meshes[0,:,0:3].mean(axis=0)
    for ii in range(num_person):
        meshes[:,:,3*ii:3*ii+3] = meshes[:,:,3*ii:3*ii+3] - mean_value
    imgs = []
    for mesh in tqdm(meshes, desc=f"Visualize {key}, action {action}"):
        img = renderer.render(background, mesh, cam, color=color)
        imgs.append(img)
        # show(img)

    imgs = np.array(imgs)
    masks = ~(imgs/255. > 0.96).all(-1)

    coords = np.argwhere(masks.sum(axis=0))
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)

    for cimg in imgs[:, y1:y2, x1:x2]:
        writer.append_data(cimg)
    writer.close()

def get_rotation(view):
    theta = - view * np.pi/4
    # axis = torch.tensor([0, 1, 0], dtype=torch.float)
    axis = torch.tensor([1, 0, 0], dtype=torch.float)
    axisangle = theta*axis
    matrix = geometry.axis_angle_to_matrix(axisangle)
    return matrix

def get_sample(data_path, num_person, body_model):
    params = {"pose_rep": "rot6d",
            "translation": True,
            "glob": True,
            "jointstype": "a2m",
            "vertstrans": True,
            "num_frames": 60,
            "sampling": "conseq",
            "sampling_step": 1}
    param2xyz = {"pose_rep": 'rot6d',
                "glob_rot": [3.141592653589793, 0, 0],
                "glob": True,
                "jointstype": 'vertices',
                "translation": True,
                "vertstrans": True,
                "num_person": 1, 
                "fixrot": False}
    if body_model == 'smpl':
        rot2xyz = Rotation2xyz(device='cuda')
    elif body_model == 'smplx':
        rot2xyz = Rotation2xyz_x(device='cuda')

    data = np.load(data_path, allow_pickle=True)
    action_text = data.item()['text']
    cmo  = data.item()['cmotion']
    data = data.item()['output'] # (1, 25, 6, 60)
    data = np.concatenate((cmo, data), axis=2)
    data = gaussian_filter1d(data, sigma=3, axis=-1)
    param2xyz.update({'num_person': num_person})
    frame_ix = range(60)
    ret_xyzs = []
    ret_actions = []
    for ind in range(data.shape[0]):
        ret = to_torch(data[ind])
        mask = torch.ones((1, ret.shape[-1]), dtype=torch.bool).cuda()
        ret_xyz = rot2xyz(ret.unsqueeze(0).cuda(), mask, **param2xyz) 
        ret_xyzs.append(ret_xyz)
        ret_actions.append(action_text[ind])
    return ret_xyzs, ret_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    # parser.add_argument("--savefolder")
    parser.add_argument("--num_person")
    parser.add_argument("--setting", default='mdm', choices=['mdm', 'cmdm'], type=str,
                       help="Use mdm mode for interactive motion generation or cmdm for conditional motion generation")
    parser.add_argument("--body_model", default='smpl', choices=['smpl', 'smplx'], type=str,
                       help="Use SMPL model or SMPl-X model.")
    opt = parser.parse_args()
    data_path = opt.data_path
    num_person = int(opt.num_person)

    output, actions = get_sample(data_path, num_person, opt.body_model)


    width = 1024
    height = 1024

    background = np.zeros((height, width, 3))
    renderer = get_renderer(width, height, setting=opt.setting, body_model=opt.body_model)

    savefolder = os.path.join(data_path.rsplit('/', 1)[0], 'rendered')
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    print(len(output))
    for ind in range(0, len(output)):
        vidmeshes = output[ind].squeeze() # [6890, 3, 60]
        action = actions[ind]
        # print("vidmeshes ", vidmeshes.shape)
        # meshes = vidmeshes.transpose(2, 0, 1)
        meshes = torch.permute(vidmeshes, (2, 0, 1)).cpu()
        # meshes: [T, 6890, 3]
        path = os.path.join(savefolder, "{}_{}.mp4".format(action, ind))
        render_video(meshes, ind, action, renderer, path, background, num_person)


if __name__ == "__main__":
    main()
