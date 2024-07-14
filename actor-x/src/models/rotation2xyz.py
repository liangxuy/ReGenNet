import torch
import src.utils.rotation_conversions as geometry
import numpy as np
import json as js

from .smpl import SMPL, SMPLX, JOINTSTYPE_ROOT
from .get_model import JOINTSTYPES

class Rotation2xyz:
    def __init__(self, device):
        self.device = device
        self.smpl_model = SMPL().eval().to(device)

    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, num_person=1, fixrot=False, **kwargs):
        if pose_rep == "xyz":
            return x

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)

        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")

        if jointstype not in JOINTSTYPES:
            raise NotImplementedError("This jointstype is not implemented.")

        if num_person > 1:
            num_dim = x.shape[2] // num_person
            x_p = torch.split(x, num_dim, dim=2)
            x_xyz_p = []
            for pid, x in enumerate(x_p):
                if translation:
                    x_translations = x[:, -1, :3]
                    x_rotations = x[:, :-1]
                else:
                    x_rotations = x

                x_rotations = x_rotations.permute(0, 3, 1, 2)
                nsamples, time, njoints, feats = x_rotations.shape

                # Compute rotations (convert only masked sequences output)
                if pose_rep == "rotvec":
                    rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
                elif pose_rep == "rotmat":
                    rotations = x_rotations[mask].view(-1, njoints, 3, 3)
                elif pose_rep == "rotquat":
                    rotations = geometry.quaternion_to_matrix(x_rotations[mask])
                elif pose_rep == "rot6d":
                    rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
                else:
                    raise NotImplementedError("No geometry for this one.")

                if not glob:
                    global_orient = torch.tensor(glob_rot, device=x.device)
                    global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
                    global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
                else:
                    if not fixrot:
                        global_orient = rotations[:, 0]
                    else: # fix global rotation during training
                        global_orient = torch.tensor(glob_rot, device=x.device)
                        global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
                        global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
                    rotations = rotations[:, 1:]

                if betas is None:
                    betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
                                        dtype=rotations.dtype, device=rotations.device)
                    betas[:, 1] = beta
                
                out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)

                # get the desirable joints
                joints = out[jointstype]

                x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
                x_xyz[~mask] = 0
                x_xyz[mask] = joints

                x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

                # the first translation root at the origin on the prediction
                if jointstype != "vertices":
                    rootindex = JOINTSTYPE_ROOT[jointstype]
                    x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

                if translation and vertstrans:
                    # add the translation to all the joints
                    x_xyz = x_xyz + x_translations[:, None, :, :]
                x_xyz_p.append(x_xyz)
            x_xyz = torch.cat(x_xyz_p, 2)
        else:
            if translation:
                x_translations = x[:, -1, :3]
                x_rotations = x[:, :-1]
            else:
                x_rotations = x

            x_rotations = x_rotations.permute(0, 3, 1, 2)
            nsamples, time, njoints, feats = x_rotations.shape

            # Compute rotations (convert only masked sequences output)
            if pose_rep == "rotvec":
                rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
            elif pose_rep == "rotmat":
                rotations = x_rotations[mask].view(-1, njoints, 3, 3)
            elif pose_rep == "rotquat":
                rotations = geometry.quaternion_to_matrix(x_rotations[mask])
            elif pose_rep == "rot6d":
                rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
            else:
                raise NotImplementedError("No geometry for this one.")

            if not glob:
                global_orient = torch.tensor(glob_rot, device=x.device)
                global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
                global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
            else:
                if not fixrot:
                    global_orient = rotations[:, 0]
                else: # fix global rotation during training
                    global_orient = torch.tensor(glob_rot, device=x.device)
                    global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
                    global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
                rotations = rotations[:, 1:]

            if betas is None:
                betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
                                    dtype=rotations.dtype, device=rotations.device)
                betas[:, 1] = beta
                
            out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)
            # get the desirable joints
            joints = out[jointstype]

            x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
            x_xyz[~mask] = 0
            x_xyz[mask] = joints

            x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

            # the first translation root at the origin on the prediction
            if jointstype != "vertices":
                rootindex = JOINTSTYPE_ROOT[jointstype]
                x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

            if translation and vertstrans:
                # the first translation root at the origin
                x_translations = x_translations - x_translations[:, :, [0]]

                # add the translation to all the joints
                x_xyz = x_xyz + x_translations[:, None, :, :]


        return x_xyz


class Rotation2xyz_x:
    def __init__(self, device):
        self.device = device
        self.smplx_model = SMPLX().eval().to(device)


    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, num_person=1, fixrot=False, **kwargs):
        if pose_rep == "xyz":
            return x

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)

        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")

        if jointstype not in JOINTSTYPES:
            raise NotImplementedError("This jointstype is not implemented.")

        if num_person > 1: # [1, 25, 12, 60]
            num_dim = x.shape[2] // num_person
            x_p = torch.split(x, num_dim, dim=2)
            x_xyz_p = []
            for pid, x in enumerate(x_p):
                if translation:
                    x_translations = x[:, -1, :3]
                    x_rotations = x[:, :-1]
                else:
                    x_rotations = x

                x_rotations = x_rotations.permute(0, 3, 1, 2)
                nsamples, time, njoints, feats = x_rotations.shape # [1, 60, 55, 6]

                # Compute rotations (convert only masked sequences output)
                if pose_rep == "rotvec":
                    rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
                elif pose_rep == "rotmat":
                    rotations = x_rotations[mask].view(-1, njoints, 3, 3)
                elif pose_rep == "rotquat":
                    rotations = geometry.quaternion_to_matrix(x_rotations[mask])
                elif pose_rep == "rot6d":
                    rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
                else:
                    raise NotImplementedError("No geometry for this one.")

                if not glob:
                    global_orient = torch.tensor(glob_rot, device=x.device)
                    global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
                    global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
                else:
                    if not fixrot:
                        global_orient = rotations[:, 0, None]
                    else: # fix global rotation during training
                        global_orient = torch.tensor(glob_rot, device=x.device)
                        global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
                        global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
                    rotations = rotations[:, 1:]

                body_pose = rotations[:, 0:21]
                jaw_pose = rotations[:, 21:22]
                leye_pose = rotations[:, 22:23]
                reye_pose = rotations[:, 23:24]
                left_hand_pose = rotations[:, 24:39]
                right_hand_pose = rotations[:, 39:54]
                # print("444, ", body_pose.shape, jaw_pose.shape, leye_pose.shape, reye_pose.shape, left_hand_pose.shape, right_hand_pose.shape, self.smplx_model.num_betas)

                if betas is None:
                    betas = torch.zeros([rotations.shape[0], self.smplx_model.num_betas],
                                        dtype=rotations.dtype, device=rotations.device)
                    betas[:, 1] = beta
                """
                motion_file = '/data/xuliang/actformer/datasets/chi3d/train/s02/smplx/Grab 1.json'
                motion = js.load(open(motion_file))
                for k, v in motion.items():
                    if type(v) is float:
                        print(k, v)
                    else:
                        print(k, np.array(v).shape)
                global_orient = np.array(motion['global_orient'])[pid].squeeze() # [209, 1, 3, 3]
                body_pose = np.array(motion['body_pose'])[pid]
                betas = np.array(motion['betas'])[pid] # [209, 10]
                left_hand_pose = np.array(motion['left_hand_pose'])[pid] # [209, 15, 3, 3]
                right_hand_pose = np.array(motion['right_hand_pose'])[pid] # [209, 15, 3, 3]
                expression = np.array(motion['expression'])[pid] # [209, 10]
                n = betas.shape[pid]

                # (219, 10) (219, 10) (219, 21, 3, 3) (219, 3, 3) (219, 15, 3, 3) (219, 15, 3, 3)
                print(betas.shape, expression.shape, body_pose.shape, global_orient.shape, left_hand_pose.shape, right_hand_pose.shape)
                betas = torch.from_numpy(betas).to(torch.float32).to(self.device)
                expression = torch.from_numpy(expression).to(torch.float32).to(self.device)
                body_pose = torch.from_numpy(body_pose).to(torch.float32).to(self.device)
                left_hand_pose = torch.from_numpy(left_hand_pose).to(torch.float32).to(self.device)
                right_hand_pose = torch.from_numpy(right_hand_pose).to(torch.float32).to(self.device)
                global_orient = torch.from_numpy(global_orient).to(torch.float32).to(self.device)
                """
                out = self.smplx_model(
                                betas=betas,
                                # expression=expression,
                                body_pose=body_pose, 
                                left_hand_pose=left_hand_pose, 
                                right_hand_pose=right_hand_pose, 
                                global_orient=global_orient, 
                                return_verts=True)
                # get the desirable joints
                joints = out[jointstype]
                x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
                x_xyz[~mask] = 0
                x_xyz[mask] = joints
                x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

                # the first translation root at the origin on the prediction
                if jointstype != "vertices":
                    rootindex = JOINTSTYPE_ROOT[jointstype]
                    x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

                if translation and vertstrans:
                    # add the translation to all the joints
                    x_xyz = x_xyz + x_translations[:, None, :, :]
                x_xyz_p.append(x_xyz)
            x_xyz = torch.cat(x_xyz_p, 2)
        else:
            if translation:
                x_translations = x[:, -1, :3]
                x_rotations = x[:, :-1]
            else:
                x_rotations = x

            x_rotations = x_rotations.permute(0, 3, 1, 2)
            nsamples, time, njoints, feats = x_rotations.shape

            # Compute rotations (convert only masked sequences output)
            if pose_rep == "rotvec":
                rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
            elif pose_rep == "rotmat":
                rotations = x_rotations[mask].view(-1, njoints, 3, 3)
            elif pose_rep == "rotquat":
                rotations = geometry.quaternion_to_matrix(x_rotations[mask])
            elif pose_rep == "rot6d":
                rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
            else:
                raise NotImplementedError("No geometry for this one.")

            if not glob:
                global_orient = torch.tensor(glob_rot, device=x.device)
                global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
                global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
            else:
                if not fixrot:
                    global_orient = rotations[:, 0]
                else: # fix global rotation during training
                    global_orient = torch.tensor(glob_rot, device=x.device)
                    global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
                    global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
                rotations = rotations[:, 1:]

            if betas is None:
                betas = torch.zeros([rotations.shape[0], self.smplx_model.num_betas],
                                    dtype=rotations.dtype, device=rotations.device)
                betas[:, 1] = beta
            
            out = self.smplx_model(body_pose=rotations, global_orient=global_orient, betas=betas)
            # get the desirable joints
            joints = out[jointstype]

            x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
            x_xyz[~mask] = 0
            x_xyz[mask] = joints

            x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

            # the first translation root at the origin on the prediction
            if jointstype != "vertices":
                rootindex = JOINTSTYPE_ROOT[jointstype]
                x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

            if translation and vertstrans:
                # the first translation root at the origin
                x_translations = x_translations - x_translations[:, :, [0]]

                # add the translation to all the joints
                x_xyz = x_xyz + x_translations[:, None, :, :]


        return x_xyz
