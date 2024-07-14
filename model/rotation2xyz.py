# This code is based on https://github.com/Mathux/ACTOR.git
import torch
import utils.rotation_conversions as geometry


from model.smpl import SMPL, SMPLX, JOINTSTYPE_ROOT
# from .get_model import JOINTSTYPES
JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "smplx", "vertices"]


class Rotation2xyz:
    def __init__(self, device, dataset='amass'):
        self.device = device
        self.dataset = dataset
        self.smpl_model = SMPL().eval().to(device)

    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, num_person=1, get_rotations_back=False, **kwargs):
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
                    print(x_rotations.shape, mask.shape, x_rotations[mask].shape)
                    rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
                else:
                    raise NotImplementedError("No geometry for this one.")

                if not glob:
                    global_orient = torch.tensor(glob_rot, device=x.device)
                    global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
                    global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
                else:
                    global_orient = rotations[:, 0]
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
                    # x_translations = x_translations - x_translations[:, :, [0]]

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
                global_orient = rotations[:, 0]
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

        if get_rotations_back:
            return x_xyz, rotations, global_orient
        else:
            return x_xyz


class Rotation2xyz_x:
    def __init__(self, device, dataset='amass'):
        self.device = device
        self.dataset = dataset
        self.smpl_model = SMPLX().eval().to(device)


    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, num_person=1, **kwargs):
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
                    global_orient = rotations[:, 0, None]
                    rotations = rotations[:, 1:]

                body_pose = rotations[:, 0:21]
                jaw_pose = rotations[:, 21:22]
                leye_pose = rotations[:, 22:23]
                reye_pose = rotations[:, 23:24]
                left_hand_pose = rotations[:, 24:39]
                right_hand_pose = rotations[:, 39:54]
                # print("444, ", body_pose.shape, jaw_pose.shape, leye_pose.shape, reye_pose.shape, left_hand_pose.shape, right_hand_pose.shape, self.smpl_model.num_betas)

                if betas is None:
                    betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
                                        dtype=rotations.dtype, device=rotations.device)
                    betas[:, 1] = beta

                out = self.smpl_model(
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
                global_orient = rotations[:, 0]
                rotations = rotations[:, 1:]

            body_pose = rotations[:, 0:21]
            jaw_pose = rotations[:, 21:22]
            leye_pose = rotations[:, 22:23]
            reye_pose = rotations[:, 23:24]
            left_hand_pose = rotations[:, 24:39]
            right_hand_pose = rotations[:, 39:54]

            if betas is None:
                betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
                                    dtype=rotations.dtype, device=rotations.device)
                betas[:, 1] = beta
            
            out = self.smpl_model(
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
                # the first translation root at the origin
                x_translations = x_translations - x_translations[:, :, [0]]

                # add the translation to all the joints
                x_xyz = x_xyz + x_translations[:, None, :, :]


        return x_xyz
