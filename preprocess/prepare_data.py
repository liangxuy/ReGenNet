import os
import pickle
import h5py
import torch
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append('./')
import utils.rotation_conversions as geometry

SRC = '/data/xuliang/NTU/ntu120_2p_rgb/pymaf-x'
DEST_H5 = 'dataset/ntu120/smplx/ntu_2p_smplx.h5'

def get_rotation(view):
    theta = - view * np.pi/4
    axis = torch.tensor([1, 0, 0], dtype=torch.float)
    axisangle = theta*axis
    matrix = geometry.axis_angle_to_matrix(axisangle)
    return matrix

def parse_motion_file(motion_file, rotation):
    data = joblib.load(motion_file)
    body_pose = []
    left_hand_pose = []
    right_hand_pose = []
    jaw_pose = []
    leye_pose = []
    reye_pose = []
    expression = []
    for batch in data['smplx_params']:
        body_pose.append(batch['body_pose'][:,0:22])
        left_hand_pose.append(batch['left_hand_pose'])
        right_hand_pose.append(batch['right_hand_pose'])
        jaw_pose.append(batch['jaw_pose'])
        leye_pose.append(batch['leye_pose'])
        reye_pose.append(batch['reye_pose'])
        expression.append(batch['expression'])

    body_pose = torch.cat(body_pose, axis=0) # [N, 22, 3, 3]
    left_hand_pose = torch.cat(left_hand_pose, axis=0)
    right_hand_pose = torch.cat(right_hand_pose, axis=0)
    jaw_pose = torch.cat(jaw_pose, axis=0)
    leye_pose = torch.cat(leye_pose, axis=0)
    reye_pose = torch.cat(reye_pose, axis=0)
    expression = torch.cat(expression, axis=0)

    body_pose = geometry.matrix_to_axis_angle(body_pose).cpu()
    left_hand_pose = geometry.matrix_to_axis_angle(left_hand_pose).cpu().numpy()
    right_hand_pose = geometry.matrix_to_axis_angle(right_hand_pose).cpu().numpy()
    jaw_pose = geometry.matrix_to_axis_angle(jaw_pose).cpu().numpy()
    leye_pose = geometry.matrix_to_axis_angle(leye_pose).cpu().numpy()
    reye_pose = geometry.matrix_to_axis_angle(reye_pose).cpu().numpy()

    root_transl = data['orig_cam_t']
    root_transl[:,-1] = root_transl[:, -1] / 20
    root_transl = root_transl @ rotation.T.numpy() # perform rotation here

    global_matrix = geometry.axis_angle_to_matrix(body_pose[:,0])
    body_pose[:,0] = geometry.matrix_to_axis_angle(rotation @ global_matrix)
    body_pose = body_pose.numpy()

    max_frame = data['frame_ids'][-1] + 1

    final_pose = np.zeros((2, max_frame, 56, 3))
    assert len(data['person_ids']) == body_pose.shape[0]
    for idx in range(len(data['person_ids'])):
        splits = data['person_ids'][idx].split('_')
        frame_idx = int(splits[-2][1:])
        person_idx = int(splits[-1][1:])
        if person_idx > 1:
            continue
        tmp = np.concatenate((body_pose[idx], jaw_pose[idx], leye_pose[idx], reye_pose[idx], left_hand_pose[idx], right_hand_pose[idx], root_transl[idx, None]), axis=0)
        final_pose[person_idx, frame_idx,:,:] = tmp

    final_pose = final_pose.transpose((1, 2, 0, 3)).reshape((-1, 56, 6))
    return final_pose


if __name__ == '__main__':
    fw = h5py.File(DEST_H5, 'w')
    rotation = get_rotation(0)
    # Load all samples
    for action_class in sorted(os.listdir(SRC)):
        print(action_class)
        motion_files = sorted(os.listdir(os.path.join(SRC, action_class)))
        for seq_name in motion_files:
            motion_file = os.path.join(SRC, action_class, seq_name, 'output.pkl')
            camera_id = int(seq_name[5:8])
            if not os.path.exists(motion_file):
                continue
            if camera_id == 1:
                poses = parse_motion_file(motion_file, rotation)
                file_name = seq_name.split('_')[0]
                fw.create_dataset(file_name, data=poses, dtype='f4')
