import os

SMPL_DATA_PATH = "models/smpl/"
SMPLX_DATA_PATH = "models/smplx/"

SMPL_KINTREE_PATH = os.path.join(SMPL_DATA_PATH, "kintree_table.pkl")
SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "SMPL_NEUTRAL.pkl")
SMPLX_MODEL_PATH = SMPLX_DATA_PATH
SMPLX_KINTREE_PATH = os.path.join(SMPLX_DATA_PATH, "SMPLX_NEUTRAL.npz")

JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DATA_PATH, 'J_regressor_extra.npy')
