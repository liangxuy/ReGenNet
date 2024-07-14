from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import ccollate as all_ccollate
from data_loaders.tensors import t2m_collate

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "ntu" or name == "chi3d":
        from .a2m.feeder import Feeder
        return Feeder
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')


def get_collate_fn(name, setting, hml_mode='train'):
    if setting == 'mdm':
        if hml_mode == 'gt':
            from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
            return t2m_eval_collate
        if name in ["humanml", "kit"]:
            return t2m_collate
        else:
            return all_collate
    elif setting == 'cmdm':
        return all_ccollate


def get_dataset(name, num_frames, num_person, data_path='', pose_rep='rot6d', body_model='smpl', ar_shuffle=False, split='train', hml_mode='train', shard=0, num_shards=1):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, num_person=num_person, datapath=data_path, pose_rep=pose_rep, body_model=body_model, dataname=name, ar_shuffle=ar_shuffle, shard=shard, num_shards=num_shards)
    else:
        dataset = DATA(split=split, num_frames=num_frames, num_person=num_person, datapath=data_path, pose_rep=pose_rep, dataname=name, body_model=body_model, ar_shuffle=ar_shuffle, shard=shard, num_shards=num_shards)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, num_person, data_path='', pose_rep='rot6d', body_model='smpl', ar_shuffle=False, setting='mdm', split='train', hml_mode='train', shard=0, num_shards=1):
    dataset = get_dataset(name, num_frames, num_person, data_path, pose_rep, body_model, ar_shuffle, split, hml_mode, shard=shard, num_shards=num_shards)
    collate = get_collate_fn(name, setting, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate, persistent_workers=True
    )

    return loader
