import copy
import torch
from tqdm import tqdm
import functools
import numpy as np
import os
from utils.fixseed import fixseed

from eval.a2m.stgcn.evaluate import Evaluation as STGCNEvaluation
from torch.utils.data import DataLoader
from data_loaders.tensors import collate, ccollate

from .tools import format_metrics
import utils.rotation_conversions as geometry
from utils import dist_util

def convert_x_to_rot6d(x, pose_rep):
    # convert rotation to rot6d
    if pose_rep == "rotvec":
        x = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(x))
    elif pose_rep == "rotmat":
        x = x.reshape(*x.shape[:-1], 3, 3)
        x = geometry.matrix_to_rotation_6d(x)
    elif pose_rep == "rotquat":
        x = geometry.matrix_to_rotation_6d(geometry.quaternion_to_matrix(x))
    elif pose_rep == "rot6d":
        x = x
    else:
        raise NotImplementedError("No geometry for this one.")
    return x


class NewDataloader:
    def __init__(self, mode, model, diffusion, dataiterator, device, dataset, num_samples, num_person, body_model, setting, auto_regressive=False):
        assert mode in ["gen", "gt"]

        self.batches = []
        sample_fn = diffusion.p_sample_loop

        with torch.no_grad():
            for motions, model_kwargs in tqdm(dataiterator, desc=f"Construct dataloader: {mode}.."):
                motions = motions.to(device)
                if num_samples != -1 and len(self.batches) * dataiterator.batch_size > num_samples:
                    continue  # do not break because it confuses the multiple loaders
                batch = dict()
                if mode == "gen":
                    for _k in model_kwargs['y'].keys():
                        if type(model_kwargs['y'][_k]) == torch.Tensor:
                            model_kwargs['y'][_k] = model_kwargs['y'][_k].to(device)
                    if auto_regressive == True:
                        cmotion_bak = model_kwargs['y']['cmotion']
                        B, V, C, T = cmotion_bak.shape
                        cmotion = torch.zeros_like(model_kwargs['y']['cmotion']).to(device)
                        if setting == 'cmdm':
                            output = torch.zeros((B, V, C*2, T)).to(device)
                        else:
                            output = torch.zeros((B, V, C, T)).to(device)
                        for frame_idx in range(cmotion.shape[-1]):
                            cmotion[:,:,:,frame_idx] = cmotion_bak[:,:,:,frame_idx]
                            model_kwargs['y']['cmotion'] = cmotion
                            sample = sample_fn(model, motions.shape, clip_denoised=False, model_kwargs=model_kwargs)
                            if setting == 'cmdm':
                                tmp = torch.cat((model_kwargs['y']['cmotion'], sample), axis=2)
                            else:
                                tmp = sample
                            output[:,:,:,frame_idx] = tmp[:,:,:,frame_idx]
                        batch['output'] = output
                    else:
                        sample = sample_fn(model, motions.shape, clip_denoised=False, model_kwargs=model_kwargs)
                        if setting == 'cmdm':
                            batch['output'] = torch.cat((model_kwargs['y']['cmotion'], sample), axis=2)
                        else:
                            batch['output'] = sample
                    batch['text'] = model_kwargs['y']['action_text']
                elif mode == "gt":
                    batch['output'] = motions

                max_n_frames = model_kwargs['y']['lengths'].max()
                mask = model_kwargs['y']['mask'].reshape(dataiterator.batch_size, max_n_frames).bool()

                batch["output_xyz"] = model.rot2xyz(x=batch["output"], mask=mask, pose_rep='rot6d', glob=True,
                                                    translation=True, jointstype=body_model, vertstrans=True, betas=None,
                                                    beta=0, glob_rot=None, get_rotations_back=False, num_person=num_person)
                ### Modification by Derek, 2023.03.23, for multi-person, root translations matter
                # if model.translation:
                #     # the stgcn model expects rotations only
                #     batch["output"] = batch["output"][:, :-1]

                batch["lengths"] = model_kwargs['y']['lengths'].to(device)
                # using torch.long so lengths/action will be used as indices
                batch["y"] = model_kwargs['y']['action'].squeeze().long().cpu()  # using torch.long so lengths/action will be used as indices
                self.batches.append(batch)

            num_samples_last_batch = num_samples % dataiterator.batch_size
            if num_samples_last_batch > 0:
                for k, v in self.batches[-1].items():
                    self.batches[-1][k] = v[:num_samples_last_batch]
            # if mode == 'gen':
            #     split = dataiterator.dataset.split
            #     outputs = []
            #     cmotions = []
            #     texts = []
            #     for idx in range(len(self.batches)):
            #         outputs.append(self.batches[idx]['output'][:,:,6:12,:].cpu())
            #         cmotions.append(self.batches[idx]['output'][:,:,0:6,:].cpu())
            #         texts.append(self.batches[idx]['text'])
            #     outputs = np.concatenate(outputs, axis=0)
            #     cmotions = np.concatenate(cmotions, axis=0)
            #     texts = np.concatenate(texts, axis=0)
            #     if not os.path.exists('vis_data'):
            #         os.makedirs('./vis_data')
            #     filename = dataiterator.dataset.data_path.split('/')[-1]
            #     np.save(os.path.join('./vis_data', '{}_split_{}_{}.npy'.format(dataset, split, filename)), {'cmotion': cmotions, 'output': outputs, 'text': texts})


    def __iter__(self):
        return iter(self.batches)


def evaluate(args, model, diffusion, data, rec_model_path, setting, acc_only, auto_regressive=False):
    torch.multiprocessing.set_sharing_strategy('file_system')

    bs = args.batch_size
    if args.dataset == 'ntu':
        args.num_classes = 26
        args.nfeats = 6
        # args.model_path = '/mnt/yardcephfs/mmyard/g_wxg_td_mmk/lxxu/projects/actor-x/recognition_training/ntu_smplx_cgen/checkpoint_0100.pth.tar'
    elif args.dataset == 'chi3d':
        args.num_classes = 8
        args.nfeats = 6
    args.model_path = rec_model_path
    
    device = dist_util.dev()

    recogparameters = args.__dict__.copy()
    recogparameters["pose_rep"] = args.pose_rep
    recogparameters["nfeats"] = args.nfeats * 2
    recogparameters["model_path"] = args.model_path
    recogparameters["num_person"] = 2 # for cmdm, also 2

    stgcnevaluation = STGCNEvaluation(args.dataset, args.body_model, recogparameters, device)

    stgcn_metrics = {}

    data_types = ['train', 'test']
    datasetGT = {'train': [data], 'test': [copy.deepcopy(data)]}

    for key in data_types:
        datasetGT[key][0].split = key

    compute_gt_gt = False #False
    if compute_gt_gt:
        for key in data_types:
            datasetGT[key].append(copy.deepcopy(datasetGT[key][0]))

    model.eval()

    allseeds = list(range(args.num_seeds))

    for index, seed in enumerate(allseeds):
        print(f"Evaluation number: {index + 1}/{args.num_seeds}")
        fixseed(seed)
        for key in data_types:
            for data in datasetGT[key]:
                data.reset_shuffle()
                data.shuffle()

        dataiterator = {key: [DataLoader(data, batch_size=bs, shuffle=False, num_workers=8, drop_last=True, collate_fn=collate)
                            for data in datasetGT[key]]
                        for key in data_types}
        dataiterator_con = {key: [DataLoader(data, batch_size=bs, shuffle=False, num_workers=8, drop_last=True, collate_fn=ccollate)
                            for data in datasetGT[key]]
                        for key in data_types}

        new_data_loader = functools.partial(NewDataloader, model=model, diffusion=diffusion, device=device,
                                            dataset=args.dataset, num_samples=args.num_samples, num_person=2, body_model=args.body_model, setting=setting, auto_regressive=auto_regressive)
        gtLoaders = {key: new_data_loader(mode="gt", dataiterator=dataiterator[key][0])
                     for key in ["train", "test"]}

        if compute_gt_gt:
            gtLoaders2 = {key: new_data_loader(mode="gt", dataiterator=dataiterator[key][0])
                          for key in ["train", "test"]}

        if setting == 'cmdm':
            genLoaders = {key: new_data_loader(mode="gen", dataiterator=dataiterator_con[key][0])
                        for key in ["train", "test"]}
        elif setting == 'mdm':
            genLoaders = {key: new_data_loader(mode="gen", dataiterator=dataiterator[key][0])
                        for key in ["train", "test"]}

        loaders = {"gen": genLoaders,
                   "gt": gtLoaders}

        if compute_gt_gt:
            loaders["gt2"] = gtLoaders2

        if not acc_only:
            stgcn_metrics[seed] = stgcnevaluation.evaluate(model, loaders, setting)
        else:
            stgcn_metrics[seed] = stgcnevaluation.evaluate_acc(model, loaders, setting)
        del loaders

    metrics = {"feats": {key: [format_metrics(stgcn_metrics[seed])[key] for seed in allseeds] for key in stgcn_metrics[allseeds[0]]}}

    return metrics
