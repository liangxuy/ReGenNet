import argparse
import yaml
import os
import numpy as np
import h5py

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# torchlight
from src.utils import torchlight
from src.utils.torchlight import import_class, DictAction, str2bool
from src.datasets.dataset import POSE_REPS

from ..rotation2xyz import Rotation2xyz

def loss_hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            m.weight.data.normal_(0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class GAN(nn.Module):
    """
    Processor for train Motion Generator.
    """
    def __init__(self, **argv):
        super().__init__()
        self.load_arg(argv)
        self.init_environment()

        self.load_model()
        self.load_data()
        self.load_optimizer()

        self.rotation2xyz = Rotation2xyz(device=self.dev)

        # ad hoc
        # self.param2xyz = {"pose_rep": 'rot6d',
        #                   "glob_rot": [3.141592653589793, 0, 0],
        #                   "glob": True,
        #                   "jointstype": 'vertices',
        #                   "translation": True,
        #                   "vertstrans": False}
        # self.param2xyz = {"pose_rep": 'rot6d', # NTU_Mocap 1p & babel smpl
        #                   "glob_rot": [3.141592653589793, 0, 0],
        #                   "glob": False,
        #                   "jointstype": 'vertices',
        #                   "translation": True,
        #                   "vertstrans": False}
        self.param2xyz = {"pose_rep": 'rot6d',
                          "glob_rot": [3.141592653589793, 0, 0],
                          "glob": True,
                          "jointstype": 'vertices',
                          "translation": True,
                          "vertstrans": True,
                          "num_person": 1, 
                          "fixrot": False}

    def rot2xyz(self, x, **kwargs):
        kargs = self.param2xyz.copy()
        kargs.update(kwargs)
        return self.rotation2xyz(x, mask=None, **kargs)

    def load_arg(self, argv=None):
        parser = self.get_parser()

        # load arg form config file
        p = parser.parse_args(argv)
        if p.config is not None:
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)

            # update parser from config file
            key = vars(p).keys()
            for k in default_arg.keys():
                assert k in key, 'Unknown Arguments: {}'.format(k)
            parser.set_defaults(**default_arg)

        arg = parser.parse_args(argv)
        self.arg = arg


    def init_environment(self):
        self.io = torchlight.IO(self.arg.work_dir, save_log=self.arg.save_log, print_log=self.arg.print_log)
        self.io.save_arg(self.arg)
        self.dev = "cuda:0"
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)


    def load_model(self):
        model_D = self.io.load_model(self.arg.model_D, **(self.arg.model_D_args))
        model_G = self.io.load_model(self.arg.model_G, **(self.arg.model_G_args))
        if self.arg.model_weights is not None:
            self.model = self.io.load_weights(nn.ModuleList([model_D, model_G]), self.arg.model_weights).to(self.dev)
        else:
            self.model = nn.ModuleList([model_D, model_G]).to(self.dev)
            self.model.apply(weights_init)

        self.global_step = 0


    def load_data(self):
        if self.arg.phase == 'train':
            Feeder = import_class(self.arg.feeder)
            if self.arg.resample:
                self.data_set = Feeder(**self.arg.train_feeder_args)
            else:
                self.data_loader = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.train_feeder_args),
                    batch_size=self.arg.batch_size,
                    shuffle=True,
                    num_workers=self.arg.num_worker * torchlight.ngpu(self.arg.device),
                    drop_last=True)


    def load_optimizer(self):
        self.optimizerD = optim.Adam(
            self.model[0].parameters(),
            lr=self.arg.base_lr * self.arg.D_lr_mult,
            betas=(self.arg.beta1, 0.999),
            weight_decay=self.arg.weight_decay)

        self.optimizerG = optim.Adam(
            self.model[1].parameters(),
            lr=self.arg.base_lr,
            betas=(self.arg.beta1, 0.999),
            weight_decay=self.arg.weight_decay)


    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info = f"\tIter {self.meta_info['iter']} Done."
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + f' | {k}: {v:.4f}'
                else:
                    info = info + f' | {k}: {v}'

            self.io.print_log(info)


    def show_epoch_info(self):
        keys = self.epoch_info.keys()
        values = [f'{v:.4f}' if isinstance(v, float) else v for v in self.epoch_info.values()]
        str_len = [max(len(k) + 2, len(v) + 2) for k, v in zip(keys, values)]

        keys = [''] + [f'{k:^{l}}' for k, l in zip(keys, str_len)] + ['']
        values = [''] + [f'{v:^{l}}' for v, l in zip(values, str_len)] + ['']
        line1 = [''] + ['-' * l for l in str_len] + ['']
        line2 = [''] + ['=' * l for l in str_len] + ['']

        key_str = '|'.join(keys)
        line1_str = '+'.join(line1)
        line2_str = '+'.join(line2)
        value_str = '|'.join(values)

        self.io.print_log(line1_str)
        self.io.print_log(key_str)
        self.io.print_log(line2_str)
        self.io.print_log(value_str)
        self.io.print_log(line1_str)


    def _get_cov(self, scale, length, level=2):
        i = np.tile(np.arange(length), (length, 1))
        j = i.transpose()
        r = np.abs(i - j)
        cov = np.exp(-(r / scale)**level)
        return cov


    def gen_noise(self, N, NN, Z, lambda_noise=1, mode='independent'):
        """
        Generate noise.
        Args:
            N: batch
            NN: num of motion element (noise)
            Z: noise channel
            mode: independent | gp
        """
        if mode == 'independent':
            return torch.cuda.FloatTensor(N, Z, 1, NN).normal_(0, 1)

        elif mode == 'constant':
            noise = torch.cuda.FloatTensor(N, Z, 1, 1).normal_(0, 1)
            return noise.expand(N, Z, 1, NN)

        elif mode == 'gp':
            noise = []
            for c in range(Z):
                scale = self.arg.length_scale * (c + 1) / Z
                cov = self._get_cov(scale, NN, level=2)
                mean = np.zeros(NN)
                n = lambda_noise * np.random.multivariate_normal(mean, cov, size=(N, 1))
                noise.append(n)
            noise = np.stack(noise, 1)
            assert noise.shape == (N, Z, 1, NN)
            return torch.from_numpy(noise).float().to(self.dev)

        elif mode == 'multi_gp':
            noise = []
            for c in range(Z):
                scale = self.arg.length_scale * (c + 1) / Z
                cov = self._get_cov(scale, NN, level=2)
                mean = np.zeros(NN)
                n = lambda_noise * np.random.multivariate_normal(mean, cov, size=(N, self.arg.n_person))
                noise.append(n)
            noise = np.stack(noise, 1)
            assert noise.shape == (N, Z, self.arg.n_person, NN)
            return torch.from_numpy(noise).float().to(self.dev)

        elif mode == 'gaussian':
            noise = np.random.normal(size=(N, Z))
            return torch.from_numpy(noise).float().to(self.dev)

        elif mode == 'gp_single_scale':
            noise = []
            for c in range(Z):
                scale = self.arg.length_scale
                cov = self._get_cov(scale, NN, level=2)
                mean = np.zeros(NN)
                n = np.random.multivariate_normal(mean, cov, size=(N, 1))
                noise.append(n)
            noise = np.stack(noise, 1)
            assert noise.shape == (N, Z, 1, NN)
            return torch.from_numpy(noise).float().to(self.dev)

        else:
            raise ValueError(f'noise mode {mode} not supported')


    def train(self):
        self.model.train()
        Z = self.arg.model_G_args['Z']
        NN = self.arg.nnoise
        bs = self.arg.batch_size
        D = self.model[0]
        G = self.model[1]

        if self.arg.resample:
            self.data_set.resample()
            loader = torch.utils.data.DataLoader(
                dataset=self.data_set,
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True)
        else:
            loader = self.data_loader

        lossD_value = []
        lossG_value = []
        accD_real_value = []
        accD_fake_value = []
        accG_value = []

        self.io.init_timer('dataloader', 'model', 'statistics')
        ii = 0
        nn = len(loader)
        it = iter(loader)
        while True:
            if ii == nn: break
            ####################################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ####################################################################
            for p in D.parameters():
                p.requires_grad = True
            for _ in range(self.arg.repeat_D):
                if ii == nn: break
                ii += 1

                # prepare real data
                data, label = next(it)
                D.zero_grad()

                # preprocess
                N = data.size(0)
                assert N == bs
                # prepare input of G
                noise = self.gen_noise(N, NN, Z, self.arg.lambda_noise, self.arg.noise_mode)
                label_ = torch.zeros(N).random_(0, self.arg.num_class)
                self.io.check_time('dataloader')

                # train with real data
                real_data = data.float().to(self.dev) # [64, 3, 25, 64]
                label = label.long().to(self.dev)
                output_D_x = D(real_data, label)

                # train with fake data
                label_ = label_.long().to(self.dev)
                with torch.no_grad():
                    fake_data = G(noise, label_)

                output_D_z1 = D(fake_data, label_)
                lossD_real, lossD_fake = loss_hinge_dis(output_D_z1, output_D_x)
                assert real_data.size(3) == fake_data.size(3)

                lossD = lossD_real + lossD_fake
                lossD.backward(retain_graph=True)

                # backward
                self.optimizerD.step()
                self.io.check_time('model')

            ####################################################################
            # (2) Update G network: maximize log(D(G(z)))
            ####################################################################
            for p in D.parameters():
                p.requires_grad = False
            G.zero_grad()
            noise = self.gen_noise(N, NN, Z, self.arg.lambda_noise, self.arg.noise_mode)
            label_ = torch.zeros(N).random_(0, self.arg.num_class)
            label_ = label_.long().to(self.dev)
            fake_data = G(noise, label_)

            output_D_z2 = D(fake_data, label_)
            lossG = loss_hinge_gen(output_D_z2)

            # backward
            lossG.backward(retain_graph=True)
            self.optimizerG.step()
            self.io.check_time('model')

            # statistics
            self.iter_info['lossD'] = lossD.item()
            self.iter_info['lossG'] = lossG.item()
            self.iter_info['accD_real'] = output_D_x.data.mean().item()
            self.iter_info['accD_fake'] = output_D_z1.data.mean().item()
            self.iter_info['accG'] = output_D_z2.data.mean().item()

            lossD_value.append(self.iter_info['lossD'])
            lossG_value.append(self.iter_info['lossG'])
            accD_real_value.append(self.iter_info['accD_real'])
            accD_fake_value.append(self.iter_info['accD_fake'])
            accG_value.append(self.iter_info['accG'])

            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.io.check_time('statistics')

        self.epoch_info['loss_D_epoch'] = np.mean(lossD_value)
        self.epoch_info['loss_G_epoch'] = np.mean(lossG_value)
        self.epoch_info['accD_real'] = np.mean(accD_real_value)
        self.epoch_info['accD_fake'] = np.mean(accD_fake_value)
        self.epoch_info['accG'] = np.mean(accG_value)
        self.show_epoch_info()
        self.io.print_timer()


    def gen_samples(self, epoch):
        if epoch > 0:
            filename = f'epoch{epoch}.gen_100_per_class.h5'
        else:
            filename = 'gen_100_per_class.h5'
        out_file = h5py.File(os.path.join(self.arg.work_dir, filename), 'w')

        self.model.eval()
        Z = self.arg.model_G_args['Z']
        NN = self.arg.nnoise
        out = []
        for i in range(self.arg.num_class):
            label = torch.zeros([100], dtype=torch.long).fill_(i)
            noise = self.gen_noise(100, NN, Z, self.arg.lambda_noise, self.arg.noise_mode)
            label = label.to(self.dev).long()
            o = self.model[1](noise, label)
            out.append(o.data.cpu().numpy())
        for class_index in range(len(out)):
            for sample_index in range(len(out[class_index])):
                out_file['A'+str(class_index+1).zfill(3)+'_'+str(sample_index)] = out[class_index][sample_index]

    def gen_samples_actor(self):
        filename = 'generation.npy'
        self.model.eval()
        Z = self.arg.model_G_args['Z']
        NN = self.arg.nnoise
        out = []
        for i in range(self.arg.num_class):
            label = torch.zeros([5], dtype=torch.long).fill_(i)
            noise = self.gen_noise(5, NN, Z, self.arg.lambda_noise, self.arg.noise_mode)
            label = label.to(self.dev).long()
            o = self.model[1](noise, label)
            out.append(o.data.cpu().numpy())
        # out = np.array(out)
        N, M, C, V, T = np.array(out).shape
        # out = np.transpose(out.reshape(N*M, C, V, T), (0, 2, 1, 3)) # [N*M, V, C, T]
        # out = torch.from_numpy(out[0:5]).to(self.dev)
        # out_xyz = self.rot2xyz(out, vertstrans=False, beta=0)
        output_all = []
        for class_index in range(len(out)):
            for sample_index in range(len(out[class_index])):
                output = np.transpose(out[class_index][sample_index], (1, 0, 2)) # [V, C, T]
                output = torch.unsqueeze(torch.from_numpy(output).to(self.dev), 0) # [1, V, C, T]
                output_xyz = self.rot2xyz(output, vertstrans=False, beta=0) # ad hoc
                output_xyz = output_xyz.cpu().numpy()
                output_all.append(output_xyz)
        output_all = np.stack(output_all).squeeze()
        output_all = output_all.reshape((N, M) + output_all.shape[1:])
        output_all = np.transpose(output_all, (1, 0, 2, 3, 4)) # [M, N, 6890, 3, T]
        np.save(os.path.join(self.arg.work_dir, filename), np.stack(output_all))

    def generate(self, label, vertstrans=False, jointstype='vertices'):
        label = label.to(self.dev).long()
        self.model.eval()
        Z = self.arg.model_G_args['Z']
        NN = self.arg.nnoise
        # out = []
        # for i in range(self.arg.num_class):
        #     label = torch.zeros([100], dtype=torch.long).fill_(i)
        #     noise = self.gen_noise(100, NN, Z, self.arg.lambda_noise, self.arg.noise_mode)
        #     label = label.to(self.dev).long()
        #     o = self.model[1](noise, label)
        #     out.append(o.data.cpu().numpy())
        noise = self.gen_noise(label.shape[0], NN, Z, self.arg.lambda_noise, self.arg.noise_mode)
        out = self.model[1](noise, label).data.cpu().numpy()
        # out = np.array(out)
        B, C, V, T = out.shape
        # out = np.transpose(out.reshape(N*M, C, V, T), (0, 2, 1, 3)) # [N*M, V, C, T]
        # out = torch.from_numpy(out[0:5]).to(self.dev)
        # out_xyz = self.rot2xyz(out, vertstrans=False, beta=0)
        # output_all = []
        # for class_index in range(len(out)):
        #     for sample_index in range(len(out[class_index])):
        #         output = np.transpose(out[class_index][sample_index], (1, 0, 2)) # [V, C, T]
        #         output = torch.unsqueeze(torch.from_numpy(output).to(self.dev), 0) # [1, V, C, T]
        #         output_xyz = self.rot2xyz(output, vertstrans=False, beta=0) # ad hoc
        #         output_xyz = output_xyz.cpu().numpy()
        #         output_all.append(output_xyz)
        out = np.transpose(out, (0, 2, 1, 3)) # [B, C, V, T] -> [B, V, C, T]
        out = torch.from_numpy(out).to(self.dev)
        output_xyz = self.rot2xyz(out, vertstrans=vertstrans, jointstype=jointstype)
        lengths = (torch.ones(B) * T).long().to(self.dev)
        return {
            'output': out,
            'output_xyz': output_xyz,
            'lengths': lengths,
            'y': label,
        }

    def start(self):
        if self.arg.phase == 'train':
            self.writer = SummaryWriter(os.path.join(self.arg.work_dir, 'exp'))
            self.io.print_log(f'Parameters:\n{str(vars(self.arg))}\n')
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.io.print_log(f'Discriminator #Params: {count_parameters(self.model[0])}')
            self.io.print_log(f'Generator #Params: {count_parameters(self.model[1])}')

            Z = self.arg.model_G_args['Z']
            NN = self.arg.nnoise

            # Iterate over epochs
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                # Training
                self.io.print_log(f'Training epoch: {epoch}')
                self.train()
                self.io.print_log(f'Done.')

                # Save model and result
                if ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch):
                    filename = f'epoch{epoch + 1}_model.pt'
                    self.io.save_model(self.model, filename)

                if ((epoch + 1) > 100) and ((epoch + 1) % 100 == 0):
                    self.io.print_log(f'Generating samples for epoch: {epoch+1}')
                    self.gen_samples(epoch+1)

        elif self.arg.phase == 'gen':
            # self.gen_samples(0)
            self.gen_samples_actor()


    @staticmethod
    def get_parser(add_help=False):
        # Parameter priority: command line > config > default
        parser = argparse.ArgumentParser( add_help=add_help, description='Train a motion generator')

        # Basis
        parser.add_argument('--work_dir', default='work_dir/generator', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')
        parser.add_argument('--phase', default='train', help='must be train or gen')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
        parser.add_argument('--dataset', type=str, default='ntu13', help='dataset name')

        # Visualize and debug
        parser.add_argument('--log_interval', type=int, default=100, help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=20, help='the interval for storing models (#iteration)')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        # parser.add_argument('--preview', type=str2bool, default=False, help='preview the dataset smpl model')

        # Feeder
        parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=6, help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for training')
        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')

        # Evaluate Feeder
        parser.add_argument('--test_feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--test_num_worker', type=int, default=6, help='the number of worker per gpu for data loader')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for testing')
        parser.add_argument('--test_batch_size', type=int, default=256, help='testing batch size')
        parser.add_argument('--niter', type=int, default=20, help='Evaluate times')

        # Condition
        parser.add_argument('--num_class', type=int, default=0)

        # Models
        parser.add_argument('--model_D', default=None, help='the discriminator will be used')
        parser.add_argument('--model_D_args', action=DictAction, default=dict(), help='the arguments of model_D')
        parser.add_argument('--model_G', default=None, help='the generator will be used')
        parser.add_argument('--model_G_args', action=DictAction, default=dict(), help='the arguments of model_G')
        parser.add_argument('--model_weights', default=None, help='the trained weights for generator')

        # GAN
        parser.add_argument('--repeat_D', type=int, default=5)

        # Train data sampling
        parser.add_argument('--resample', type=str2bool, default=False, help='square-root sampling or not')

        # Optim
        parser.add_argument('--base_lr', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--D_lr_mult', type=float, default=1, help='multiple of lr for D')
        parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for optimizer')
        parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')

        # Noise
        parser.add_argument('--lambda_noise', type=float, default=1.0, help='scale of gp noise')
        parser.add_argument('--nnoise', default=8, type=int, help='how many noise to train')
        parser.add_argument('--noise_mode', default='independent', help='independent | gp | multi_gp | gaussian')
        parser.add_argument('--length_scale', type=float, default=5)
        parser.add_argument('--n_person', type=int, default=1, help='number of independent noise to represent multiple persons')

        # parser.add_argument("--dataset", required=True, help="Dataset to load")
        parser.add_argument("--datapath", help="Path of the data")
        parser.add_argument("--num_frames", default=60, type=int, help="number of frames or -1 => whole, -2 => random between min_len and total")
        parser.add_argument("--sampling", default="conseq", choices=["conseq", "random_conseq", "random"], help="sampling choices")
        parser.add_argument("--sampling_step", default=1, type=int, help="sampling step")
        parser.add_argument("--pose_rep", default='rot6d', choices=POSE_REPS, help="xyz or rotvec etc")
        parser.add_argument("--num_person", default=1, type=int, help="number of persons")
        
        parser.add_argument("--max_len", default=-1, type=int, help="number of frames maximum per sequence or -1")
        parser.add_argument("--min_len", default=-1, type=int, help="number of frames minimum per sequence or -1")
        parser.add_argument("--num_seq_max", default=-1, type=int, help="number of sequences maximum to load or -1")

        parser.add_argument("--glob", dest='glob', action='store_true', help="if we want global rotation")
        parser.add_argument('--no-glob', dest='glob', action='store_false', help="if we don't want global rotation")
        parser.set_defaults(glob=True)
        parser.add_argument("--glob_rot", type=int, nargs="+", default=[3.141592653589793, 0, 0],
                        help="Default rotation, usefull if glob is False")
        parser.add_argument("--translation", dest='translation', action='store_true',
                        help="if we want to output translation")
        parser.add_argument('--no-translation', dest='translation', action='store_false',
                        help="if we don't want to output translation")
        parser.set_defaults(translation=True)

        parser.add_argument("--debug", dest='debug', action='store_true', help="if we are in debug mode")
        parser.set_defaults(debug=False)

        return parser
