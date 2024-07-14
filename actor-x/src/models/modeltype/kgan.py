import argparse
import yaml
import os
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from tensorboardX import SummaryWriter

# torchlight
from src.utils import torchlight
from src.utils.torchlight import import_class, DictAction, str2bool

from ..tools.losses import get_loss_function
from ..rotation2xyz import Rotation2xyz

cuda = True if torch.cuda.is_available() else False
Tensor     = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

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


class KGAN(nn.Module):
    def __init__(self, device, lambdas, latent_dim, outputxyz,
                 pose_rep, glob, glob_rot, translation, jointstype, vertstrans, **kwargs):
        super().__init__()
        self.load_arg(kwargs)
        self.init_environment()
        self.load_model()
        self.load_optimizer()

        # self.encoder = encoder
        # self.decoder = decoder

        self.outputxyz = outputxyz
        
        self.lambdas = lambdas
        
        self.latent_dim = latent_dim
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.device = device
        self.translation = translation
        self.jointstype = jointstype
        self.vertstrans = vertstrans
        self.kwargs = kwargs
        self.idx = 0
        
        self.losses = list(self.lambdas) + ["mixed"]

        self.rotation2xyz = Rotation2xyz(device=self.device)

        self.param2xyz = {"pose_rep": self.pose_rep,
                          "glob_rot": self.glob_rot,
                          "glob": self.glob,
                          "jointstype": self.jointstype,
                          "translation": self.translation,
                          "vertstrans": self.vertstrans,
                          "num_person": kwargs.get("num_person", 1), 
                          "fixrot": kwargs.get("fixrot", False)}
        
    def rot2xyz(self, x, mask, **kwargs):
        kargs = self.param2xyz.copy()
        kargs.update(kwargs)
        return self.rotation2xyz(x, mask, **kargs)

    def load_arg(self, kwargs=None):
        if kwargs['config'] is not None:
            with open(kwargs['config'], 'r') as f:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
        self.arg = default_arg

    def init_environment(self):
        os.makedirs(self.arg['work_dir'], exist_ok=True)
        self.io = torchlight.IO(self.arg['work_dir'], save_log=self.arg['save_log'], print_log=self.arg['print_log'])
        # self.io.save_arg(self.arg)
        self.dev = "cuda:0"
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)


    def load_model(self):
        model_D = self.io.load_model(self.arg['model_D'], **(self.arg['model_D_args']))
        model_G = self.io.load_model(self.arg['model_G'], **(self.arg['model_G_args']))
        if self.arg['model_weights'] is not None:
            self.model = self.io.load_weights(nn.ModuleList([model_D, model_G]), self.arg['model_weights']).to(self.dev)
        else:
            self.model = nn.ModuleList([model_D, model_G]).to(self.dev)
            self.model.apply(weights_init)

        self.global_step = 0

    def load_optimizer(self):
        self.optimizerD = optim.Adam(
            self.model[0].parameters(),
            lr=self.arg['base_lr'] * self.arg['D_lr_mult'],
            betas=(self.arg['beta1'], 0.999),
            weight_decay=self.arg['weight_decay'])

        self.optimizerG = optim.Adam(
            self.model[1].parameters(),
            lr=self.arg['base_lr'],
            betas=(self.arg['beta1'], 0.999),
            weight_decay=self.arg['weight_decay'])


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

    def generate(self, label, durations, vertstrans=False, jointstype='vertices', nspa=1,
                 noise_same_action="random", noise_diff_action="random",
                 fact=1, fixrot=False):
        label = label.to(self.dev).long()
        self.model.eval()
        Z = self.arg['model_G_args']['Z']
        NN = self.arg['nnoise']
        # out = []
        # for i in range(self.arg.num_class):
        #     label = torch.zeros([100], dtype=torch.long).fill_(i)
        #     noise = self.gen_noise(100, NN, Z, self.arg.lambda_noise, self.arg.noise_mode)
        #     label = label.to(self.dev).long()
        #     o = self.model[1](noise, label)
        #     out.append(o.data.cpu().numpy())
        noise = self.gen_noise(label.shape[0], NN, Z, self.arg['lambda_noise'], self.arg['noise_mode'])
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
        mask = torch.ones((B, T), dtype=bool).to(self.dev)
        output_xyz = self.rot2xyz(out, mask, vertstrans=vertstrans, jointstype=jointstype)
        lengths = (torch.ones(B) * T).long().to(self.dev)
        return {
            'output': out,
            'output_xyz': output_xyz,
            'lengths': lengths,
            'y': label,
        }


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


    def sample_action(self, n_row, batches_done):
        z = Variable(Tensor(np.random.normal(0, 1, (10*n_row, opt.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(10) for num in range(n_row)])
        labels = Variable(LongTensor(labels))
        gen_imgs = generator(z, labels)
        with open(os.path.join(actions_out, str(batches_done)+'.npy'), 'wb') as npf:
            np.save(npf, gen_imgs.data.cpu())


    def compute_gradient_penalty(self, D, real_samples, fake_samples, labels):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        labels = LongTensor(labels)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates, labels)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    def forward(self, batch):

        discriminator = self.model[0]
        generator = self.model[1]

        # Configure input
        imgs, labels = batch['x'].permute(0, 2, 3, 1), batch['y']
        bs = imgs.shape[0]
        real_imgs = Variable(imgs.type(Tensor))
        labels    = Variable(labels.type(LongTensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizerD.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (bs, self.arg['latent_dim']))))

        # Generate a batch of actions
        fake_imgs = generator(z, labels)

        # Real actions
        real_validity = discriminator(real_imgs, labels)
        # Fake actions
        fake_validity = discriminator(fake_imgs, labels)
        # Gradient penalty
        gradient_penalty = self.compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, labels.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.arg['lambda_gp'] * gradient_penalty

        d_loss.backward()
        self.optimizerD.step()

        self.optimizerG.zero_grad()

        # Train the generator after n_critic discriminator steps
        if self.idx % self.arg['n_critic'] == 0:
            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of actions
            fake_imgs = generator(z, labels)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake actions
            fake_validity = discriminator(fake_imgs, labels)
            g_loss = -torch.mean(fake_validity)
            self.iter_info['lossG'] = g_loss.item()

            g_loss.backward()
            self.optimizerG.step()
        self.idx += 1

        # print(
        #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        #     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        # )

        self.iter_info['lossD'] = d_loss.item()

        return self.iter_info

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

