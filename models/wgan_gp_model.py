import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from torch.autograd import grad
import util.util as util
from .base_model import BaseModel
from . import networks


class WGAN_GP_Model(BaseModel):
    def name(self):
        return 'WGAN_GP_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1
        self.batchSize = opt.batchSize
        self.img_nc = opt.img_nc
        self.fineSize = opt.fineSize
        self.LAMBDA = opt.LAMBDA
        self.isTrain = opt.isTrain
        # define tensors
        self.input_img = self.Tensor(opt.batchSize, opt.img_nc,
                                   opt.fineSize, opt.fineSize)
        use_gpu = len(self.gpu_ids) > 0
        if use_gpu:
            assert(torch.cuda.is_available())
        if len(self.gpu_ids) > 0:
            self.one = self.one.cuda(self.gpu_ids[0])
            self.mone = self.mone.cuda(self.gpu_ids[0])

        # load/define networks
        self.netG = networks.define_G(opt.fineSize*opt.fineSize, opt.ngf,
                                    opt.init_type, self.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt.ndf, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.9))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.9))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(self.batchSize, 1)
        alpha = alpha.expand(real_data.size())
        if len(self.gpu_ids) > 0:
            alpha = alpha.cuda(self.gpu_ids[0])

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if len(self.gpu_ids) > 0:
            interpolates = interpolates.cuda(self.gpu_ids[0])
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(self.gpu_ids[0]) if len(self.gpu_ids) > 0 else torch.ones(
                                      disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def set_input(self, input):
        input_img = input['Img']
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['Img_paths']

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def update_D(self):
        if len(self.gpu_ids) > 0:
            real_data = self.input_img.cuda(self.gpu_ids[0])
        self.real_data_v = Variable(real_data)
        # self.real_data_v = Variable(self.input_img)
        self.real_data_v = self.real_data_v.view(-1, self.img_nc*self.fineSize*self.fineSize)

        self.netD.zero_grad()

        # train with real
        D_real = self.netD(self.real_data_v)
        D_real = D_real.mean()
        # print D_real
        D_real.backward(self.mone)

        # train with fake
        noise = torch.randn(self.batchSize, 128)
        if len(self.gpu_ids) > 0:
            noise = noise.cuda(self.gpu_ids[0])
        noisev = Variable(noise, volatile=True)  # totally freeze netG
        self.fake = Variable(self.netG(noisev).data)
        # print(self.fake)
        inputv = self.fake
        D_fake = self.netD(inputv)
        D_fake = D_fake.mean()
        D_fake.backward(self.one)

        # train with gradient penalty
        gradient_penalty = self.calc_gradient_penalty(self.netD, self.real_data_v.data, self.fake.data)
        gradient_penalty.backward()

        self.D_cost = D_fake - D_real + gradient_penalty
        self.Wasserstein_D = D_real - D_fake
        self.optimizer_D.step()

    def update_G(self):
        self.netG.zero_grad()

        noise = torch.randn(self.batchSize, 128)
        if len(self.gpu_ids) > 0:
            noise = noise.cuda(self.gpu_ids[0])
        noisev = Variable(noise)
        fake = self.netG(noisev)
        G = self.netD(fake)
        G = G.mean()
        G.backward(self.mone)
        self.G_cost = -G
        self.optimizer_G.step()


    def optimize_parameters(self, only_update_D):
        self.only_update_D = only_update_D
        ############################
        # (1) Update D network
        ###########################
        for p in self.netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        self.update_D()

        if not only_update_D:
            ############################
            # (2) Update G network
            ###########################
            for p in self.netD.parameters():
                p.requires_grad = False  # to avoid computation
            self.update_G()


    def get_current_errors(self):
        return OrderedDict([('G_loss', self.G_cost.data[0]),
                                ('D_loss', self.D_cost.data[0]),
                                ('Wasserstein_D', self.Wasserstein_D.data[0])
                                ])

    def get_current_visuals(self):
        # print(self.fake.data)
        real_data = util.tensor2im(self.real_data_v.data.view(-1, self.img_nc,
                                   self.fineSize, self.fineSize))
        fake_data = util.tensor2im(self.fake.data.view(-1, self.img_nc,
                                   self.fineSize, self.fineSize))
        return OrderedDict([('real_data', real_data), ('fake_data', fake_data)])
        # return OrderedDict([('real_data', real_data)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
