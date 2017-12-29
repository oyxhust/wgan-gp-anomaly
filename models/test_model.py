import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from torch.autograd import grad
import util.util as util
from .base_model import BaseModel
from . import networks
import ntpath


class TEST_Model(BaseModel):
    def name(self):
        return 'TEST_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.batchSize = opt.batchSize
        self.img_nc = opt.img_nc
        self.fineSize = opt.fineSize
        self.isTrain = opt.isTrain
        # define tensors
        self.input_img = self.Tensor(opt.batchSize, opt.img_nc,
                                   opt.fineSize, opt.fineSize)
        use_gpu = len(self.gpu_ids) > 0
        if use_gpu:
            assert(torch.cuda.is_available())

        # load/define networks
        self.netG = networks.define_G_Z(opt.batchSize, opt.fineSize*opt.fineSize, opt.ngf,
                                    opt.init_type, self.gpu_ids)
        # for name, param in self.netG.named_parameters():
        #     print(name)
        self.load_network_update_Z(self.netG, 'G', opt.which_epoch)

        # define loss functions
        self.criterionL1 = torch.nn.L1Loss()
        # initialize optimizers
        self.schedulers = []
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, 0.9))
        self.optimizers.append(self.optimizer_G)
        for optimizer in self.optimizers:
            self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_img = input['Img']
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['Img_paths']
        self.one = torch.FloatTensor([1])
        if len(self.gpu_ids) > 0:
            self.one = self.one.cuda(self.gpu_ids[0])
        self.one = Variable(self.one)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def update_G(self):
        self.netG.zero_grad()
        if len(self.gpu_ids) > 0:
            real_data = self.input_img.cuda(self.gpu_ids[0])
        self.real_data_v = Variable(real_data)
        # self.real_data_v = Variable(self.input_img)
        self.real_data_v = self.real_data_v.view(-1, self.img_nc*self.fineSize*self.fineSize)
        self.fake = self.netG(self.one)
        self.loss_G = self.criterionL1(self.fake, self.real_data_v)
        # print(self.loss_G.data[0])
        self.loss_G.backward()
        self.optimizer_G.step()


    def optimize_noise(self):
        ############################
        # Update input noise Z and fix the parameters of G network
        ###########################
        for name, param in self.netG.named_parameters():
            if name == 'noise.Z':
                param.requires_grad = True # Update input noise Z
            else:
                param.requires_grad = False # Fix the parameters of G network
        self.update_G()


    def get_current_errors(self):
        # return OrderedDict([('G_loss', self.loss_G.data[0])
        #                     # ('D_loss', self.loss_G.data[0])
        #                         ])
        return OrderedDict({'G_loss': self.loss_G.data[0], 'X':0})

    def get_current_visuals(self):
        # print(self.fake.data)
        real_data = util.tensor2im(self.real_data_v.data.view(-1, self.img_nc,
                                   self.fineSize, self.fineSize))
        fake_data = util.tensor2im(self.fake.data.view(-1, self.img_nc,
                                   self.fineSize, self.fineSize))
        return OrderedDict([('real_data', real_data), ('fake_data', fake_data)])
        # return OrderedDict([('real_data', real_data)])

    def save(self, webpage, image_path, label):
        image_dir = os.path.join(webpage.get_image_dir(), 'Testing_models')
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]
        if not os.path.isdir(image_dir):
            os.mkdir(image_dir)
        save_dir = os.path.join(image_dir, name)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        self.save_network(self.netG, 'G_Testing', label, self.gpu_ids, save_dir)
