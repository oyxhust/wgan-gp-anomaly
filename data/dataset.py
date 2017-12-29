import os.path
import random
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from data.image_folder import make_dataset
from PIL import Image


class Dataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_img = os.path.join(opt.dataroot, opt.phase)

        self.img_paths = sorted(make_dataset(self.dir_img))

        # assert(opt.resize_or_crop == 'resize_and_crop')
        print(opt.resize_or_crop)
        print('image size: {}'.format(opt.fineSize))

        # transform_list = [transforms.ToTensor(),
        #                   transforms.Normalize((0.5, 0.5, 0.5),
        #                                        (0.5, 0.5, 0.5))]
        transform_list = [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)
        img = self.transform(img)

        # w_total = AB.size(2)
        # w = int(w_total / 2)
        # h = AB.size(1)
        # w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        # h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        # A = AB[:, h_offset:h_offset + self.opt.fineSize,
        #        w_offset:w_offset + self.opt.fineSize]
        # B = AB[:, h_offset:h_offset + self.opt.fineSize,
        #        w + w_offset:w + w_offset + self.opt.fineSize]

        # if self.opt.which_direction == 'BtoA':
        #     input_nc = self.opt.output_nc
        #     output_nc = self.opt.input_nc
        # else:
        #     input_nc = self.opt.input_nc
        #     output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(img.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            img = img.index_select(2, idx)

        if self.opt.img_nc == 1:  # RGB to gray
            tmp = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = tmp.unsqueeze(0)

        # if output_nc == 1:  # RGB to gray
        #     tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
        #     B = tmp.unsqueeze(0)
    
        return {'Img': img, 'Img_paths': img_path}

    def __len__(self):
        return len(self.img_paths)

    def name(self):
        return 'NormalDataset'
