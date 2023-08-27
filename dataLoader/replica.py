import json
import os
import cv2
import h5py
import pickle
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *
from .render_util import *
# from utils.ray_utils import *
# from utils.render_util import *
from utils.mani.colormap_utils import *


class ReplicaDataset(Dataset):
    def __init__(self, datadir, split, near=0.05, far=47, scene_bbox_stretch=5.5,
                 downsample=1.0, is_stack=False,
                 dino_feature_dir=None, label_interval=1):
        self.split = split
        self.root_dir = datadir
        self.dino_feature_dir = dino_feature_dir

        self.is_stack = is_stack

        img_w, img_h = 640, 480
        self.img_wh = (int(img_w / downsample), int(img_h / downsample))
        self.define_transforms()  # tensor transforms

        self.img_total_num = len(glob(os.path.join(self.root_dir, "rgb", "rgb_*.png")))

        # replica near_far
        self.near_far = [near, far]  # used in sample_ray(tensorBase.py) for clipping samples, near must be 0.1 ?
        self.scene_bbox_stretch = scene_bbox_stretch

        self.label_interval = label_interval
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.downsample = downsample

    def read_meta(self):
        w, h = self.img_wh

        hfov = 90
        self.focal_x = 0.5 * w / np.tan(0.5 * np.radians(hfov))  # w ?
        self.focal_y = self.focal_x
        cx = (w - 1.) / 2
        cy = (h - 1.) / 2

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal_x, self.focal_y])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal_x, 0, cx], [0, self.focal_y, cy], [0, 0, 1]]).float().cpu()

        # load c2w for all images in the video
        traj_file = os.path.join(self.root_dir, "traj_w_c.txt")
        self.Ts_full = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)

        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_ins_imgs = []  # original images
        self.all_ins_labels = []  # continuous labels
        self.label2ins_map = {}
        self.all_colored_sems = []
        self.all_masks = []
        self.all_depth = []
        self.downsample = 1.0

        img_eval_interval = 5
        if self.split == "train":
            self.indices = list(range(0, self.img_total_num, img_eval_interval))
        elif self.split == "test":
            self.indices = list(range(img_eval_interval // 2, self.img_total_num, img_eval_interval))

        for i in tqdm(self.indices, desc=f'Loading data {self.split} ({len(self.indices)})'):  # img_list:#
            c2w = torch.FloatTensor(self.Ts_full[i])
            self.poses += [c2w]

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

            image_path = os.path.join(self.root_dir, "rgb", f"rgb_{i}.png")
            img = Image.open(image_path)
            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w), normalized
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3)
            self.all_rgbs += [img]

            ins_img_path = os.path.join(self.root_dir, 'semantic_instance', f"semantic_instance_{i}.png")
            ins_img = np.array(Image.open(ins_img_path))
            if self.img_wh is not None:
                ins_img = cv2.resize(ins_img, self.img_wh, interpolation=cv2.INTER_NEAREST)  # (h, w)

            self.all_ins_imgs.append(np.array(ins_img))  # ndarray ins_img

        self.poses = torch.stack(self.poses)

        all_rays_o = torch.stack(self.all_rays)[..., :3]  # for all images, (N_imgs, h*w, 3)
        all_rays_o = all_rays_o.reshape(-1, 3)

        scene_min = torch.min(all_rays_o, 0)[0] - self.scene_bbox_stretch
        scene_max = torch.max(all_rays_o, 0)[0] + self.scene_bbox_stretch
        self.scene_bbox = torch.stack([scene_min, scene_max]).reshape(-1, 3)

        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

        if not self.is_stack:
            # pixel-wise in training
            self.all_rays = torch.cat(self.all_rays, 0)  # (N_imgs*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (N_imgs*h*w, 3)
        else:
            # image-wise in testing
            self.all_rays = torch.stack(self.all_rays, 0)  # (N_imgs,h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1], 3)  # (N_imgs,h,w,3)

        ''''novel view-points'''
        center = torch.mean(self.scene_bbox, dim=0)
        radius = torch.norm(self.scene_bbox[1]-center)*0.1
        up = torch.mean(self.poses[:, :3, 1], dim=0).tolist()
        pos_gen = circle(radius=radius, h=-0.2*up[1], axis='y')
        self.render_path = gen_path(pos_gen, up=up, frames=200).float().cpu()
        self.render_path[:, :3, 3] += center

    def remap_ins_gt_label(self, train_ins_imgs=None, test_ins_imgs=None,
                           load_map=False, ins2label_map=None):
        self.all_ins_imgs = np.asarray(self.all_ins_imgs)
        self.all_ins_remaps = self.all_ins_imgs.copy()

        if load_map:
            assert ins2label_map, "map file path must be provided"

            scene_id = self.root_dir.split('/')[-1]  # todo: change?
            ins2label = json.load((open(ins2label_map, 'r')))['replica'][scene_id]
            for ins, label in ins2label.items():
                self.all_ins_remaps[self.all_ins_imgs == int(ins)] = label

            self.num_ins_class = len(ins2label.keys())
        else:
            self.ins_classes = np.unique(np.concatenate((np.unique(train_ins_imgs), np.unique(test_ins_imgs))).astype(np.uint8))
            self.num_ins_class = self.ins_classes.shape[0]

            for i in range(self.num_ins_class):
                self.all_ins_remaps[self.all_ins_imgs == self.ins_classes[i]] = i

    def select_sems(self):
        assert self.is_stack
        self.selected_rays = self.all_rays[::self.label_interval, ...]
        self.selected_rgbs = self.all_rgbs[::self.label_interval, ...]
        self.selected_sems = torch.tensor(self.all_ins_remaps[::self.label_interval, ...]).unsqueeze(-1)

    def set_label2color_map(self, label2color_path=None):
        if label2color_path:  # label to assigned color
            color_f = os.path.join(label2color_path)
        else:
            color_f = os.path.join(self.root_dir, 'ins_rgb.hdf5')

        with h5py.File(color_f, 'r') as f:
            ins_rgbs = f['datasets'][:]  # ndarray
        f.close()

        def label2color_map(ins_label):
            color_map = np.zeros(shape=(int(self.img_wh[0] * self.img_wh[1]), 3))
            for label in np.unique(ins_label):
                valid_label_list = list(range(0, ins_rgbs.shape[0]))
                if label in valid_label_list:
                    color_map[ins_label == label] = ins_rgbs[label]
            return color_map

        self.label2color_map = label2color_map

    def plot_label2color_map(self, label2color_path=None):
        if label2color_path:  # label to assigned color
            color_f = os.path.join(label2color_path)
        else:
            color_f = os.path.join(self.root_dir, 'ins_rgb.hdf5')
        assert os.path.exists(color_f), 'label_colormap must be provided'

        save_path = os.path.join(os.path.dirname(color_f), 'vis_ins_rgb.png')

        if os.path.exists(save_path):
            return

        with h5py.File(color_f, 'r') as f:
            ins_rgbs = f['datasets'][:]  # ndarray
        f.close()

        colors = []
        # rgb: 0-255
        for rgb in ins_rgbs:
            colors.append(tuple(rgb / 255.))

        plot_colormap(colors, save_path)

    def load_dino_features(self, normalized=True, resize=True):
        scene_id = self.dino_feature_dir.split('/')[-4]
        '''multi layers'''
        self.N_dino_blocks = len(glob(os.path.join(self.dino_feature_dir, '*.pth')))
        '''single layer'''
        # self.N_dino_blocks = 1

        for block_idx in range(self.N_dino_blocks):
            '''multi layers'''
            features = torch.load(os.path.join(self.dino_feature_dir, f'{scene_id}_layer_{block_idx}.pth'))
            '''single layer'''
            # features = torch.load(os.path.join(self.dino_feature_dir, f'{scene_id}_layer_{3-block_idx}.pth'))

            n_channels, feamap_h, feamap_w,  = features[list(features.keys())[0]].shape

            # reshape dino features
            if resize:
                for k in features:
                    features[k] = torch.nn.functional.interpolate(features[k][None], size=(self.img_wh[1], self.img_wh[0]), mode="nearest")[0]
                n_channels, feamap_h, feamap_w = features[k].shape

            if normalized:
                for k in features:
                    features[k] = torch.nn.functional.normalize(features[k], dim=0)

            features_array = np.zeros(shape=(len(self.indices), n_channels, feamap_h, feamap_w))
            for i in tqdm(range(len(self.indices)), desc='Loading DINO features: '):
                fn_idx = self.indices[i]
                features_array[i] = features[f'rgb_{fn_idx}.png'].numpy()
            # features = np.stack([features[f'rgb_{fn}.png'].permute(1, 2, 0).numpy() for fn in self.idxs], axis=-1)  # (h, w, feature_channels, num_imgs)
            # features = np.moveaxis(features, -1, 0)  # (num_imgs, h, w, feature_channels)

            # todo: remove numpy
            assert n_channels in [3, 64]  # either rgb or pca features
            # features_array = features_array[:, None]  # (N, 1, H, W, C)
            # features_array = np.transpose(features_array, [0, 2, 3, 1, 4])  # (N, H, W, 1, C)
            features_array = np.moveaxis(features_array, 1, -1)
            features_array = np.reshape(features_array, [-1, n_channels]).astype(np.float32)  # (N*H*W, C)

            if block_idx == 0:
                self.all_semfeas = torch.tensor(features_array)
            else:
                self.all_semfeas = torch.cat((self.all_semfeas, torch.tensor(features_array)), dim=-1)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]

    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def get_ins_loss(self, ins_map, ins_train):
        ins_loss_fun = torch.nn.CrossEntropyLoss()
        ins_loss = lambda logit, label: ins_loss_fun(logit, label)

        return ins_loss(ins_map, ins_train.squeeze().long())

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        img = self.all_rgbs[idx]
        rays = self.all_rays[idx]
        ins = torch.tensor(self.all_ins_imgs[idx])

        sample = {
            'rays': rays,
            'rgbs': img,
            'ins': ins
        }

        return sample
