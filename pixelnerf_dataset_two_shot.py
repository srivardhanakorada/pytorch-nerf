import numpy as np
import torch

from torch.utils.data import Dataset

class PixelNeRFDataset(Dataset):
    def __init__(self, data_dir,num_iters,test_obj_idx,test_source_pose_idx_one,test_source_pose_idx_two,test_target_pose_idx):
        self.data_dir = data_dir
        self.N = num_iters
        with open(f"{data_dir}/objs.txt") as f:
            self.objs = f.read().split("\n")[:-1]

        self.test_obj_idx = test_obj_idx
        self.test_source_pose_idx_one = test_source_pose_idx_one
        self.test_source_pose_idx_two = test_source_pose_idx_two
        self.test_target_pose_idx = test_target_pose_idx
        data = np.load(f"{data_dir}/poses.npz")
        self.poses = poses = data["poses"]
        (n_objs, n_poses) = poses.shape[:2]
        self.z_len = len(str(n_poses - 1))
        self.poses = torch.Tensor(poses)

        self.channel_means = torch.Tensor([0.485, 0.456, 0.406])
        self.channel_stds = torch.Tensor([0.229, 0.224, 0.225])

        samp_img = np.load(f"{data_dir}/{self.objs[0]}/{str(0).zfill(self.z_len)}.npy")
        img_size = samp_img.shape[0]
        self.pix_idxs = np.arange(img_size ** 2)
        xs = torch.arange(img_size) - (img_size / 2 - 0.5)
        ys = torch.arange(img_size) - (img_size / 2 - 0.5)
        (xs, ys) = torch.meshgrid(xs, -ys, indexing="xy")
        focal = float(data["focal"])
        pixel_coords = torch.stack([xs, ys, torch.full_like(xs, -focal)], dim=-1)
        camera_coords = pixel_coords / focal
        self.init_ds = camera_coords
        self.camera_distance = camera_distance = float(data["camera_distance"])
        self.init_o = torch.Tensor(np.array([0, 0, camera_distance]))
        self.scale = (img_size / 2) / focal

    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        obj_idx = np.random.randint(self.poses.shape[0])
        obj = self.objs[obj_idx]
        obj_dir = f"{self.data_dir}/{obj}"

        source_pose_idx_one = np.random.randint(self.poses.shape[1])
        source_pose_idx_two = np.random.randint(self.poses.shape[1])
        if obj_idx == self.test_obj_idx:
            while source_pose_idx_one == self.test_source_pose_idx_one:
                source_pose_idx_one = np.random.randint(self.poses.shape[1])
            while source_pose_idx_two == self.test_source_pose_idx_two:
                source_pose_idx_two = np.random.randint(self.poses.shape[1])

        source_img_f_one = f"{obj_dir}/{str(source_pose_idx_one).zfill(self.z_len)}.npy"
        source_img_f_two = f"{obj_dir}/{str(source_pose_idx_two).zfill(self.z_len)}.npy"
        source_image_one = torch.Tensor(np.load(source_img_f_one)/255)
        source_image_two = torch.Tensor(np.load(source_img_f_two)/255)
        source_image_one = (source_image_one - self.channel_means) / self.channel_stds
        source_image_two = (source_image_two - self.channel_means) / self.channel_stds
        source_pose_one = self.poses[obj_idx, source_pose_idx_one]
        source_pose_two = self.poses[obj_idx, source_pose_idx_two]
        source_R_one = source_pose_one[:3, :3]
        source_R_two = source_pose_two[:3, :3]

        target_pose_idx = np.random.randint(self.poses.shape[1])
        if obj_idx == self.test_obj_idx:
            while (target_pose_idx == self.test_source_pose_idx_one) or (
                target_pose_idx == self.test_target_pose_idx
            ):
                target_pose_idx = np.random.randint(self.poses.shape[1])
            while (target_pose_idx == self.test_source_pose_idx_two) or (
                target_pose_idx == self.test_target_pose_idx
            ):
                target_pose_idx = np.random.randint(self.poses.shape[1])
        
        target_img_f = f"{obj_dir}/{str(target_pose_idx).zfill(self.z_len)}.npy"
        target_image = np.load(target_img_f)
        not_gray_pix = np.argwhere((target_image == 128).sum(-1) != 3)
        top_row = not_gray_pix[:, 0].min()
        bottom_row = not_gray_pix[:, 0].max()
        left_col = not_gray_pix[:, 1].min()
        right_col = not_gray_pix[:, 1].max()
        bbox = (top_row, left_col, bottom_row, right_col)

        target_image = np.load(target_img_f) / 255
        target_pose = self.poses[obj_idx, target_pose_idx]
        target_R = target_pose[:3, :3]

        R_one = source_R_one.T @ target_R
        R_two = source_R_two.T @ target_R

        return (source_image_one, source_image_two,torch.Tensor(R_one), torch.Tensor(R_two), torch.Tensor(target_image), bbox)