# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib

try:
    import pyspng
except ImportError:
    pyspng = None

from training.volumetric_rendering.utils import calc_pose

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._raw_labels_face = None
        self._raw_labels_hand = None
        self._label_shape = None
        self._hand_visibility = []
        self._face_visibility = []

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        self._raw_idx = self._balance_samples()

        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        assert xflip==False, "xflip not implemented"
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            # assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
        return self._raw_labels

    def _get_raw_labels_face(self):
        if self._raw_labels_face is None:
            self._raw_labels_face = self._load_raw_labels(is_face=True) if self._use_labels else None
            if self._raw_labels_face is None:
                self._raw_labels_face = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels_face, np.ndarray)
            # assert self._raw_labels_face.shape[0] == self._raw_shape[0]
            assert self._raw_labels_face.dtype in [np.float32, np.int64]
            if self._raw_labels_face.dtype == np.int64:
                assert self._raw_labels_face.ndim == 1
                assert np.all(self._raw_labels_face >= 0)
            self._raw_labels_face_std = self._raw_labels_face.std(0)
        return self._raw_labels_face
    
    def _get_raw_labels_hand(self):
        if self._raw_labels_hand is None:
            self._raw_labels_hand = self._load_raw_labels(is_hand=True) if self._use_labels else None
            if self._raw_labels_hand is None:
                self._raw_labels_hand = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels_hand, np.ndarray)
            # assert self._raw_labels_hand.shape[0] == self._raw_shape[0]
            assert self._raw_labels_hand.dtype in [np.float32, np.int64]
            if self._raw_labels_hand.dtype == np.int64:
                assert self._raw_labels_hand.ndim == 1
                assert np.all(self._raw_labels_hand >= 0)
            self._raw_labels_hand_std = self._raw_labels_hand.std(0)
        return self._raw_labels_hand

    def _balance_samples(self): # to be overridden by subclass
        return self._raw_idx

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_label_face(self, idx):
        label_face = self._get_raw_labels_face()[self._raw_idx[idx]]
        if label_face.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label_face] = 1
            label_face = onehot
        return label_face.copy()

    def get_label_hand(self, idx):
        label_hand = self._get_raw_labels_hand()[self._raw_idx[idx]]
        if label_hand.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label_hand] = 1
            label_hand = onehot
        return label_hand.copy()   

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        # assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1:]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}  
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION and not any(substr in fname for substr in ['albedo', 'normal', 'id']))
        self._image_fnames_face = []
        self._image_fnames_hand = []
        if len(self._image_fnames) == 0:
            with open(f"{path}/dataset.json", "r") as f:
                self.meta_data = json.load(f)['labels']

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self.meta_data)] + [512, 512, 3] #list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution[0] or raw_shape[3] != resolution[1]):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname, is_face=False, is_hand=False):
        if is_face:
            fname = os.path.join(self._path.replace('pixie', 'face-pixie'), fname)
        elif is_hand:
            fname = os.path.join(self._path.replace('pixie', 'hand-pixie'), fname)
        else:
            fname = os.path.join(self._path, fname)
        if self._type == 'dir':
            return open(fname, 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx, is_face=False, is_hand=False, resize=None):
        if is_face:
            fname = self._image_fnames_face[raw_idx]
        elif is_hand:
            fname = self._image_fnames_hand[raw_idx]
        else:
            fname = self._image_fnames[raw_idx]
        with self._open_file(fname, is_face=is_face, is_hand=is_hand) as f:
            image = np.array(PIL.Image.open(f))
            image = np.array(image)
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                if (is_face or is_hand) and resize:
                    image = PIL.Image.open(f)
                    image = image.resize((resize, resize), resample=PIL.Image.ANTIALIAS)
                    image = np.array(image)
                else:
                    image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image
    
    def _load_raw_labels(self, is_face=False, is_hand=False):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname, is_face=is_face, is_hand=is_hand) as f:
            _labels = json.load(f)['labels']
        if _labels is None:
            return None
        labels = dict(_labels)
        labels_list = []
        count = 0
        _gen = torch.Generator()
        _gen.manual_seed(0)
        for key in dict(self.meta_data):
            # key = fname.split('/')[-1]
            if key in labels:
                labels_list.append(labels[key])
                if is_hand:
                    self._hand_visibility.append(True)
                    self._image_fnames_hand.append(fname)
                if is_face:
                    self._face_visibility.append(True)
                    self._image_fnames_face.append(fname)
            else:
                rand_idx = torch.randint(0, len(_labels), [1], generator=_gen).item()
                labels_list.append(_labels[rand_idx][1])
                if is_hand:
                    self._hand_visibility.append(False)
                    self._image_fnames_hand.append(os.path.join('images', _labels[rand_idx][0]))
                if is_face:
                    self._face_visibility.append(False)
                    self._image_fnames_face.append(os.path.join('images', _labels[rand_idx][0]))
                count += 1
        print_summary(count, is_face, is_hand)
        labels = np.array(labels_list)
        labels[:, [0, 2, 4, 5]] /= 512
        labels[:, 9: 25] = np.linalg.inv(labels[:, 9: 25].reshape(-1, 4, 4)).reshape(-1, 16)
        intrinsics = labels[:, :9]
        extrinsics = labels[:, 9: 25]
        labels = np.concatenate([extrinsics, intrinsics, labels[:, 25:]], axis=1)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

def print_summary(count, is_face, is_hand):
    if is_face:
        print(f'{count} faces resampled')
    elif is_hand:
        print(f'{count} hands resampled')
    else:
        print(f'{count} bodys resampled')

#----------------------------------------------------------------------------

# USED FOR FLIP SMPL POSE
SMPL_JOINTS_FLIP_PERM = np.array([2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22])-1
SMPL_POSE_FLIP_PERM = []
for i in SMPL_JOINTS_FLIP_PERM:
    SMPL_POSE_FLIP_PERM.append(3*i)
    SMPL_POSE_FLIP_PERM.append(3*i+1)
    SMPL_POSE_FLIP_PERM.append(3*i+2)

class SMPLLabeledDataset(ImageFolderDataset):
    def _balance_samples(self):
        # raw_labels = self._get_raw_labels()[self._raw_idx]
        # raw_labels_face = self._get_raw_labels_face()[self._raw_idx]
        # raw_labels_hand = self._get_raw_labels_hand()[self._raw_idx]
        return self._raw_idx

    def get_visibility(self, idx, is_hand=False):
        if is_hand:
            return self._hand_visibility[self._raw_idx[idx]]
        else:
            return self._face_visibility[self._raw_idx[idx]]
    
    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        image_face = self._load_raw_image(self._raw_idx[idx], is_face=True, resize=256)
        image_hand = self._load_raw_image(self._raw_idx[idx], is_hand=True, resize=256)
        label = self.get_label(idx)
        label_face = self.get_label_face(idx)
        label_hand = self.get_label_hand(idx)
        vis_face = self.get_visibility(idx)
        vis_hand = self.get_visibility(idx, is_hand=True)
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        return image.copy(), image_face.copy(), image_hand.copy(), label, label_face, label_hand, vis_face, vis_hand


class RebalancedSMPLLabeledDataset(ImageFolderDataset):

    def _balance_samples(self):
        raw_labels = self._get_raw_labels()[self._raw_idx]
        raw_labels_face = self._get_raw_labels_face()[self._raw_idx]
        raw_labels_hand = self._get_raw_labels_hand()[self._raw_idx]
        poses = [calc_pose(param[:16])[1] for param in raw_labels]
        poses = np.array(poses)
        # repeat large pose, calculated from dataset distribution.
        large_pose = (poses[:,0]<=-25) | (poses[:,0]>=35) # | ((180 - poses[:, 1])<=20)
        if not np.all(large_pose):
            print(f"Rebalancing large pose samples by repeating: ratio = {large_pose.sum()}/{len(self._raw_idx)}")
        new_idx = np.append(self._raw_idx, np.tile(self._raw_idx[large_pose], 1))

        return new_idx

    def get_visibility(self, idx, is_hand=False):
        if is_hand:
            return self._hand_visibility[self._raw_idx[idx]]
        else:
            return self._face_visibility[self._raw_idx[idx]]
    
    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        image_face = self._load_raw_image(self._raw_idx[idx], is_face=True, resize=256)
        image_hand = self._load_raw_image(self._raw_idx[idx], is_hand=True, resize=256)
        label = self.get_label(idx)
        label_face = self.get_label_face(idx)
        label_hand = self.get_label_hand(idx)
        vis_face = self.get_visibility(idx)
        vis_hand = self.get_visibility(idx, is_hand=True)
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        return image.copy(), image_face.copy(), image_hand.copy(), label, label_face, label_hand, vis_face, vis_hand