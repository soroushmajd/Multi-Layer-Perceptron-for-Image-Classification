import os
import h5py
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np


class SignDigitDataset(Dataset):

    def __init__(self, root_dir='data/', h5_name='train_signs.h5',
                 train=True, transform=None):
        self.transform = transform
        self.train = train
        self.h5_path = os.path.join(root_dir, h5_name)
        key = 'train_set' if self.train else 'test_set'
        self.dataset_images = np.array(self._read_data(self.h5_path)[key + '_x'])
        self.dataset_labels = np.array(self._read_data(self.h5_path)[key + '_y'])

    def __len__(self):
        return len(self.dataset_images)

    def __getitem__(self, index):
        # loading an image and its label
        img = self.dataset_images[index]
        label = self.dataset_labels[index]

        # preprocessing
        img = self.transform(img)
        # label = get_one_hot(label, 6) #CE loss function will implicitly does this encoding

        return {
            'image': img,
            'label': label
        }

    def _read_data(self, h5_path):
        dataset = h5py.File(h5_path, "r")
        return dataset


