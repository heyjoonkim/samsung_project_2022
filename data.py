from torch.utils.data import Dataset
import torch
import os
import zipfile
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class SEMDataset(Dataset):
    def __init__(self, path, mode):
        self._mode = mode
        self._path = os.path.join(path, mode)

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._sem_dir = os.path.join(self._path, "SEM")
            assert os.path.isdir(self._sem_dir)
            self._sem_fnames = os.listdir(self._sem_dir)

            # filenames = os.listdir(self._sem_dir)
            # self._sem_fnames = [os.path.join(self._sem_dir, f) for f in filenames]
            # print(self._sem_fnames[0][:-9].split("/")[-1])

            if mode != "Test":
                self._depth_dir = os.path.join(self._path, "Depth")
                assert os.path.isdir(self._depth_dir)

                # filenames = os.listdir(self._depth_dir)
                # self._dpt_fnames = [os.path.join(self._depth_dir, f) for f in filenames]
        else:
            raise IOError(f"{self._path} should be dir")

        self._transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self._sem_fnames)

    def __getitem__(self, idx):
        sem_path = os.path.join(self._sem_dir, self._sem_fnames[idx])
        sem_img = Image.open(sem_path)
        if self._mode != "Test":
            dpt_name = self._sem_fnames[idx][:-9]
            dpt_path = os.path.join(self._depth_dir, f"{dpt_name}.png")
            dpt_img = Image.open(dpt_path)
            return self._transform(sem_img), self._transform(dpt_img)
        else:
            return self._transform(sem_img), 0


def unzip_dir(zip_path):
    '''
    :param zip_path: zip_path.zip
    :return:
    '''

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(zip_path[:-4])


if __name__=="__main__":
    zip_path = f"./dataset/AI_challenge_data.zip"
    
    if not os.path.exists(zip_path):
        raise IOError(f"{zip_path} does not exists")

    if not os.path.isdir(zip_path[:-4]):
        unzip_dir(zip_path)
        data_dir = zip_path[:-4]
    else:
        data_dir = zip_path[:-4]
        print("dataset ready")

    train_dataset = SEMDataset(path=data_dir, mode="Train")
    sem, dpt = train_dataset[0]
    print(sem.shape)
    print(dpt.shape)

    # val_dataset = SEMDataset(path=data_dir, mode="Validation")
    # test_dataset = SEMDataset(path=data_dir, mode="Test")
    





    




