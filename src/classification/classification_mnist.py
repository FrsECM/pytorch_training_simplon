from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from .classificiation_image import ClassificationImage

class ClassificationMNIST(Dataset):
    def __init__(self,rootdir:str,progress:bool=True,nmax_per_class:int=None):
        assert os.path.exists(rootdir),'Root dir should be an existing directory.'
        self.train_transforms=A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GridDistortion(),
            A.ToFloat(max_value=255),
            A.Normalize(mean=0.5,std=0.5),
            ToTensorV2()
        ])
        self.test_transform=A.Compose([
            A.ToFloat(max_value=255),
            A.Normalize(mean=0.5,std=0.5),
            ToTensorV2()
        ])

        self.splits = ['train','test']
        self.data={}
        self.is_train=True
        self.classes={}
        for split in self.splits:
            if progress:
                print(f'Indexing split {split}')
            self.data[split]=[]
            split_dir = os.path.join(rootdir,split)
            assert os.path.exists(split_dir),f'Split {split_dir} should exists'
            cls_dirnames = os.listdir(split_dir)
            for cls_dirname in cls_dirnames:
                # We create the directory path
                cls_dir = os.path.join(split_dir,cls_dirname)
                # We parse the name
                cls_id,cls_name = cls_dirname.replace(' ','').split('-')
                self.classes[int(cls_id)]=cls_name
                for i,imgname in enumerate(tqdm(os.listdir(cls_dir),disable=not progress)):
                    if nmax_per_class is not None:
                        if i>=nmax_per_class:
                            break
                    imgpath = os.path.join(cls_dir,imgname)
                    # Ici ajouter l'image dans le dataset...
                    cls_img = ClassificationImage(imgpath,cls_id)
                    if cls_img.check():
                        self.data[split].append(cls_img)
                              
    def train(self,train:bool=True):
        self.is_train = train
    
    def eval(self,eval:bool=True):
        self.is_train = not eval
    
    def __len__(self):
        if self.is_train:
            return len(self.data['train'])
        else:
            return len(self.data['test'])

    def __getitem__(self, index):
        assert index<self.__len__(),f"index should be lower than {len(self)}"
        data = self.data['train'] if self.is_train else self.data['test']
        transform = self.train_transforms if self.is_train else self.test_transform
        item = data[index]
        item:ClassificationImage
        image = item.get_image()
        target = item.get_target()
        if transform:
            output = transform(image=image)
            image = output['image']

        return image,target,item.name


