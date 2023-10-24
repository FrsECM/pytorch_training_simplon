from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm
from .classificiation_image import ClassificationImage

class ClassificationCALTECH(Dataset):
    def __init__(self,rootdir:str,progress:bool=True,nmax_per_class:int=None):
        assert os.path.exists(rootdir),'Root dir should be an existing directory.'
        self.train_transforms=A.Compose([
            A.Resize(256,256),
            A.RandomBrightnessContrast(p=0.5),
            A.GridDistortion(),
            A.ToFloat(max_value=255),
            A.Normalize(mean=0.5,std=0.5),
            ToTensorV2()
        ])
        self.test_transform=A.Compose([
            A.Resize(256,256),
            A.ToFloat(max_value=255),
            A.Normalize(mean=0.5,std=0.5),
            ToTensorV2()
        ])
        self.data=[]
        self.is_train=True
        self.classes={}
        cls_dirnames = tqdm(os.listdir(rootdir),disable=not progress)
        for cls_id,cls_dirname in enumerate(cls_dirnames):
            cls_dirnames.set_postfix(**{'cls_name':cls_dirname})
            # We create the directory path
            cls_dir = os.path.join(rootdir,cls_dirname)
            # We parse the name
            self.classes[cls_id]=cls_dirname
            imgnames = os.listdir(cls_dir)
            for i,imgname in enumerate(imgnames):
                if nmax_per_class is not None:
                    if i>=nmax_per_class:
                        break
                imgpath = os.path.join(cls_dir,imgname)
                # Ici ajouter l'image dans le dataset...
                cls_img = ClassificationImage(imgpath,cls_id)
                if cls_img.check():
                    self.data.append(cls_img)
                              
    def train(self,train:bool=True):
        self.is_train = train
    
    def eval(self,eval:bool=True):
        self.is_train = not eval
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        assert index<self.__len__(),f"index should be lower than {len(self)}"
        transform = self.train_transforms if self.is_train else self.test_transform
        item = self.data[index]
        item:ClassificationImage
        image = item.get_image()
        target = item.get_target()
        if transform:
            output = transform(image=image)
            image = output['image']

        return image,target,item.name


