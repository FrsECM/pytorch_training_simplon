from torch.utils.data import Dataset
import os
from tqdm import tqdm

class ClassificationDataset(Dataset):
    def __init__(self,rootdir:str,progress:bool=True,nmax_per_class:int=None):
        assert os.path.exists(rootdir),'Root dir should be an existing directory.'
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


class ClassificationImage:
    def __init__(self,imgpath:str,cls_id:int):
        self.name = os.path.basename(imgpath)
        self.imgpath = imgpath
        self.cls_id = int(cls_id)
    
    def check(self):
        return os.path.exists(self.imgpath)