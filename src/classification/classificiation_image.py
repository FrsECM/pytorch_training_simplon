import os
from PIL import Image
import numpy as np
from typing import Dict


class ClassificationImage:
    def __init__(self,imgpath:str,cls_id:int,**kwargs):
        self.name = os.path.basename(imgpath)
        self.imgpath = imgpath
        self.cls_id = int(cls_id)
    def check(self):
        return os.path.exists(self.imgpath)
    def get_target(self):
        return self.cls_id
    def get_image(self):
        with Image.open(self.imgpath) as img_pil:
            img_np = np.array(img_pil.convert('RGB'))
        return img_np
    
    def to_dict(self):
        return self.__dict__
    
    def from_dict(cdict:Dict[str,str])->'ClassificationImage':
        assert 'imgpath' in cdict,'imagepath should be in dict'
        assert 'cls_id' in cdict,'cls_id should be in dict'
        cI = ClassificationImage(**cdict)
        return cI