from src.classification import ClassificationCALTECH
from src.classification.models import Resnet18
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_file',default="/mnt/c/BUSDATA/Datasets/CALTECH256/caltech.json")
parser.add_argument('--nmax',type=int,default=None)

def main(dataset_file:str,nmax:int=None):
    dataset = ClassificationCALTECH.from_json(dataset_file)
    model = Resnet18(in_channels=3,nClasses=len(dataset.classes))

    

if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
    