from src.classification import ClassificationDataset
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',default="/mnt/c/BUSDATA/Datasets/MNIST/flat/")
parser.add_argument('--nmax',type=int,default=None)

def main(data_dir:str,nmax:int=None):
    cls_dataset = ClassificationDataset(data_dir,nmax_per_class=nmax)
    

if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
    