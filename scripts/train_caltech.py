from src.classification import ClassificationCALTECH,ClassificationTrainer
from src.classification.models import Resnet18
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_file',default="/mnt/c/BUSDATA/Datasets/CALTECH256/caltech.json")
parser.add_argument('--log_dir',default="/mnt/c/BUSDATA/Datasets/CALTECH256/tensorboard")
parser.add_argument('--nmax',type=int,default=None)
parser.add_argument('--train_ratio',type=float,default=0.8)
parser.add_argument('--val_ratio',type=float,default=0.1)
parser.add_argument('--test_ratio',type=float,default=0.1)
parser.add_argument('--learning_rate',type=float,default=1e-4)
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--epochs',type=int,default=15)

def main(
        dataset_file:str,
        log_dir:str,
        train_ratio:float,
        val_ratio:float,
        test_ratio:float,
        learning_rate:float,
        batch_size:int,
        epochs:int,
        nmax:int=None):
    ####### We set the dataset
    dataset = ClassificationCALTECH.from_json(dataset_file)
    dataset.split(train_ratio,val_ratio,test_ratio,seed=42)
    ####### We set the model
    model = Resnet18(in_channels=3,nClasses=len(dataset.classes))
    trainer = ClassificationTrainer(model,log_dir=log_dir,log_batch=True)
    trainer.set_AdamOptimizer(learning_rate,momentum=0.9)
    trainer.fit(dataset,batch_size,epochs)
    

if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
    