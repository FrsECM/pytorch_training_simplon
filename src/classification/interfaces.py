from abc import ABC,abstractclassmethod, abstractmethod
from torch.utils.data import Dataset

class IClassificationDataset(Dataset,ABC):
    @property
    @abstractmethod
    def classes(self):pass

    @property
    @abstractmethod
    def train_indices(self):pass

    @property
    @abstractmethod
    def val_indices(self):pass

    @property
    @abstractmethod
    def test_indices(self):pass

    @abstractmethod
    def eval(self,eval:bool):pass

    @abstractmethod
    def train(self,train:bool):pass