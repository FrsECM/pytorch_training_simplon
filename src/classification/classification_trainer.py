import torch
import torch.nn as nn
from torch.optim import Adam,SGD
from torch.utils.tensorboard.writer import SummaryWriter
from .interfaces import IClassificationDataset
from torch.utils.data import Dataset,DataLoader,Subset,SubsetRandomSampler
from torch.cuda.amp.grad_scaler import GradScaler
import os
from tqdm import tqdm
from datetime import datetime
from .utils import grad_norm,plot_grad_flow

class ClassificationTrainer:
    def __init__(self,model:nn.Module,log_dir:str,log_batch:bool=False):
        log_dir_name = datetime.now().strftime('%Y%m%d-%H%M%S-CLS')
        self.log_dir = os.path.join(log_dir,log_dir_name)
        os.makedirs(self.log_dir,exist_ok=True)
        print(f'Logging in {self.log_dir}')
        self.writer = SummaryWriter(self.log_dir)
        self.model = model
        self.log_batch=log_batch
        self.best_loss = None
        self.saved_model_path = None

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.dtype = torch.float16
            self.device_type = 'cuda'
            self.scaler = GradScaler()
            self.amp = True
        else:
            self.device = torch.device('cpu')
            self.dtype = torch.float32
            self.device_type = 'cpu'
            self.amp = False
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def set_AdamOptimizer(self,Lr0:float,**kwargs):
        self.optimizer=Adam(params=self.model.parameters(),lr=Lr0)
    def set_SGDOptimizer(self,Lr0:float,momentum:float=0.9,**kwargs):
        self.optimizer=SGD(params=self.model.parameters(),lr=Lr0,momentum=momentum)

    def fit(self,dataset:IClassificationDataset,batch_size:int=4,epochs:int=5):
        assert self.optimizer is not None,"set_optimizer should be use before fitting"
        assert len(dataset.train_indices)>0,"dataset should have been splited before use."
        # We create dataloaders
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(indices=dataset.train_indices),
            num_workers=os.cpu_count())
        val_loader = DataLoader(Subset(dataset,dataset.val_indices),batch_size=batch_size,shuffle=False)
        
        self.model = self.model.to(self.device)

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            tbar = tqdm(train_loader)
            self.model=self.model.train()
            dataset.train()
            outputs_train = {}
            # Training Loop
            for i,batch in enumerate(tbar):
                outputs_batch={}
                # We reset the gradient state to be zeros.
                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device_type,dtype=self.dtype,enabled=self.amp):
                    X,y,name = batch
                    X=X.to(self.device)
                    y=y.to(self.device)
                    pred = self.model(X)
                    total_loss=self.criterion(pred,y.long())
                    # We recompute gradients based to the loss (partial derivatives)
                    if self.scaler:
                        self.scaler.scale(total_loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        total_loss.backward()
                        self.optimizer.step()
                    g_norm = grad_norm(self.model)
                    if not i%100:
                        fig = plot_grad_flow(self.model.named_parameters())
                        self.writer.add_figure(f'Train/Grad_norm{i}',fig,global_step=epoch)
                        del(fig)
                    outputs_batch['total_loss']=total_loss.item()
                    outputs_batch['g_norm']=g_norm.item()
                    # We update model parameters.
                    tbar.set_postfix(**outputs_batch)
                    for k,v in outputs_batch.items():
                        if k in outputs_train:
                            outputs_train[k].append(v)
                        else:
                            outputs_train[k]=[v]
                    if self.log_batch:
                        for k,v in outputs_batch.items():
                            self.writer.add_scalar(f'Train/{k}_batch',v,global_step=epoch*len(train_loader)+i)
            # We log in tensorboard
            for k,v in outputs_train.items():
                self.writer.add_scalar(f'Train/{k}_mean',torch.tensor(v).mean().item(),global_step=epoch)
            # Validation Loop
            outputs_val = {}
            vbar = tqdm(val_loader)
            self.model=self.model.eval()
            dataset.eval()
            for i,batch in enumerate(vbar):
                outputs_batch={}
                # We reset the gradient state to be zeros.
                with torch.no_grad():
                    with torch.autocast(device_type=self.device_type,dtype=self.dtype,enabled=self.amp):
                        X,y,name = batch
                        X=X.to(self.device)
                        y=y.to(self.device)
                        pred = self.model(X)
                        total_loss=self.criterion(pred,y.long())
                        # We recompute gradients based to the loss (partial derivatives)
                        outputs_batch['total_loss']=total_loss.item()
                    # We update model parameters.
                    vbar.set_postfix(**outputs_batch)
                    for k,v in outputs_batch.items():
                        if k in outputs_val:
                            outputs_val[k].append(v)
                        else:
                            outputs_val[k]=[v]
                    if self.log_batch:
                        for k,v in outputs_batch.items():
                            self.writer.add_scalar(f'Val/{k}_batch',v,global_step=epoch*len(train_loader)+i)
            # We log in tensorboard
            for k,v in outputs_val.items():
                loss_val = torch.tensor(v).mean().item()
                self.writer.add_scalar(f'Val/{k}_mean',loss_val,global_step=epoch)
                if k=='total_loss':
                    if self.best_loss is None or loss_val<self.best_loss:
                        new_model_path = os.path.join(self.log_dir,f"model_l{round(loss_val,2)}_e{epoch}.pth")
                        if self.saved_model_path is not None:
                            os.remove(self.saved_model_path)
                        torch.save(self.model.state_dict(),new_model_path)
                        self.saved_model_path=new_model_path