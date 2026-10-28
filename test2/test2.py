import torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import random_split,DataLoader
from tqdm import tqdm
import csv,logging
import re
import random
import numpy as np
from pathlib import Path
from torchvision.transforms import v2
LOG_DIR=Path(__file__).resolve().parent/'logs'
logger=logging.getLogger('training')
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.setLevel(logging.INFO)
fmt=logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh = logging.FileHandler(LOG_DIR / "train.log", encoding="utf-8")   
fh.setFormatter(fmt)
logger.handlers[:] = [fh]            
logger.propagate = False             
CSV_PATH = LOG_DIR / "results.csv"
FIELDS = ["run_id", "hidden_dims", "activation", "dropout", "lr","loss",
          "batch_size", "epoch", "train_loss", "train_acc", "val_acc", "val_loss"]
W_DIR = LOG_DIR / "weights"
W_DIR.mkdir(parents=True, exist_ok=True)
def safe_name(s):
    return re.sub(r'[^0-9A-Za-z._-]+', '_', s)
new_file = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
csv_writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
csv_writer.writeheader()
if new_file:
    csv_writer.writeheader()
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" 
transfrom = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.286,),(0.353,))])
train_dataset=datasets.FashionMNIST(root=DATA_ROOT,train=True,download=True,transform=transfrom)
test_dataset=datasets.FashionMNIST(root=DATA_ROOT,train=False,download=True,transform=transfrom)
aug = v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))
generator=torch.Generator().manual_seed(seed)
train_subset,val_subset=random_split(train_dataset,[0.9,0.1],generator=generator)
class Myblock(nn.Module):
    def __init__(self,Cin,Cout,dropout,BN,activation,pool,res,kernel_size):
        super().__init__()
        self.meta = dict(cin=Cin, cout=Cout, dropout=dropout, bn=BN,
                 res=res, pool=str(pool), kernel_size=kernel_size)
        self.conv1=nn.Conv2d(Cin,Cout,kernel_size[0],padding=kernel_size[1])
        self.conv2=nn.Conv2d(Cout,Cout,kernel_size[0],padding=kernel_size[1])
        if res:self.conv3=nn.Conv2d(Cin,Cout,1,padding=0)
        if BN:
            self.bn1=nn.BatchNorm2d(Cout)
            self.bn2=nn.BatchNorm2d(Cout)
        self.activation=activation
        self.pool=pool
        self.dropout=nn.Dropout(dropout)
        self.bn=BN
        self.res=res
    def forward(self,x):
        x1=x
        x=self.conv1(x)
        if self.bn:x=self.bn1(x)
        x=self.activation(x)
        x=self.conv2(x)
        if self.bn:x=self.bn2(x)
        if self.res:x=x+self.conv3(x1)
        x=self.activation(x)
        if self.pool is not None:x=self.pool(x)
        x=self.dropout(x)
        return x
class Mynet(nn.Module):
    def __init__(self,hidden_dims,activation,dropout,BN,res,pool,mlp_dims,kernel_size):
        super().__init__()
        self.meta=dict(hidden_dims=hidden_dims,activation=str(activation),dropout=dropout,BN=BN,res=res,pool=str(pool),mlp_dims=mlp_dims,kernel_size=kernel_size)
        layers=[]
        a=1
        for h in hidden_dims:
            layers.append(Myblock(a,h,dropout,BN,activation,pool,res,kernel_size[0],kernel_size[1]))
            a=h
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten(1))
        if mlp_dims is not None:
            for i in mlp_dims:
                layers.append(nn.Linear(a,i))
                layers.append(activation)
                a=i
        layers.append(nn.Linear(a,10))  
        self.net=nn.Sequential(*layers)
    def forward(self,x):
        x=self.net(x)
        return x
def run_experiment(loader,loss_fn,model,optimizer,aug):
    model.train()
    total_loss=0.0
    correct=0
    total=0
    for x,y in loader:
        x,y=x.to('cuda'),y.to('cuda')
        x=aug(x)
        optimizer.zero_grad()
        y_pred=model(x)
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            loss=loss_fn(y_pred, y)
        else:
            loss=loss_fn(nn.functional.one_hot(y, num_classes=10).float(), nn.functional.softmax(y_pred, dim=1))
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()*x.size(0)
        _,predicted=torch.max(y_pred.data,1)
        total+=y.size(0)
        correct+=(predicted==y).sum().item()
    train_loss=total_loss/total
    train_accuracy=correct/total
    return train_loss,train_accuracy
def eval(model,loader,loss_fn):
    model.eval()
    total_loss=0.0
    correct=0
    total=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to('cuda'),y.to('cuda')
            y_pred=model(x)
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                loss=loss_fn(y_pred, y)
            else:
                loss=loss_fn(nn.functional.one_hot(y, num_classes=10).float(), nn.functional.softmax(y_pred, dim=1))
            total_loss+=loss.item()*x.size(0)
            _,predicted=torch.max(y_pred.data,1)
            total+=y.size(0)
            correct+=(predicted==y).sum().item()
    avg_loss=total_loss/total
    accuracy=correct/total
    return avg_loss,accuracy
def run(hidden_dims,activation,dropout,loss_fn,lr,batch_size,res,pool,mlp_dims,kernel_size,epoch):
    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(train_subset,batch_size,shuffle=True,generator=generator)
    valid_loader=DataLoader(val_subset,batch_size,shuffle=False,generator=generator)
    torch.manual_seed(seed)
    model=Mynet(hidden_dims,activation,dropout,res,pool,mlp_dims,kernel_size)
    model=model.to('cuda')
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    best_val, best_ep, best_state = -1.0, 0, None
    run_id=f"{hidden_dims}_{activation.__class__.__name__}_{dropout}_{loss_fn.__class__.__name__}_{lr}_{batch_size}_{res}_{pool}_{mlp_dims}_{kernel_size}"
    if (W_DIR/f"{safe_name(run_id)}.pt").exists():
        logger.info(f"SKIP {run_id}: checkpoint exists")
    logger.info(f"Run ID: {run_id}, Hidden Dims: {hidden_dims}, Activation: {activation.__class__.__name__}, Dropout: {dropout}, Loss: {loss_fn.__class__.__name__}, LR: {lr}, Batch Size: {batch_size}, Residual: {res}, Pooling: {pool}, MLP Dims: {mlp_dims}, Kernel Size: {kernel_size}")
    bar=tqdm(range(1,epoch+1), desc=f"Training {hidden_dims} with LR {lr} and Batch Size {batch_size}")
    for e in bar:
        train_loss,train_accuracy=run_experiment(train_loader,loss_fn,model,optimizer,aug)
        logger.info(f"Epoch {e}: Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        val_loss,val_acc=eval(model,valid_loader,loss_fn)
        logger.info(f"Epoch {e}: Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        if val_acc>best_val:
            best_val=val_acc
            best_ep=e
            best_state=model.state_dict()
    bar.set_postfix(train_loss=train_loss,val_loss=val_loss,train_acc=train_accuracy,val_acc=val_acc)
    csv_writer.writerow({
                "run_id": run_id, "hidden_dims": str(model.hidden_dims), "activation": "GELU",
                "dropout": 0.1, "lr": 1e-4, "batch_size": 64,
                "epoch": e, "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_accuracy:.6f}",
                "val_acc": f"{val_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "loss": loss_fn.__class__.__name__,
            })
    csv_file.flush()
    bar.close()
    model.load_state_dict(best_state)
    ckpt={
        "state_dict":best_state,
        "run_id":run_id,
        "config":{
            "hidden_dims":list(model.hidden_dims),
            "activation":model.activation.__class__.__name__,
            "dropout":model.dropout_rate,
            "lr":optimizer.param_groups[0]["lr"],
            "batch_size":train_loader.batch_size,
            "loss":loss_fn.__class__.__name__,
            "epochs":epoch,
            "seed":seed,
            "augment":"v2.RandomAffine(10deg,translate=0.05,scale=0.9-1.1),batch-level,GPU,train-only",
        },
        "best_epoch":best_ep,
        "best_val_acc":best_val,
    }
    torch.save(ckpt,W_DIR/f"{safe_name(run_id)}.pt")
    logger.info(f"BEST {run_id}: epoch={best_ep}, val_acc={best_val:.4f}")
hidden_dims_list=[[64],[256],[1024],[64,256],[256,1024],[64,1024],[64,256,1024]]
pool_list=[None,nn.MaxPool2d(2),nn.AvgPool2d(2)]
mlp_dims_list=[None,[512],[256,128]]
activation_list=[nn.ReLU(),nn.GELU(),nn.SiLU(),nn.Sigmoid]
loss_list=[nn.CrossEntropyLoss(),nn.MSELoss()]
lr_list=[1e-2,1e-4,1e-6]
batch_size_list=[16,64,256]
res_list=[True,False]
bn_list=[True,False]
dropout_list=[0.0,0.1,0.3,0.5]
kernel_size_list=[[3,1],[5,2],[7,3]]
experiment_configs={
    
}
