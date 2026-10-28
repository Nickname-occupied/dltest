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
csv_file = open(CSV_PATH, "a", newline="", encoding="utf-8")
csv_writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
if new_file:
    csv_writer.writeheader()
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" 
transfrom = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
train_dataset=datasets.MNIST(root=DATA_ROOT,train=True,download=True,transform=transfrom)
test_dataset=datasets.MNIST(root=DATA_ROOT,train=False,download=True,transform=transfrom)
aug = v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))
generator=torch.Generator().manual_seed(seed)
train_subset,val_subset=random_split(train_dataset,[0.9,0.1],generator=generator)
class mynet(nn.Module):
    def __init__(self,hidden_dims,activation,dropout_rate):
        super().__init__()
        self.hidden_dims=hidden_dims
        self.dropout_rate=dropout_rate
        self.flatten=nn.Flatten()
        self.activation=activation
        layers=[]
        input_dim=28*28
        for i in hidden_dims:
            layers.append(nn.Linear(input_dim,i))
            layers.append(self.activation)
            if dropout_rate>0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim=i
        layers.append(nn.Linear(input_dim,10))
        self.net=nn.Sequential(*layers)
    def forward(self,x):
        x=self.flatten(x)
        x=self.net(x)
        return x
activations=[nn.ReLU(),nn.Tanh(),nn.Sigmoid(),nn.GELU()]
hidden_dims=[[2048],[2048,2048],[1024,1024],[2048,2048,2048],[4096,4096],[256,256]]
looses=[nn.CrossEntropyLoss(),nn.MSELoss()]
learning_rates=[1e-2,1e-4,1e-6]
batch_sizes=[16,64,256]
dropout_rates=[0,0.1,0.3,0.5]
epoch=50
#1不同网络结构
for hidden_dim in hidden_dims:
    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(train_subset,batch_size=64,shuffle=True,generator=generator)
    valid_loader=DataLoader(val_subset,batch_size=64,shuffle=False,generator=generator)
    torch.manual_seed(seed)
    model=mynet(hidden_dim,nn.GELU(),0.1)
    model=model.to('cuda')
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    best_val, best_ep, best_state = -1.0, 0, None
    loss_fn=nn.CrossEntropyLoss()
    run_id=f"{model.hidden_dims}"
    if (W_DIR/f"{safe_name(run_id)}.pt").exists():
        logger.info(f"SKIP {run_id}: checkpoint exists")
        continue
    logger.info(f"Run ID: {run_id}, Hidden Dims: {model.hidden_dims}, Activation: GELU, Dropout: 0.1, LR: 1e-4, Batch Size: 64, Epochs: {epoch}")
    bar=tqdm(range(1,epoch+1), desc=f"Training {model.hidden_dims}")
    for e in bar:
        model.train()
        loss_sum=total=correct=0
        for x,y in train_loader:
            x,y=x.to('cuda'),y.to('cuda')
            x = aug(x)
            optimizer.zero_grad()
            y_pred=model(x)
            loss=loss_fn(y_pred,y)
            loss.backward()
            optimizer.step()
            loss_sum+=loss.item()*y.size(0)
            total+=y.size(0)
            correct+=(y_pred.argmax(1)==y).type(torch.float).sum().item()
        train_loss=loss_sum/total
        train_accuracy=correct/total
        logger.info(f"Epoch {e}: Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        model.eval()
        loss_sum=total=correct=0
        with torch.no_grad():
            for x,y in valid_loader:
                x,y=x.to('cuda'),y.to('cuda')
                y_pred=model(x)
                loss=loss_fn(y_pred,y)
                loss_sum+=loss.item()*y.size(0)
                correct+=(y_pred.argmax(1)==y).type(torch.float).sum().item()
                total+=y.size(0)
            val_loss=loss_sum/total
            val_acc=correct/total
            if val_acc > best_val:
                best_val, best_ep = val_acc, e
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        bar.set_postfix(train_loss=train_loss,val_loss=val_loss,train_acc=train_accuracy,val_acc=val_acc)
        logger.info(f"Epoch {e}: Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
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
#2.不同激活函数
for activation in activations:
    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(train_subset,batch_size=64,shuffle=True,generator=generator)
    valid_loader=DataLoader(val_subset,batch_size=64,shuffle=False,generator=generator)
    torch.manual_seed(seed)
    model=mynet([2048,2048],activation,0.1)
    model=model.to('cuda')
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    best_val, best_ep, best_state = -1.0, 0, None
    loss_fn=nn.CrossEntropyLoss()
    run_id=f"{model.hidden_dims}_{activation.__class__.__name__}"
    if (W_DIR/f"{safe_name(run_id)}.pt").exists():
        logger.info(f"SKIP {run_id}: checkpoint exists")
        continue
    logger.info(f"Run ID: {run_id}, Hidden Dims: {model.hidden_dims}, Activation: {activation.__class__.__name__}, Dropout: 0.1, LR: 1e-4, Batch Size: 64, Epochs: {epoch}")
    bar=tqdm(range(1,epoch+1), desc=f"Training {model.hidden_dims} with {activation.__class__.__name__}")
    for e in bar:
        model.train()
        loss_sum=total=correct=0
        for x,y in train_loader:
            x,y=x.to('cuda'),y.to('cuda')
            x = aug(x)
            optimizer.zero_grad()
            y_pred=model(x)
            loss=loss_fn(y_pred,y)
            loss.backward()
            optimizer.step()
            loss_sum+=loss.item()*y.size(0)
            total+=y.size(0)
            correct+=(y_pred.argmax(1)==y).type(torch.float).sum().item()
        train_loss=loss_sum/total
        train_accuracy=correct/total
        logger.info(f"Epoch {e}: Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        model.eval()
        loss_sum=total=correct=0
        with torch.no_grad():
            for x,y in valid_loader:
                x,y=x.to('cuda'),y.to('cuda')
                y_pred=model(x)
                loss=loss_fn(y_pred,y)
                loss_sum+=loss.item()*y.size(0)
                correct+=(y_pred.argmax(1)==y).type(torch.float).sum().item()
                total+=y.size(0)
            val_loss=loss_sum/total
            val_acc=correct/total
            if val_acc > best_val:
                best_val, best_ep = val_acc, e
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        bar.set_postfix(train_loss=train_loss,val_loss=val_loss,train_acc=train_accuracy,val_acc=val_acc)
        logger.info(f"Epoch {e}: Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        csv_writer.writerow({
            "run_id": run_id, "hidden_dims": str(model.hidden_dims), "activation": activation.__class__.__name__,
            "dropout": 0.1, "lr": 1e-4, "batch_size": 64,
            "epoch": e, "train_loss": f"{train_loss:.6f}", "train_acc": f"{train_accuracy:.6f}",
            "val_acc": f"{val_acc:.6f}", "val_loss": f"{val_loss:.6f}",
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
#3.不同损失函数
for loss_fn in looses:
    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(train_subset,batch_size=64,shuffle=True,generator=generator)
    valid_loader=DataLoader(val_subset,batch_size=64,shuffle=False,generator=generator)
    torch.manual_seed(seed)
    model=mynet([2048,2048],nn.GELU(),0.1)
    model=model.to('cuda')
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    best_val, best_ep, best_state = -1.0, 0, None
    run_id=f"{model.hidden_dims}_{loss_fn.__class__.__name__}"
    if (W_DIR/f"{safe_name(run_id)}.pt").exists():
        logger.info(f"SKIP {run_id}: checkpoint exists")
        continue
    logger.info(f"Run ID: {run_id}, Hidden Dims: {model.hidden_dims}, Activation: GELU, Dropout: 0.1, Loss Function: {loss_fn.__class__.__name__}, LR: 1e-4, Batch Size: 64, Epochs: {epoch}")
    bar=tqdm(range(1,epoch+1), desc=f"Training {model.hidden_dims} with {loss_fn.__class__.__name__}")
    for e in bar:
        model.train()
        loss_sum=total=correct=0
        for x,y in train_loader:
            x,y=x.to('cuda'),y.to('cuda')
            x = aug(x)
            optimizer.zero_grad()
            y_pred=model(x)
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                loss=loss_fn(y_pred, y)
            else:
                loss=loss_fn(nn.functional.one_hot(y, num_classes=10).float(), nn.functional.softmax(y_pred, dim=1))
            loss.backward()
            optimizer.step()
            loss_sum+=loss.item()*y.size(0)
            total+=y.size(0)
            correct+=(y_pred.argmax(1)==y).type(torch.float).sum().item()
        train_loss=loss_sum/total
        train_accuracy=correct/total
        logger.info(f"Epoch {e}: Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        model.eval()
        loss_sum=total=correct=0
        with torch.no_grad():
            for x,y in valid_loader:
                x,y=x.to('cuda'),y.to('cuda')
                y_pred=model(x)
                if isinstance(loss_fn, nn.CrossEntropyLoss):
                    loss=loss_fn(y_pred, y)
                else:
                    loss=loss_fn(nn.functional.one_hot(y, num_classes=10).float(), nn.functional.softmax(y_pred, dim=1))
                loss_sum+=loss.item()*y.size(0)
                correct+=(y_pred.argmax(1)==y).type(torch.float).sum().item()
                total+=y.size(0)
            val_loss=loss_sum/total
            val_acc=correct/total
            if val_acc > best_val:
                best_val, best_ep = val_acc, e
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        bar.set_postfix(train_loss=train_loss,val_loss=val_loss,train_acc=train_accuracy,val_acc=val_acc)
        logger.info(f"Epoch {e}: Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
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
#4.不同训练超参数
for lr in learning_rates:
    for batch_size in batch_sizes:
        generator=torch.Generator().manual_seed(seed)
        train_loader=DataLoader(train_subset,batch_size=batch_size,shuffle=True,generator=generator)
        valid_loader=DataLoader(val_subset,batch_size=batch_size,shuffle=False,generator=generator)
        torch.manual_seed(seed)
        model=mynet([2048,2048],nn.GELU(),0.1)
        model=model.to('cuda')
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        best_val, best_ep, best_state = -1.0, 0, None
        loss_fn=nn.CrossEntropyLoss()
        run_id=f"{model.hidden_dims}_lr_{lr}_batch_{batch_size}"
        if (W_DIR/f"{safe_name(run_id)}.pt").exists():
            logger.info(f"SKIP {run_id}: checkpoint exists")
            continue
        logger.info(f"Run ID: {run_id}, Hidden Dims: {model.hidden_dims}, Activation: GELU, Dropout: 0.1, LR: {lr}, Batch Size: {batch_size}, Epochs: {epoch}")
        bar=tqdm(range(1,epoch+1), desc=f"Training {model.hidden_dims} with LR {lr} and Batch Size {batch_size}")
        for e in bar:
            model.train()
            loss_sum=total=correct=0
            for x,y in train_loader:
                x,y=x.to('cuda'),y.to('cuda')
                x = aug(x)
                optimizer.zero_grad()
                y_pred=model(x)
                loss=loss_fn(y_pred,y)
                loss.backward()
                optimizer.step()
                loss_sum+=loss.item()*y.size(0)
                total+=y.size(0)
                correct+=(y_pred.argmax(1)==y).type(torch.float).sum().item()
            train_loss=loss_sum/total
            train_accuracy=correct/total
            logger.info(f"Epoch {e}: Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
            model.eval()
            loss_sum=total=correct=0
            with torch.no_grad():
                for x,y in valid_loader:
                    x,y=x.to('cuda'),y.to('cuda')
                    y_pred=model(x)
                    loss=loss_fn(y_pred,y)
                    loss_sum+=loss.item()*y.size(0)
                    correct+=(y_pred.argmax(1)==y).type(torch.float).sum().item()
                    total+=y.size(0)
                val_loss=loss_sum/total
                val_acc=correct/total
                if val_acc > best_val:
                    best_val, best_ep = val_acc, e
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bar.set_postfix(train_loss=train_loss,val_loss=val_loss,train_acc=train_accuracy,val_acc=val_acc)
            logger.info(f"Epoch {e}: Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            csv_writer.writerow({
                "run_id": run_id, "hidden_dims": str(model.hidden_dims), "activation": "GELU",
                "dropout": 0.1, "lr": lr, "batch_size": batch_size,
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
#5.不同dropout率
for dropout_rate in dropout_rates:
    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(train_subset,batch_size=64,shuffle=True,generator=generator)
    valid_loader=DataLoader(val_subset,batch_size=64,shuffle=False,generator=generator)
    torch.manual_seed(seed)
    model=mynet([2048,2048],nn.GELU(),dropout_rate)
    model=model.to('cuda')
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    best_val, best_ep, best_state = -1.0, 0, None
    loss_fn=nn.CrossEntropyLoss()
    run_id=f"{model.hidden_dims}_dropout_{dropout_rate}"
    if (W_DIR/f"{safe_name(run_id)}.pt").exists():
        logger.info(f"SKIP {run_id}: checkpoint exists")
        continue
    logger.info(f"Run ID: {run_id}, Hidden Dims: {model.hidden_dims}, Activation: GELU, Dropout: {dropout_rate}, LR: 1e-4, Batch Size: 64, Epochs: {epoch}")
    bar=tqdm(range(1,epoch+1), desc=f"Training {model.hidden_dims} with dropout {dropout_rate}")
    for e in bar:
        model.train()
        loss_sum=total=correct=0
        for x,y in train_loader:
            x,y=x.to('cuda'),y.to('cuda')
            x = aug(x)
            optimizer.zero_grad()
            y_pred=model(x)
            loss=loss_fn(y_pred,y)
            loss.backward()
            optimizer.step()
            loss_sum+=loss.item()*y.size(0)
            total+=y.size(0)
            correct+=(y_pred.argmax(1)==y).type(torch.float).sum().item()
        train_loss=loss_sum/total
        train_accuracy=correct/total
        logger.info(f"Epoch {e}: Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        model.eval()
        loss_sum=total=correct=0
        with torch.no_grad():
            for x,y in valid_loader:
                x,y=x.to('cuda'),y.to('cuda')
                y_pred=model(x)
                loss=loss_fn(y_pred,y)
                loss_sum+=loss.item()*y.size(0)
                correct+=(y_pred.argmax(1)==y).type(torch.float).sum().item()
                total+=y.size(0)
            val_loss=loss_sum/total
            val_acc=correct/total
            if val_acc > best_val:
                best_val, best_ep = val_acc, e
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        bar.set_postfix(train_loss=train_loss,val_loss=val_loss,train_acc=train_accuracy,val_acc=val_acc)
        logger.info(f"Epoch {e}: Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        csv_writer.writerow({
            "run_id": run_id, "hidden_dims": str(model.hidden_dims), "activation": "GELU",
            "dropout": dropout_rate, "lr": 1e-4, "batch_size": 64,
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
csv_file.close()
FINAL_RUN_ID="[2048, 2048]"   
ckpt_paths=[W_DIR/f"{safe_name(FINAL_RUN_ID)}.pt"]
if ckpt_paths[0].exists():
    _best=torch.load(ckpt_paths[0],map_location="cpu",weights_only=True)
    logger.info(f"FINAL MODEL source: {FINAL_RUN_ID}")
    _cfg=_best["config"]
    _acts={"ReLU":nn.ReLU(),"Tanh":nn.Tanh(),"Sigmoid":nn.Sigmoid(),"GELU":nn.GELU()}
    _model=mynet(_cfg["hidden_dims"],_acts[_cfg["activation"]],_cfg["dropout"]).to('cuda')
    _model.load_state_dict(_best["state_dict"])
    _model.eval()
    _loss=nn.CrossEntropyLoss() if _cfg["loss"]=="CrossEntropyLoss" else nn.MSELoss()
    _test_loader=DataLoader(test_dataset,batch_size=512)
    _loss_sum=_correct=0
    with torch.no_grad():
        for x,y in _test_loader:
            x,y=x.to('cuda'),y.to('cuda')
            _out=_model(x)
            if isinstance(_loss,nn.CrossEntropyLoss):
                _l=_loss(_out,y)
            else:
                _l=_loss(nn.functional.one_hot(y,num_classes=10).float(),
                         nn.functional.softmax(_out,dim=1))
            _loss_sum+=_l.item()*y.size(0)
            _correct+=(_out.argmax(1)==y).sum().item()
    _best["test_loss"]=_loss_sum/len(test_dataset)
    _best["test_acc"]=_correct/len(test_dataset)
    torch.save(_best,W_DIR/"final_model.pt")               
    logger.info(f"FINAL MODEL: {_best['run_id']}, best_epoch={_best['best_epoch']}, "
                f"val_acc={_best['best_val_acc']:.4f}, test_loss={_best['test_loss']:.4f}, "
                f"test_acc={_best['test_acc']:.4f}")
    print(f"final model: {_best['run_id']} val={_best['best_val_acc']:.4f} "
          f"test={_best['test_acc']:.4f} -> logs/weights/final_model.pt")
