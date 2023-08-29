import time
import torch
from os.path import join
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
from torchvision.models import ResNet
from torch.utils.data import DataLoader
from tempfile import TemporaryDirectory

def train_model(model:ResNet, loss_fn:CrossEntropyLoss, dataloaders:dict[str, DataLoader], optimizer:optim, scheduler:lr_scheduler, num_epochs=25, device = "cuda"):
    since = time.time()
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = join(tempdir, 'best_model_params.pt')
    
        torch.save(model.state_dict(), best_model_params_path)
        best_validation_acc = 0.0
        model.to(device)

        for epoch in range(num_epochs):
            print(f"Current at {epoch} epoch")
            model.train()
            model, avg_train_acc, avg_train_loss = __model_fit(model, dataloaders['train'], optimizer, scheduler, loss_fn, device)
            model.eval()
            avg_val_acc, avg_val_loss = __model_validate(model, dataloaders['val'], loss_fn, device)
            
            if avg_val_acc.item() > best_validation_acc:
                torch.save(model.state_dict(), best_model_params_path)
                best_validation_acc = avg_val_acc

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f"Highest validation accuracy is {best_validation_acc}")

        model.load_state_dict(torch.load(best_model_params_path))
    return model

def __model_fit(model:ResNet, train_dataloader:DataLoader, optimizer:optim, scheduler:lr_scheduler, loss_fn:CrossEntropyLoss, device:str):

    avg_epoach_train_acc = torch.zeros(1, dtype = torch.float32, device = device)
    avg_epoach_train_loss = torch.zeros_like(avg_epoach_train_acc)
    for inputs, labels in train_dataloader:
        inputs.to(device)
        labels.to(device)
        batch_train_loss, batch_train_predict = __predict_and_get_batch_loss(model, inputs, labels, loss_fn, device)
        batch_train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        avg_epoach_train_acc += torch.sum(batch_train_predict == labels)
        avg_epoach_train_loss += batch_train_loss

    return model, avg_epoach_train_acc.mean(), avg_epoach_train_loss.mean()

def __model_validate(model:ResNet, val_dataloader:DataLoader, loss_fn:CrossEntropyLoss, device:str):
    avg_epoach_validation_acc = torch.zeros(1, dtype = torch.float32, device = device)
    avg_epoach_validation_loss = torch.zeros_like(avg_epoach_validation_acc)
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs.to(device)
            labels.to(device)
            batch_val_loss, batch_val_predict = __predict_and_get_batch_loss(model, inputs, labels, loss_fn, device)
            avg_epoach_validation_acc += torch.sum(batch_val_predict == labels)
            avg_epoach_validation_loss += batch_val_loss
    return avg_epoach_validation_acc.mean(), avg_epoach_validation_loss.mean()


def __predict_and_get_batch_loss(model:ResNet, inputs:torch.tensor, labels:torch.tensor, loss_fn:CrossEntropyLoss, device:str):
    outputs = model(inputs)
    _, predicts = torch.max(outputs, 1)
    loss = loss_fn(outputs, labels)
    return loss, predicts
