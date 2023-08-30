import time
import torch
from os.path import join
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
from torchvision.models import ResNet
from torch.utils.data import DataLoader
from tempfile import TemporaryDirectory


def tutorial_train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=5):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model
 

def my_train_model(model:ResNet, loss_fn:CrossEntropyLoss, dataloaders:dict[str, DataLoader], dataset_size:dict['str', int], optimizer:optim, scheduler:lr_scheduler, device:torch.device, num_epochs=25) -> ResNet:
    since = time.time()
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = join(tempdir, 'best_model_params.pt')
    
        torch.save(model.state_dict(), best_model_params_path)
        best_validation_acc = 0.0
        model = model.to(device)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            model.train()
            model, avg_train_acc, avg_train_loss = __model_fit(model, dataloaders['train'], dataset_size['train'], optimizer, scheduler, loss_fn, device)
            model.eval()
            avg_val_acc, avg_val_loss = __model_validate(model, dataloaders['val'], dataset_size['val'], loss_fn, device)
            
            if avg_val_acc > best_validation_acc:
                torch.save(model.state_dict(), best_model_params_path)
                best_validation_acc = avg_val_acc

            print(f'Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f}')
            print(f'Vali Loss : {avg_val_loss:.4f} Acc: {avg_val_acc:.4f}')

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f"Highest validation accuracy is {best_validation_acc:.3f}")

        model.load_state_dict(torch.load(best_model_params_path))
    return model 

def __model_fit(model:ResNet, train_dataloader:DataLoader, train_size:int, optimizer:optim, scheduler:lr_scheduler, loss_fn:CrossEntropyLoss, device:str) -> tuple[ResNet, float, float]:

    avg_epoach_train_acc = torch.zeros(1, dtype = torch.float32, device = device, requires_grad = False)
    avg_epoach_train_loss = torch.zeros_like(avg_epoach_train_acc)
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        batch_train_loss, batch_train_predict = __predict_and_get_batch_loss(model, inputs, labels, loss_fn)
        batch_train_loss.backward()
        optimizer.step()

        avg_epoach_train_acc += torch.sum(batch_train_predict == labels.data)
        avg_epoach_train_loss += batch_train_loss * inputs.size(0)
        scheduler.step()

    return model, avg_epoach_train_acc.item() / train_size, avg_epoach_train_loss.item() / train_size

def __model_validate(model:ResNet, val_dataloader:DataLoader, val_size:int, loss_fn:CrossEntropyLoss, device:str) -> tuple[float, float]:
    avg_epoach_validation_acc = torch.zeros(1, dtype = torch.float32, device = device, requires_grad = False)
    avg_epoach_validation_loss = torch.zeros_like(avg_epoach_validation_acc)
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_val_loss, batch_val_predict = __predict_and_get_batch_loss(model, inputs, labels, loss_fn)
            avg_epoach_validation_acc += torch.sum(batch_val_predict == labels.data)
            avg_epoach_validation_loss += batch_val_loss
    return avg_epoach_validation_acc.item() / val_size, avg_epoach_validation_loss.item() / val_size


def __predict_and_get_batch_loss(model:ResNet, inputs:torch.tensor, labels:torch.tensor, loss_fn:CrossEntropyLoss) -> tuple[torch.tensor, torch.tensor]:
    outputs = model(inputs)
    _, predicts = torch.max(outputs, 1)
    loss = loss_fn(outputs, labels)
    return loss, predicts

    """
The inputs = inputs.to(device) and labels = labels.to(device) scattered around the code may seem redundant, but they are necessary.

the ```tensor.to()``` method does not perform tensor transformation in-place, instead they always return a copy of the original tensor. Therefor it
is necessary for us to accept the return value and consider the original tensor and the returned tensor as different instances.

If we fail to use the returned tensor, Pytorch gives us the following error message that is quite hard to understand.

---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[4], line 2
      1 from train_and_test_model import train_model
----> 2 model_ft = train_model(model_ft, criterion, dataloaders, dataset_sizes, optimizer_ft, exp_lr_scheduler, device = device)

File ~/Documents/LinuxFiles/TryPyTorch/SummerVacationMeeting/W4_TransferLearning/train_and_test_model.py:25, in train_model(model, loss_fn, dataloaders, dataset_size, optimizer, scheduler, num_epochs, device)
     23 print(f"Current at {epoch} epoch")
     24 model.train()
---> 25 model, avg_train_acc, avg_train_loss = __model_fit(model, dataloaders['train'], dataset_size['train'], optimizer, scheduler, loss_fn, device)
     26 model.eval()
     27 avg_val_acc, avg_val_loss = __model_validate(model, dataloaders['val'], dataset_size['val'], loss_fn, device)

File ~/Documents/LinuxFiles/TryPyTorch/SummerVacationMeeting/W4_TransferLearning/train_and_test_model.py:47, in __model_fit(model, train_dataloader, train_size, optimizer, scheduler, loss_fn, device)
     43 avg_epoach_train_loss = torch.zeros_like(avg_epoach_train_acc)
     44 for inputs, labels in train_dataloader:
     45     # inputs = inputs.to(device)
     46     # labels = labels.to(device)
---> 47     batch_train_loss, batch_train_predict = __predict_and_get_batch_loss(model, inputs, labels, loss_fn, device)
     48     batch_train_loss.backward()
     49     optimizer.step()

File ~/Documents/LinuxFiles/TryPyTorch/SummerVacationMeeting/W4_TransferLearning/train_and_test_model.py:74, in __predict_and_get_batch_loss(model, inputs, labels, loss_fn, device)
     71 def __predict_and_get_batch_loss(model:ResNet, inputs:torch.tensor, labels:torch.tensor, loss_fn:CrossEntropyLoss, device:str) -> tuple[torch.tensor, torch.tensor]:
     72     # labels = labels.to(device)
...
    458                     _pair(0), self.dilation, self.groups)
--> 459 return F.conv2d(input, weight, bias, self.stride,
    460                 self.padding, self.dilation, self.groups)

RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor
    """