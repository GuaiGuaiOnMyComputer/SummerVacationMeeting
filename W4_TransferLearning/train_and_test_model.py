import time
import torch
from os.path import join
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
from torchvision.models import ResNet
from torch.utils.data import DataLoader
from tempfile import TemporaryDirectory

def train_model(model:ResNet, loss_fn:CrossEntropyLoss, train_dataloader:DataLoader, optimizer:optim, scheduler:lr_scheduler, num_epochs=25, device = "cuda"):
    since = time.time()
    model.train()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = join(tempdir, 'best_model_params.pt')
    
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            # print(f'Epoch {epoch}/{num_epochs - 1}')
            # print('-' * 10)

            # Iterate over data.
            for inputs, labels in train_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                scheduler.step()

                epoch_loss = running_loss.mean()
                epoch_acc = running_corrects.double().mean()

                # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model