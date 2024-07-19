import torch
import argparse
from utils.data import get_dataloaders
from utils.training import train
from utils.models import get_model, get_clip_model
import os
from PIL import Image
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="resnet20", help="model architecture")
args = parser.parse_args()

model_name = args.model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 36

if model_name in ['resnet18', 'vit', 'clipvit']:
    dataset = 'imagenette'
else:
    dataset = 'cifar10'

dataloaders = get_dataloaders(dataset, batch_size, batch_size, shuffle_train=True, shuffle_test=False)

if model_name == 'clipvit':
    model, preprocess = get_clip_model()
else:
    model = get_model(model_name)

if model_name == 'resnet20':
    epochs = 200
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
elif model_name == 'clipvit':
    epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                0.01,
                epochs=epochs,
                steps_per_epoch=len(dataloaders['train']),
                pct_start=0.1
            )
else:
    epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                0.01,
                epochs=epochs,
                steps_per_epoch=len(dataloaders['train']),
                pct_start=0.1
            )

if not os.path.exists('trained_models/' + model_name):
    os.makedirs('trained_models/' + model_name)

print(f'Training {model_name} model')

model = model.to(device)

def tensor_to_pil(image_tensor):
    return transforms.ToPILImage()(image_tensor)

class CustomDataLoader:
    def __init__(self, processed_data):
        self.processed_data = processed_data
        self.dataset = self  # For compatibility with the train function

    def __iter__(self):
        return iter(self.processed_data)

    def __len__(self):
        return len(self.processed_data)

if model_name == 'clipvit':
    # Adjust dataloader to preprocess images for CLIP model
    for phase in ['train', 'test']:
        processed_data = []
        for images, labels in dataloaders[phase]:
            images = [preprocess(tensor_to_pil(image)).unsqueeze(0) for image in images]
            images = torch.cat(images).to(device)
            processed_data.append((images, labels.to(device)))
        dataloaders[phase] = CustomDataLoader(processed_data)

train(model, dataloaders, epochs, optimizer, scheduler)
torch.save(model.state_dict(), "trained_models/" + model_name + "/clean.pt")
