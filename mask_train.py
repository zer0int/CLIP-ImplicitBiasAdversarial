import foolbox
from utils.training import ess_train, adv_train
import torch
import argparse
from torch.utils.data import DataLoader
import os
from utils.data import get_dataloaders, AdversarialDataset
from utils.models import get_model, get_clip_model, ClipWrapper
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import warnings
warnings.filterwarnings('ignore')

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

parser = argparse.ArgumentParser()

parser.add_argument('--attack', type=str, default="FMN", help="attack type")
parser.add_argument('--model', type=str, default="resnet20", help="model architecture")
parser.add_argument('--mask', type=str, default="essential", help="type of mask (essential or adversarial)")
args = parser.parse_args()

def convert_image_to_rgb(image):
    return image.convert("RGB")

device = 'cuda'

transforming = Compose([
    Resize(size=224, interpolation=BICUBIC, max_size=None, antialias=True),
    Resize(size=(224, 224)),
    CenterCrop(size=(224, 224)),
    convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


def debug_unnormalized_images(clean_imgs, adv_imgs):
    pass
    #print(f"Unnormalized clean images min: {clean_imgs.min()}, max: {clean_imgs.max()}, mean: {clean_imgs.mean()}")
    #print(f"Unnormalized adversarial images min: {adv_imgs.min()}, max: {adv_imgs.max()}, mean: {adv_imgs.mean()}")


save_figs = True
model_name = args.model
if model_name in ['resnet18', 'vit', 'clipvit']:
    dataset = 'imagenette'
else:
    dataset = 'cifar10'
lam = 0.1 if dataset == 'imagenette' else 0.5

eps = 8 / 255
if 'PGD' in args.attack and len(args.attack) > 3:
    eps = int(args.attack[4:]) / 255
    print(eps)

image_size = 32 if dataset == 'cifar10' else 224
attack = args.attack
batch_size = 36
if args.mask == 'essential':
    path = "./essential/" + model_name + "/"
elif args.mask == 'adversarial':
    path = "./adversarial/" + attack + "/" + model_name + "/"
n_classes = 10

if not os.path.exists(path):
    for i in range(n_classes):
        os.makedirs(path + "figures/" + str(i), exist_ok=True)
        os.makedirs(path + "masks/" + str(i), exist_ok=True)

dataloaders = get_dataloaders(dataset, batch_size, batch_size, shuffle_train=True, shuffle_test=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_name == 'clipvit':
    base_model, preprocess = get_clip_model()
else:
    base_model = get_model(model_name)

base_model = base_model.to(device)
base_model.load_state_dict(torch.load("trained_models/" + model_name + "/clean.pt"))
base_model.eval()
fmodel = foolbox.models.PyTorchModel(base_model, bounds=(0, 1))

if dataset == 'cifar10':
    adv_dataloader = DataLoader(AdversarialDataset(fmodel, model_name, attack, dataloaders['test'], image_size, 'test', eps=eps), batch_size=batch_size, shuffle=False)
elif dataset == 'imagenette':
    if model_name == 'clipvit':
        # Adjust dataloader to preprocess images for CLIP model
        processed_data = []
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        for images, labels in dataloaders['all']:
            images = [unnormalize(transforming(transforms.ToPILImage()(image)), mean, std).unsqueeze(0) for image in images]
            images = torch.cat(images).to(device)
            processed_data.append((images, labels.to(device)))
        dataloaders['all'] = processed_data

    adv_dataloader = DataLoader(AdversarialDataset(fmodel, model_name, attack, dataloaders['all'], image_size, 'all', eps=eps), batch_size=batch_size, shuffle=False)

idx = 0
for x, xadv, y, yadv in tqdm(adv_dataloader):
    if args.mask == 'essential':
        idx = ess_train(base_model, x, y, lam, idx, path, image_size, save_figs)
    elif args.mask == 'adversarial':
        idx = adv_train(base_model, x, xadv, y, lam, idx, path, image_size, save_figs)

# After the loop, check unnormalized values
#clean_images = torch.cat([x for x, _, _, _ in adv_dataloader])
#adv_images = torch.cat([xadv for _, xadv, _, _ in adv_dataloader])
#unnormalized_clean_images = unnormalize(clean_images, mean, std)
#unnormalized_adv_images = unnormalize(adv_images, mean, std)
#debug_unnormalized_images(unnormalized_clean_images, unnormalized_adv_images)
