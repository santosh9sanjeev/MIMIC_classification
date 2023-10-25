from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import argparse
import PIL
import time
from dataset.MIMIC_dataset import MIMIC_Dataset, XRayCenterCrop, XRayResizer
from models.model import ImageNetModel, CLIPModel, ViTModel, CheXpertModel

from transformers import AutoImageProcessor, ViTForImageClassification
import torchvision.models as models
from sklearn.metrics import f1_score, roc_auc_score
import sys
import random
from timm.models import vit_base_patch16_224
from torchvision import datasets, transforms
import torchxrayvision as xrv
import os
from scripts import train_utils

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model', default='vit_b_16', type=str, help='pretrained model')
    parser.add_argument('--task', default='binary-class', type=str, help='available things {multi-label, multi-class, binary-class}')
    parser.add_argument('--csv', default='/home/santosh.sanjeev/model-soups/my_soups/metadata/RSNA_final_df.csv', type=str, help='Data directory')
    parser.add_argument('--data_dir', default='/home/santosh.sanjeev/rsna_18/train/', type=str, help='csv file containing the stats')
    parser.add_argument('--n_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--initialisation', default='imagenet', type=str, help='weight initialisation')
    parser.add_argument('--dataset', default='pneumoniamnist', type=str, help='which dataset')
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--lp_ft', default='LP', type=str, help='which type of finetuning')
    
    parser.add_argument('--device', default='cuda', type=str, help='which device')
    parser.add_argument('--norm', default=0.5, type=float, help='which norm')

    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--output_model_path', default='/home/santosh.sanjeev/model-soups/my_soups/checkpoints/full_finetuning/pneumoniamnist/initial_full_finetuning_model.pth', type=str, help='model path')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set seed for PyTorch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    NUM_EPOCHS = args.n_epochs
    BATCH_SIZE = args.batch_size
    lr = args.lr
    load_model = args.model
    data_aug = None

    print(args.use_pretrained)
    
    if args.norm!=0.5:
        print('USING IMAGENET NORM')
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
    else:
        print('NOT USING IMAGENET NORM')
        mean = [0.5,]
        std = [0.5,]
        normalize = transforms.Normalize(mean=mean, std=std)

    # preprocessing
    data_transform = transforms.Compose([XRayCenterCrop(), XRayResizer(512)])


    train_dataset = MIMIC_Dataset(imgpath="/share/ssddata/physionet.org/files/mimic-cxr-jpg/2.0.0/files/", 
                                  csvpath=args.data_dir + "mimic-cxr-2.0.0-chexpert.csv.gz",
                                  metacsvpath=args.data_dir + "mimic-cxr-2.0.0-metadata.csv.gz",
                                  splitpath = args.data_dir + "mimic-cxr-2.0.0-split.csv.gz", split = 'train',
                                  transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
 
    val_dataset = MIMIC_Dataset(imgpath="/share/ssddata/physionet.org/files/mimic-cxr-jpg/2.0.0/files/", 
                                  csvpath=args.data_dir + "mimic-cxr-2.0.0-chexpert.csv.gz",
                                  metacsvpath=args.data_dir + "mimic-cxr-2.0.0-metadata.csv.gz",
                                  splitpath = args.data_dir + "mimic-cxr-2.0.0-split.csv.gz", split = 'validate',
                                  transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    test_dataset = MIMIC_Dataset(imgpath="/share/ssddata/physionet.org/files/mimic-cxr-jpg/2.0.0/files/", 
                                  csvpath=args.data_dir + "mimic-cxr-2.0.0-chexpert.csv.gz",
                                  metacsvpath=args.data_dir + "mimic-cxr-2.0.0-metadata.csv.gz",
                                  splitpath = args.data_dir + "mimic-cxr-2.0.0-split.csv.gz", split = 'test',
                                  transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = args.num_workers)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = args.num_workers)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = args.num_workers)
    
    print(args.dataset)
    print(len(train_dataset))
    print("===================")
    print(len(val_dataset))
    print("===================")
    print(len(test_dataset))
    

    #model
    if args.initialisation == 'imagenet' or args.initialisation == 'chexpert':
        if args.use_pretrained:
            print('USING PRETRAINED WEIGHTS')
            if args.model == 'resnet50':
                print('RESNET-50')
                full_model =  getattr(models, load_model)(pretrained=True)
                full_model.head = nn.Linear(full_model.head.in_features, args.n_classes)


            elif args.model == 'densenet121-res224-chex':
                print('CHEXPERT DENSENET')
                full_model = CheXpertModel(xrv.models.DenseNet(weights="densenet121-res224-chex"),2) # CheXpert (Stanford)
                full_model.classifier = nn.Linear(full_model.classifier.in_features, args.n_classes)

            elif args.model == 'densenet121':
                print('IMAGENET DENSENET')
                full_model =  getattr(models, load_model)(pretrained=True)
                full_model.head = nn.Linear(full_model.head.in_features, args.n_classes)


            elif args.model=='vit_b_16':
                print('ViT')
                full_model = vit_base_patch16_224(pretrained=True)
                full_model.head = nn.Linear(full_model.head.in_features, args.n_classes)

        else:
            print('NOT USING PRETRAINED WEIGHTS')
            if args.model == 'resnet50':
                print('RESNET-50')
                full_model =  getattr(models, load_model)(pretrained=False)
                full_model.head = nn.Linear(full_model.head.in_features, args.n_classes)

            elif args.model == 'densenet121':
                print('IMAGENET DENSENET')
                full_model =  getattr(models, load_model)(pretrained=False)
                full_model.head = nn.Linear(full_model.head.in_features, args.n_classes)

            elif args.model=='vit_b_16':
                print('ViT')
                full_model = vit_base_patch16_224(pretrained=False)
                full_model.head = nn.Linear(full_model.head.in_features, args.n_classes)

    print(full_model)
    if args.lp_ft=='LP':
        print('ONLY LINEAR PROBING')
        for name,param in full_model.backbone.named_parameters():
            if not name.startswith('fc'):
                # print('LPPPPPPPPPPPPPP')
                param.requires_grad = False
    else:
        print('FULL FINETUNING')
    
    full_model.to(device)
    criterion = nn.CrossEntropyLoss()    
    optimizer = optim.AdamW(full_model.parameters(), lr=args.lr,weight_decay=1e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    train_utils.train(full_model, train_dataset, train_loader, val_loader, device, args)
