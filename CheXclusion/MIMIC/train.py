import time
import csv
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import datetime
import torch.optim
import torch.utils.data
from torchvision import  models
from torch import nn
import torch
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from dataset import MIMICCXRDataset
from utils import *
from batchiterator import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import random
import numpy as np



def ModelTrain(train_df, val_df, path_image, ModelType, CriterionType, device,LR, output_folder):



    # Training parameters
    batch_size = 192 #48

    workers = 24  # mean: how many subprocesses to use for data loading.
    N_LABELS = 14
    start_epoch = 0
    num_epochs = 64  # number of epochs to train for (if early stopping is not triggered)

    # dest_dir = os.path.join(result_path, datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S-%f'))  # ???
    # os.makedirs(dest_dir)


    random_seed = 47 #random.randint(0,100)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    train_df_size = len(train_df)
    val_df_size = len(val_df)
    



    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # train_loader = torch.utils.data.DataLoader(
    #     MIMICCXRDataset(train_df, path_image=path_image, transform=transforms.Compose([normalize])),
    #     batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(
    #     MIMICCXRDataset(val_df,path_image=path_image, transform=transforms.Compose([normalize])),
    #     batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    train_dataset = MIMICCXRDataset(train_df, path_image=path_image, transform=transforms.Compose([
                                                                    transforms.ToPILImage(),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.RandomRotation(15),
                                                                    transforms.Scale(256),
                                                                    transforms.CenterCrop(256),
                                                                    transforms.ToTensor(),
                                                                    normalize
                                                                ]))
    val_dataset = MIMICCXRDataset(val_df,path_image=path_image, transform=transforms.Compose([
                                                                transforms.ToPILImage(),
                                                                transforms.Scale(256),
                                                                transforms.CenterCrop(256),
                                                                transforms.ToTensor(),
                                                                normalize
                                                            ]))
    
    print(len(train_dataset), len(val_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    if ModelType == 'densenet':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features


        model.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    

        

    if ModelType == 'Resume':
        CheckPointData = torch.load('results_v2/checkpoint.pth')
        model = CheckPointData['model']



    # if torch.cuda.device_count() > 1:
    #     print('Using', torch.cuda.device_count(), 'GPUs')
    #     model = nn.DataParallel(model)

    model = model.to(device)
    
    if CriterionType == 'BCELoss':
        criterion = nn.BCELoss().to(device)

    epoch_losses_train = []
    epoch_losses_val = []

    since = time.time()

    best_loss = 999999
    best_epoch = -1
    best_auc = 0
    best_f1 = 0
    best_accuracy = 0
#--------------------------Start of epoch loop
    for epoch in (range(start_epoch, num_epochs + 1)):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        phase = 'train'
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        running_loss, train_preds, train_labels = BatchIterator(model=model, phase=phase, Data_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device)
        epoch_loss_train = running_loss / train_df_size
        epoch_losses_train.append(epoch_loss_train)#.item())
        # print("Train_losses:", epoch_losses_train)
        # print(train_preds)
        # train_accuracy = accuracy_score(np.round(train_labels), np.round(train_preds))
        train_auc = roc_auc_score(train_labels, train_preds)
        # train_f1 = f1_score(np.round(train_labels), np.round(train_preds), average = 'macro')
        print(f"Train Loss: {epoch_loss_train:.4f},  AUC: {train_auc:.4f}") #Accuracy: {train_accuracy:.4f},, F1: {train_f1:.4f} -", end=' ')
        print('hellloooooo')
        phase = 'val'
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        running_loss, val_preds, val_labels = BatchIterator(model=model, phase=phase, Data_loader=val_loader, criterion=criterion, optimizer=optimizer, device=device)
        epoch_loss_val = running_loss / val_df_size
        epoch_losses_val.append(epoch_loss_val)#.item())
        # print("Validation_losses:", epoch_losses_val)
        # val_accuracy = accuracy_score(np.round(val_labels), np.round(val_preds))
        val_auc = roc_auc_score(val_labels, val_preds)
        # val_f1 = f1_score(np.round(val_labels), np.round(val_preds), average = 'macro')
        print(f"Validation Loss: {epoch_loss_val:.4f}, AUC: {val_auc:.4f}") #, Accuracy: {val_accuracy:.4f}, , F1: {val_f1:.4f}")

        # checkpoint model if has best val loss yet
        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            best_epoch = epoch
            best_auc = val_auc
            checkpoint(model, best_loss, best_epoch, LR, output_folder, best_auc)

                # log training and validation loss over each epoch
        with open("results/log_train", 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            if (epoch == 1):
                logwriter.writerow(["epoch", "train_loss", "val_loss", "train_AUC", "val_AUC", "Seed", "LR"])
            logwriter.writerow([epoch, epoch_loss_train, epoch_loss_val, train_auc, val_auc, random_seed, LR])

# -------------------------- End of phase

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            if epoch_loss_val > best_loss:
                print("decay loss from " + str(LR) + " to " + str(LR / 2) + " as not seeing improvement in val loss")
                LR = LR / 2
                print("created new optimizer with LR " + str(LR))
                if ((epoch - best_epoch) >= 10):
                    print("no improvement in 10 epochs, break")
                    break
        #old_epoch = epoch 
    #------------------------- End of epoch loop
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    Saved_items(epoch_losses_train, epoch_losses_val, time_elapsed, batch_size)
    #
    model_path = os.path.join(output_folder, 'checkpoint.pth')
    checkpoint_best = torch.load(model_path)
    model = checkpoint_best['model']

    best_epoch = checkpoint_best['best_epoch']
    print(best_epoch)



    return model, best_epoch


