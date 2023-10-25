
import torch
from utils import *
import numpy as np
from tqdm import tqdm
#from evaluation import *

def BatchIterator(model, phase, Data_loader, criterion, optimizer, device):
    grad_clip = 0.5
    print_freq = 2000
    running_loss = 0.0

    preds = []  # List to store predictions
    labels = []  # List to store labels

    for imgs, true_labels in tqdm(Data_loader):
        batch_size = imgs.shape[0]
        imgs = imgs.to(device)
        true_labels = true_labels.to(device)

        if phase == "train":
            optimizer.zero_grad()
            model.train()
        else:
            model.eval()

        with torch.set_grad_enabled(phase == "train"):
            outputs = model(imgs)
            loss = criterion(outputs, true_labels)

            if phase == 'train':
                loss.backward()
                if grad_clip is not None:
                    clip_gradient(optimizer, grad_clip)
                optimizer.step()

            running_loss += loss.item() * batch_size

            preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            labels.extend(true_labels.detach().cpu().numpy())

    return running_loss, preds, labels

# def BatchIterator(model, phase,Data_loader,criterion,optimizer,device):
#     # --------------------  Initial paprameterd
#     grad_clip = 0.5  # clip gradients at an absolute value of
#     print_freq = 2000
#     running_loss = 0.0

#     outs = []
#     gts = []

#     for imgs, labels in tqdm(Data_loader):
#         batch_size = imgs.shape[0]
#         imgs = imgs.to(device)
#         labels = labels.to(device)

#         if phase == "train":
#             optimizer.zero_grad()
#             model.train()
#             outputs = model(imgs)
#         else:

#             for label in labels.cpu().numpy().tolist():
#                 gts.append(label)

#             model.eval()
#             with torch.no_grad():
#                 outputs = model(imgs)
#                # out = torch.sigmoid(outputs).data.cpu().numpy()
#                # outs.extend(out)
#             # outs = np.array(outs)
#             # gts = np.array(gts)
#         #    evaluation_items(gts, outs)

#         loss = criterion(outputs, labels)

#         if phase == 'train':

#             loss.backward()
#             if grad_clip is not None:
#                 clip_gradient(optimizer, grad_clip)
#             optimizer.step()  # update weights

#         running_loss += loss * batch_size

#         # if i % 500 == 0:
#         #     print(i* batch_size)





#     return running_loss
