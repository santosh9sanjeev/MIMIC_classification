from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import clip
import argparse
import PIL

from dataset.MIMIC_dataset import RSNADataset
from models.model import ImageNetModel, CLIPModel
from models.model import ResNet18
import medmnist
from medmnist import INFO, Evaluator

import torchvision.models as models
from sklearn.metrics import f1_score, roc_auc_score
import sys
import random
from timm.models import vit_base_patch16_224
from torchvision import datasets, transforms


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model', default='resnet18', type=str, help='pretrained model')
    parser.add_argument('--task', default='binary-class', type=str, help='available things {multi-label, multi-class, binary-class}')
    parser.add_argument('--csv', default='/home/santosh.sanjeev/model-soups/my_soups/metadata/RSNA_final_df.csv', type=str, help='Data directory')
    parser.add_argument('--data_dir', default='/home/santosh.sanjeev/rsna_18/train/', type=str, help='csv file containing the stats')
    parser.add_argument('--n_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--initialisation', default='imagenet', type=str, help='weight initialisation')
    parser.add_argument('--dataset', default='RSNA', type=str, help='which dataset')
    parser.add_argument('--system', default='hpc', type=str, help='which system you are using')
    parser.add_argument('--load_model_weights', default='/home/santosh.sanjeev/soups/my_soups/checkpoints/full_finetuning/pneumoniamnist/initial_full_finetuning_model.pth', type=str, help='which system you are using')
    parser.add_argument('--use_pretrained', action='store_true')

    parser.add_argument('--norm', default=0.5, type=float, help='which norm')

    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)
    

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set seed for PyTorch
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.dataset[-5:] == 'mnist':
        data_flag = args.dataset
        download = True
        info = INFO[data_flag]
        task = info['task']            
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
    else:
        n_classes = args.n_classes
    BATCH_SIZE = args.batch_size
    load_model = args.model
    if args.norm!=0.5:
        print('USING IMAGENET NORM')
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
    else:
        print('NOT USING IMAGENET NORM')
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize = transforms.Normalize(mean=mean, std=std)

    # preprocessing
    data_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
        transforms.ToTensor(),
        normalize
    ])
    if args.dataset == 'cifar10':
        print('CIFARRRRRRRRRRRRRRRRR')
        # # Define transformations to be applied to the images
        # transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        #     transforms.RandomCrop(32, padding=4),  # Randomly crop the image
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
        # ])

        # Load CIFAR-10 dataset
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=data_transform, download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=data_transform, download=True)

        # Split the training dataset into training and validation sets
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    elif args.dataset == 'RSNA':
        test_dataset = RSNADataset(csv_file = args.csv, data_folder = args.data_dir, split='test', transform=data_transform)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=True)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=True)
        val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=True)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)




    print(args.dataset)
    print(len(test_dataset))
    if args.initialisation == 'clip':
        base_model, preprocess = clip.load(load_model, 'cuda', jit=False)
        full_model = CLIPModel(base_model, num_classes=n_classes)
    elif args.initialisation == 'imagenet':
        if args.use_pretrained:
            print('USING PRETRAINED WEIGHTS')
            if args.model == 'resnet18':
                model =  getattr(models, load_model)(pretrained=True)
            elif args.model=='vit_b_16':
                print('in thisssssssssss')
                #image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
                # model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
                # print(model)
                # model_name = 'vit_base_patch16_224'  # Example model, you can use other available ViT models
                full_model = vit_base_patch16_224(pretrained=True)
                print(full_model)
                # model =  getattr(models, load_model)(pretrained=True)
        else:
            model =  getattr(models, load_model)(pretrained=False)
        # model = torch.nn.Sequential(*list(model.children())[:-1])
        # print('modellllll after', model)
        # full_model = ViTModel(model, in_channels = 768, num_classes=n_classes)
        # print('finallll modellll', full_model)
    full_model.head = nn.Linear(full_model.head.in_features, 10)
    print(full_model)


    # if args.initialisation == 'clip':
    #     base_model, preprocess = clip.load(load_model, 'cuda', jit=False)
    #     full_model = CLIPModel(base_model, num_classes=n_classes)
    # elif args.initialisation == 'imagenet':
    #     model =  getattr(models, load_model)(pretrained=False)
    #     model = torch.nn.Sequential(*list(model.children())[:-1])
    #     full_model = ImageNetModel(model, num_classes=n_classes)
    
    


    checkpoint = torch.load(args.load_model_weights)  # Provide the path to your weights file
    full_model.load_state_dict(checkpoint['model_state_dict'])
    
    
    # full_model = getattr(models, load_model)(pretrained=False, num_classes=2)#ResNet18(in_channels=3, num_classes=3)
    # full_model.load_state_dict(torch.load(args.load_model_weights, map_location='cuda')['net'], strict=True)

    # weights = [(name, param.data) for name, param in full_model.named_parameters()]
    # _ = [print(f"Layer: {name}\n  - Shape: {weight.shape}\n  - Values:\n{weight}\n") for name, weight in weights[:5]]

    test_correct = 0
    test_total = 0
    test_predictions = []
    test_targets = []
    test_f1_predictions = []
    device = 'cuda'
    full_model.to(device)

    full_model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = full_model(inputs)
            m = nn.Softmax(dim=1)            
            predicted = m(outputs)
            # _, predicted = torch.max(outputs.data, 1)
            targets = targets.squeeze().long()
            test_total += targets.size(0)
            _,t = torch.max(outputs.data, 1)
            test_correct += (t == targets).sum().item()
            # print(predicted)
            if args.task == 'binary-class':
                predicted = predicted[:,-1]

            test_predictions.extend(predicted.detach().cpu().numpy())
            test_f1_predictions.extend(t.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
            # print(test_correct, test_total)
        test_accuracy = 100 * test_correct / test_total
        if args.task !='binary-class':
            test_f1 = f1_score(test_targets, test_f1_predictions, average = 'macro')
            test_auc = roc_auc_score(test_targets, test_predictions, average = 'macro', multi_class = 'ovr')
        else:
            test_f1 = f1_score(test_targets, test_f1_predictions)
            test_auc = roc_auc_score(test_targets, test_predictions)
        # Print or log validation metrics
        print(f"Test Accuracy: {test_accuracy:.2f}%, F1-Score: {test_f1:.4f}, AUC: {test_auc:.4f}")



    # train_evaluator = medmnist.Evaluator(data_flag, 'train')
    # val_evaluator = medmnist.Evaluator(data_flag, 'val')
    # test_evaluator = medmnist.Evaluator(data_flag, 'test')


    # criterion = nn.CrossEntropyLoss()
    # run = args.run

    # def test(model, evaluator, data_loader, task, criterion, device, run, save_folder=None):

    #     model.eval()
        
    #     total_loss = []
    #     y_score = torch.tensor([]).to(device)

    #     with torch.no_grad():
    #         for batch_idx, (inputs, targets) in enumerate(data_loader):
    #             outputs = model(inputs.to(device))
                
    #             if task == 'multi-label, binary-class':
    #                 targets = targets.to(torch.float32).to(device)
    #                 loss = criterion(outputs, targets)
    #                 m = nn.Sigmoid()
    #                 outputs = m(outputs).to(device)
    #             else:
    #                 targets = torch.squeeze(targets, 1).long().to(device)
    #                 loss = criterion(outputs, targets)
    #                 m = nn.Softmax(dim=1)
    #                 outputs = m(outputs).to(device)
    #                 targets = targets.float().resize_(len(targets), 1)
    #             total_loss.append(loss.item())
    #             y_score = torch.cat((y_score, outputs), 0)

    #         y_score = y_score.detach().cpu().numpy()
    #         auc, acc = evaluator.evaluate(y_score, save_folder, run)
            
    #         test_loss = sum(total_loss) / len(total_loss)

    #         return [test_loss, auc, acc]


    # # train_metrics = test(full_model, train_evaluator, train_loader, task, criterion, 'cpu', run, '/home/santosh.sanjeev/soups/my_soups/checkpoints/full_finetuning/pneumoniamnist/')
    # val_metrics = test(full_model, val_evaluator, val_loader, task, criterion, 'cpu', run, '/home/santosh.sanjeev/soups/my_soups/checkpoints/full_finetuning/pneumoniamnist/')
    # test_metrics = test(full_model, test_evaluator, test_loader, task, criterion, 'cpu', run, '/home/santosh.sanjeev/soups/my_soups/checkpoints/full_finetuning/pneumoniamnist/')

    # # train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    # val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    # test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])
    # log = '%s\n' % (data_flag) + val_log + test_log
    # print(log)