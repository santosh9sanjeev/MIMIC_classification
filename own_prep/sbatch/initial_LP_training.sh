#!/bin/bash

#SBATCH --job-name=initial_LP_training            # Job name
#SBATCH --output=/home/santosh.sanjeev/model-soups/my_soups/logs/exp-full-finetuning-pneumoniamnist-AEp_%A_%N.txt   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=65G                   # Total RAM to be used
#SBATCH --cpus-per-task=32          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH -q gpu-8                    # To enable the use of up to 8 GPUs

hostname
cd /home/santosh.sanjeev/model-soups/my_soups/
python initial_LP_training.py --model resnet18 --initialisation 'imagenet' --n_epochs 5 --batch_size 1024 --lr 0.01 --dataset "pneumoniamnist" --task "multi-class"


python initial_LP_training.py --model vit_b_16 --initialisation 'imagenet' --n_epochs 30 --batch_size 256 --lr 0.001 --dataset "cifar10" --task "multi-class" --output_model_path '/home/santosh.sanjeev/model-soups/my_soups/checkpoints/initial_LP/cifar10/LP_v2.
pth' --norm 0.4 --use_pretrained --lp_ft LP --n_classes 10 --device cuda 



python initial_LP_training.py --model vit_b_16 --initialisation 'imagenet' --csv /home/santosh.sanjeev/model-soups/my_soups/rsna_18/csv/final_dataset_wo_not_normal_cases.csv --n_epochs 30 --batch_size 400 --lr 1e-3 --dataset "RSNA" --task "binary-class" --output_model_path '/home/santosh.sanjeev/model-soups/my_soups/checkpoints/initial_LP/rsna/LP_VIT.pth' --norm 0.5 --use_pretrained --lp_ft LP --n_classes 2 --data_dir /home/santosh.sanjeev/model-soups/my_soups/rsna_18/