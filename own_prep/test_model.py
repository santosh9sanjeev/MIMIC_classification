import torch
import glob
import os

weights_files = '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/MIMIC_classification/own_prep/checkpoints/MIMIC-densenet-e21.pt'#glob(join(args.output_dir, f'{dataset_name}-e*.pt'))  # Find all weights files
if len(weights_files):
    # Find most recent epoch
    epochs = np.array(
        [int(w[len(join(args.output_dir, f'{dataset_name}-e')):-len('.pt')].split('-')[0]) for w in weights_files])
    start_epoch = epochs.max()
    weights_file = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()][0]
    model.load_state_dict(torch.load(weights_file).state_dict())
