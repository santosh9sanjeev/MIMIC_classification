import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import shutil
# Define the transformation to resize images to 512x512
transform = transforms.Compose([transforms.Resize((512, 512)), transforms.PILToTensor()])

# Define the input and output directories
input_dir = '/share/ssddata/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
# output_dir = '/share/ssddata/mimic_pt'

# Walk through the directory structure
for root, dirs, files in os.walk(input_dir):
    # Create the corresponding directory structure in the output directory
    relative_path = os.path.relpath(root, input_dir)
    # output_subdir = os.path.join(output_dir, relative_path)
    # os.makedirs(output_subdir, exist_ok=True)
    print(root)
    # print(dirs)
    count = 0
    # Process each file
    for file in files:
        count+=1
        print(count)

        # input_    path = os.path.join(root, file)
        # output_path = os.path.join(output_subdir, file)

        # if os.path.isfile(input_path):
        #     if file.lower().endswith('.jpg'):
        #         # Load and transform the image
        #         img = Image.open(input_path)
        #         img = transform(img)

        #         # Save as .pt file
        #         output_path = os.path.join(output_subdir, os.path.splitext(file)[0] + '.pt')
        #         torch.save(img, output_path)
        #         print(f'Saved {output_path}')

# print('Conversion completed.')
