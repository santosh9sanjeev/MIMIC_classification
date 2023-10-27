# import os
# import random
# from PIL import Image

# # Define the root directory
# root_dir = '/share/ssddata/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'

# # Define the directory where you want to save the images
# output_dir = '/home/santoshsanjeev/MIMIC_classification'

# # Define the number of random samples you want to select
# num_samples = 10

# # List to store paths of randomly selected images
# selected_images = []

# # Walk through the directory structure
# for root, dirs, files in os.walk(root_dir):
#     for file in files:
#         if file.lower().endswith('.jpg'):
#             selected_images.append(os.path.join(root, file))
#     if len(selected_images) >= num_samples:
#         break

# # Randomly select 10 images
# sample_images = random.sample(selected_images, num_samples)

# # Create the output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# # Copy the selected images to the output directory
# for img_path in sample_images:
#     img_name = os.path.basename(img_path)
#     output_path = os.path.join(output_dir, img_name)
#     with open(img_path, 'rb') as source:
#         with open(output_path, 'wb') as target:
#             target.write(source.read())

# print(f'Saved {num_samples} images to {output_dir}')


import torch
from matplotlib import pyplot as plt
from PIL import Image

# Load the torch file
loaded_data = torch.load('/share/ssddata/mimic_pt/p13/p13975682/s51646043/d898444a-8a28260e-48b5c50b-fad4ca9f-29d68a99.pt')  # Replace with the actual file path
print(loaded_data, loaded_data.shape, torch.max(loaded_data), torch.min(loaded_data), loaded_data.dtype)
# Assuming the torch file contains an image tensor
# image_tensor = loaded_data['image']

# Convert the tensor to a numpy array
# image_array = loaded_data.numpy()

# Step 3: Save the Image
output_image_path = 'output_image.png'
Image.fromarray((loaded_data[0]).numpy().astype('uint8')).save(output_image_path)

print(f"Image saved at {output_image_path}")