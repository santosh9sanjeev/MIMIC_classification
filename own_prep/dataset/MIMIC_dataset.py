import pandas as pd
import collections
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import cv2
import torch
import numpy as np
import random
import sys
import warnings
import zipfile
import skimage
from skimage.io import imread
from typing import List, Dict
random.seed(42)

def normalize(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024]."""

    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))

    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024


def apply_transforms(sample, transform, seed=None) -> Dict:
    """Applies transforms to the image and masks.
    The seeds are set so that the transforms that are applied
    to the image are the same that are applied to each mask.
    This way data augmentation will work for segmentation or 
    other tasks which use masks information.
    """

    if seed is None:
        MAX_RAND_VAL = 2147483647
        seed = np.random.randint(MAX_RAND_VAL)

    if transform is not None:
        random.seed(seed)
        torch.random.manual_seed(seed)
        sample["img"] = transform(sample["img"])

        if "pathology_masks" in sample:
            for i in sample["pathology_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["pathology_masks"][i] = transform(sample["pathology_masks"][i])

        if "semantic_masks" in sample:
            for i in sample["semantic_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["semantic_masks"][i] = transform(sample["semantic_masks"][i])

    return sample

class Dataset:
    """The datasets in this library aim to fit a simple interface where the
    imgpath and csvpath are specified. Some datasets require more than one
    metadata file and for some the metadata files are packaged in the library
    so only the imgpath needs to be specified.
    """
    def __init__(self):
        pass

    pathologies: List[str]
    """A list of strings identifying the pathologies contained in this 
    dataset. This list corresponds to the columns of the `.labels` matrix. 
    Although it is called pathologies, the contents do not have to be 
    pathologies and may simply be attributes of the patient. """

    labels: np.ndarray
    """A NumPy array which contains a 1, 0, or NaN for each pathology. Each 
    column is a pathology and each row corresponds to an item in the dataset. 
    A 1 represents that the pathology is present, 0 represents the pathology 
    is absent, and NaN represents no information. """

    csv: pd.DataFrame
    """A Pandas DataFrame of the metadata .csv file that is included with the 
    data. For some datasets multiple metadata files have been merged 
    together. It is largely a "catch-all" for associated data and the 
    referenced publication should explain each field. Each row aligns with 
    the elements of the dataset so indexing using .iloc will work. Alignment 
    between the DataFrame and the dataset items will be maintained when using 
    tools from this library. """

    def totals(self) -> Dict[str, Dict[str, int]]:
        """Compute counts of pathologies.

        Returns: A dict containing pathology name -> (label->value)
        """
        counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
        return dict(zip(self.pathologies, counts))

    # def __repr__(self) -> str:
    #     """Returns the name and a description of the dataset such as:

    #     .. code-block:: python

    #         CheX_Dataset num_samples=191010 views=['PA', 'AP']

    #     If in a jupyter notebook it will also print the counts of the
    #     pathology counts returned by .totals()

    #     .. code-block:: python

    #         {'Atelectasis': {0.0: 17621, 1.0: 29718},
    #          'Cardiomegaly': {0.0: 22645, 1.0: 23384},
    #          'Consolidation': {0.0: 30463, 1.0: 12982},
    #          ...}

    #     """
    #     if xrv.utils.in_notebook():
    #         pprint.pprint(self.totals())
    #     return self.string()

    def check_paths_exist(self):
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")

    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the
        images by view based on the values in .csv['view']
        """
        if type(views) is not list:
            views = [views]
        if '*' in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views

        # missing data is unknown
        self.csv.view.fillna("UNKNOWN", inplace=True)

        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view

class MIMIC_Dataset(Dataset):
    """MIMIC-CXR Dataset

    Citation:

    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY,
    Mark RG, Horng S. MIMIC-CXR: A large publicly available database of
    labeled chest radiographs. arXiv preprint arXiv:1901.07042. 2019 Jan 21.

    https://arxiv.org/abs/1901.07042

    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """

    def __init__(self,
                 imgpath,
                 csvpath,
                 metacsvpath,
                 splitpath,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 seed=0,
                 split = 'train',
                 unique_patients=True
                 ):

        super(MIMIC_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.pathologies = ["Enlarged Cardiomediastinum",
                            "Cardiomegaly",
                            "Lung Opacity",
                            "Lung Lesion",
                            "Edema",
                            "Consolidation",
                            "Pneumonia",
                            "Atelectasis",
                            "Pneumothorax",
                            "Pleural Effusion",
                            "Pleural Other",
                            "Fracture",
                            "Support Devices"]

        self.pathologies = sorted(self.pathologies)

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.metacsvpath = metacsvpath
        self.splitpath = splitpath
        
        self.metacsv = pd.read_csv(self.metacsvpath)
        # print('metaaaaaaaaaaaaaaa',self.metacsv)
        self.split_dataset = pd.read_csv(self.splitpath)
        test_df = self.split_dataset[(self.split_dataset['split'] == split)]
        test_df.reset_index(drop=True, inplace=True)
        # print('testttttttttttttttt',test_df)

        final_df = pd.merge(test_df, self.metacsv, on=['dicom_id', 'subject_id', 'study_id'], how='inner')
        final_df = final_df[self.metacsv.columns]

        self.csv = self.csv.set_index(['subject_id', 'study_id'])
        final_df = final_df.set_index(['subject_id', 'study_id'])

        self.csv = self.csv.join(final_df, how='inner').reset_index()
        # Keep only the desired view
        self.csv["view"] = self.csv["ViewPosition"]
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("subject_id").first().reset_index()

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]

            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        self.labels[self.labels == -1] = np.nan
        print(self.labels.shape)
        self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))


        # offset_day_int
        self.csv["offset_day_int"] = self.csv["StudyDate"]

        # patientid
        self.csv["patientid"] = self.csv["subject_id"].astype(str)
        print('final df', self.csv)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid = str(self.csv.iloc[idx]["study_id"])
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])

        img_path = os.path.join(self.imgpath, "p" + subjectid[:2], "p" + subjectid, "s" + studyid, dicom_id + ".jpg")
        # img_path = os.path.join(self.imgpath, dicom_id + '.jpg' + '_' + 'p' + subjectid[:2] + '_' + 'p' + subjectid + '_' + 's' + studyid + '_' + 'GT_img1' + '.jpeg')
        # print(img_path)
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample



class XRayResizer(object):
    """Resize an image to a specific size"""
    def __init__(self, size: int, engine="skimage"):
        self.size = size
        self.engine = engine
        if 'cv2' in sys.modules:
            print("Setting XRayResizer engine to cv2 could increase performance.")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.engine == "skimage":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return skimage.transform.resize(img, (1, self.size, self.size), mode='constant', preserve_range=True).astype(np.float32)
        elif self.engine == "cv2":
            return cv2.resize(img[0, :, :],
                              (self.size, self.size),
                              interpolation=cv2.INTER_AREA
                              ).reshape(1, self.size, self.size).astype(np.float32)
        else:
            raise Exception("Unknown engine, Must be skimage (default) or cv2.")


class XRayCenterCrop(object):
    """Perform a center crop on the long dimension of the input image"""
    def crop_center(self, img: np.ndarray) -> np.ndarray:
        _, y, x = img.shape
        crop_size = np.min([y, x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty:starty + crop_size, startx:startx + crop_size]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.crop_center(img)
