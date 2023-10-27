import torch
from train import *
from LearningCurve import *
from predictions import *
# from TPR_Disparity import *
import pandas as pd

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#----------------------------- q
path_image = "/share/ssddata/mimic_pt/"


df_path ="/share/ssddata/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv"
dataset_split_path = "/share/ssddata/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv"
metadata_path = "/share/ssddata/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv"
output_folder = '/home/santoshsanjeev/MIMIC_classification/CheXclusion/MIMIC/results_v2'

if not os.path.exists(output_folder):
   os.makedirs(output_folder)
   
   
# we use MIMIC original validation dataset as our new test dataset and the new_test.csv as out validation dataset

diseases = ['Lung Opacity', 'Atelectasis', 'Cardiomegaly',
       'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
       'Lung Lesion', 'No Finding', 'Pleural Effusion', 'Pleural Other',
       'Pneumonia', 'Pneumothorax', 'Support Devices']
# diseases = ['Airspace Opacity', 'Atelectasis', 'Cardiomegaly',
#        'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
#        'Lung Lesion', 'Pleural Effusion', 'Pleural Other',
#        'Pneumonia', 'Pneumothorax', 'Support Devices']
age_decile = ['60-80', '40-60', '20-40', '80-', '0-20']

gender = ['M', 'F']
race = ['WHITE', 'BLACK/AFRICAN AMERICAN',
        'HISPANIC/LATINO', 'OTHER', 'ASIAN',
        'AMERICAN INDIAN/ALASKA NATIVE']
# race = ['WHITE', 'BLACK/AFRICAN AMERICAN', 'ASIAN',
#         'AMERICAN INDIAN/ALASKA NATIVE']

insurance = ['Medicare', 'Other', 'Medicaid']


def preprocess_csv(path, split_path, metacsvpath, split):
    split_df = pd.read_csv(split_path)
    df = pd.read_csv(path)
    metacsv = pd.read_csv(metacsvpath)
    test_df = split_df[(split_df['split'] == split)]

    test_df.reset_index(drop=True, inplace=True)

    final_df = pd.merge(test_df, metacsv, on=['dicom_id', 'subject_id', 'study_id'], how='inner')
    final_df = final_df[metacsv.columns]

    df = df.set_index(['subject_id', 'study_id'])
    final_df = final_df.set_index(['subject_id', 'study_id'])

    df = df.join(final_df, how='inner').reset_index()

    # Keep only the desired view
    df = df[df['ViewPosition'].isin(['PA','AP'])]
    df = df.drop(columns=['PerformedProcedureStepDescription', 'ViewPosition', 
                     'Rows', 'Columns', 'StudyDate', 'StudyTime', 'ProcedureCodeSequence_CodeMeaning', 
                     'ViewCodeSequence_CodeMeaning', 'PatientOrientationCodeSequence_CodeMeaning'])
    # print(df.columns, df['dicom_id'].nunique())
    return df


def main():

    MODE = "train"  # Select "train" or "test", "Resume", "plot", "Threshold"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_df = preprocess_csv(df_path, dataset_split_path, metadata_path, 'train')
    train_df_size = len(train_df)
    # print(train_df)
    print("Train_df size:",train_df_size)

    val_df = preprocess_csv(df_path, dataset_split_path, metadata_path, 'validate')
    val_df_size = len(val_df)
    # print(val_df)
    print("val_df size:",val_df_size)

    test_df = preprocess_csv(df_path, dataset_split_path, metadata_path, 'test')
    test_df_size = len(test_df)
    # print(test_df)
    print("Test_df size:",test_df_size)

    if MODE == "train":
        ModelType = "densenet"  # select 'ResNet50','densenet','ResNet34', 'ResNet18'
        CriterionType = 'BCELoss'
        LR = 0.5e-3

        model, best_epoch = ModelTrain(train_df, val_df, path_image, ModelType, CriterionType, device,LR, output_folder)
        print('DONEEEEEEEEEEEEEEEEEEEEE')
        PlotLearnignCurve()


    if MODE =="test":

        CheckPointData = torch.load('results_v2/checkpoint.pth')
        model = CheckPointData['model']

        make_pred_multilabel(model, test_df, val_df, path_image, device)


    if MODE == "Resume":
        ModelType = "Resume"  # select 'ResNet50','densenet','ResNet34', 'ResNet18'
        CriterionType = 'BCELoss'
        LR = 0.5e-3

        model, best_epoch = ModelTrain(train_df, val_df, path_image, ModelType, CriterionType, device,LR)

        PlotLearnignCurve()

    if MODE == "plot":
        TrueWithMeta = pd.read_csv("./True_withMeta.csv")
        pred = pd.read_csv("./results_v2/bipred.csv")
        factor = [gender, age_decile, race, insurance]
        factor_str = ['gender', 'age_decile', 'race', 'insurance']



        # plot()
        for i in range(len(factor)):
            
            #plot_frequency(gt, diseases, factor[i], factor_str[i])
            
            TPR_Disparities(TrueWithMeta, pred, diseases, factor[i], factor_str[i])
           

        
if __name__ == "__main__":
    main()
