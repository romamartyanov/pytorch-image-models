from os import tcgetpgrp
import pandas as pd
import numpy as np


import os
from glob import glob
from shutil import copyfile

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def generate_meta(data_dir):
    """
    Generate meta dataframe.
    """
    img_path_mapping = {
        os.path.basename(path): path
        for path in glob(f"{data_dir}/train_images/*.jpg")
    }

    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    df["path"] = df["image_id"].map(img_path_mapping)
    df["label"] = df["label"].astype("category")
    # df["target"] = df["dx"].cat.codes

    return df


def stratified_split(df: pd.DataFrame, train_frac: float = 0.8, seed=42):
    """
    Using stratified split to generate training, validation and testing ids.
    Arg:
        df (pd.DataFrame): a pandas dataframe with ['image_id', 'target'] columns.
        train_frac (float): fraction of data to use for training.
    Return:
        train_ids (np.ndarray): ids for training.
        valid_ids (np.ndarray): ids for validation.
        test_ids (np.ndarray): ids for testing.
    """
    sss = StratifiedShuffleSplit(train_size=train_frac, random_state=seed)
    ids = df["image_id"].values
    train_idx, test_idx = next(sss.split(ids, df["label"].values))
    train_ids, test_ids = ids[train_idx], ids[test_idx]

    return train_ids, test_ids


def over_sample(df: pd.DataFrame, sample_prop, frac=0.3, imbalance=True):
    """
    If the distribution of target is imbalanced, do oversample
    Args:
        df (pd.DataFrame): training dataframe.
        sample_prop (float): oversample the least class to the proportion of largest class.
        frac (float): between 0 and 1, to reduce some data.
        imbalance (bool): to do oversampling or not.
    """
    if imbalance:
        weights = df["label"].value_counts()
        weights = (weights.max() / weights * sample_prop).apply(int).sort_index().to_list()
        for cat in range(len(weights)):
            df = pd.concat([df] + [df.loc[df["label"] == cat]] * weights[cat], axis=0)
        df = df.sample(frac=frac).reset_index(drop=True)

    return df


def move_images(df: pd.DataFrame, dataset_dir: str, output_dir: str):
    classes = ['0', '1', '2', '3', '4']
    for _class in classes:
        if not os.path.exists(os.path.join(output_dir, _class)):
            os.makedirs(os.path.join(output_dir, _class))

    for index, row in df.iterrows():
        _class = str(row['label'])
        image_path = row['path']
        filename = os.path.basename(image_path)

        copyfile(os.path.join(dataset_dir, 'train_images') + f'/{filename}',
                 os.path.join(output_dir, _class) + f'/{filename}')


dataset_path = 'dataset'

df = generate_meta(dataset_path)
train_ids, valid_ids = stratified_split(df, train_frac=0.95)
# n_classes = df["label"].max() + 1

train_df = df.loc[df["image_id"].isin(train_ids)]
# train_df = over_sample(train_df, 0.15, frac=1.0)
valid_df = df.loc[df["image_id"].isin(valid_ids)]

t = train_df.groupby('label')['image_id'].nunique()
print(t)
move_images(train_df, dataset_path, os.path.join(dataset_path, 'train'))
print('moved: train')
move_images(valid_df, dataset_path, os.path.join(dataset_path, 'val'))
print('moved: val')
