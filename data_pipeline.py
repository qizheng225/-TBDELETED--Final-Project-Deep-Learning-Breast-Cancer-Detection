#EVERYTHING RELATED TO TURNING THE TWO CBIS-DDSM CSV FILES INTO TRAIN/VAL/TEST tf.data.Dataset OBJECTS, 
#HELPERS FOR HANDLING CLASS IMBALANCE (OVERSAMPLING AT THE DATAFRAME LEVEL, OR CLASS WEIGHTS).
#ADDITIONALLY, ALL FUNCTIONS TAKE PLAIN ARGUMENTS SO EACH ONE IS UNIT TESTABLE IN ISOLATION

from typing import Optional, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


#CSV LOADING / MERGING
def load_raw_csvs(mass_csv_path: str, dicom_csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #loads the 2 raw CBIS-DDSM CSV
    mass_df = pd.read_csv(mass_csv_path)
    dicom_df = pd.read_csv(dicom_csv_path)
    return mass_df, dicom_df


def add_binary_label(mass_df: pd.DataFrame, pathology_col: str = "pathology") -> pd.DataFrame:
    #creating binary labels (1=malignant, 0=everything else)
    mass_df = mass_df.copy()
    mass_df["label"] = (mass_df[pathology_col] == "MALIGNANT").astype(int)
    return mass_df


def build_dataset_df(
        mass_df: pd.DataFrame,
        dicom_df: pd.DataFrame,
        jpeg_prefix_old: str = "CBIS-DDSM/jpeg/",
        jpeg_prefix_new: str = "cbis-ddsm/jpeg/",
    ) -> pd.DataFrame:
    mass_df = add_binary_label(mass_df)

    #selecting (keeping only) cropped lesion images from dicom metadata
    cropped_df = dicom_df[dicom_df["SeriesDescription"] == "cropped images"].copy()

    #extract UID from dicom metadata
    cropped_df["uid"] = cropped_df["image_path"].str.split("/").str[2]

    #extract UID from mass csv
    mass_df = mass_df.copy()
    mass_df["uid"] = mass_df["cropped image file path"].str.split("/").str[2]

    #merge pathology labels with image paths (inner join on UID)
    dataset_df = mass_df.merge(cropped_df[["uid", "image_path"]], on="uid", how="inner")

    #convert image paths to local directory structure (point at local jpeg directory)
    dataset_df["real_path"] = dataset_df["image_path"].str.replace(
        jpeg_prefix_old, jpeg_prefix_new, regex=False
    )
    return dataset_df


#SPLITTING
def split_dataset (
        dataset_df: pd.DataFrame,
        label_col: str = "label",
        holdout_size: float = 0.30,
        test_size_of_holdout: float = 0.50,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #stratified train/val/test split
    train_df, temp_df = train_test_split(
        dataset_df,
        test_size=holdout_size,
        stratify=dataset_df[label_col],
        random_state=random_state,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size_of_holdout,
        stratify=temp_df[label_col],
        random_state=random_state,
    )
    return train_df, val_df, test_df


#CREATING TENSORFLOW DATASET
def load_image(path: tf.Tensor, label: tf.Tensor, img_size: int = 224):
    # read, decode, and resize to a single jpeg (this function helps with unit testing) 
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32)
    return image, label


def make_tf_dataset(
        df: pd.DataFrame,
        img_size: int = 224,
        batch_size: int = 16,
        shuffle: bool = False,
        shuffle_buffer: int = 1000,
        path_col: str = "real_path",
        label_col: str = "label",
        cache: bool = True,
    ) -> tf.data.Dataset:
    #builds a batched, prefetched tf.data.Dataset from a dataframe of paths/labels
    ds = tf.data.Dataset.from_tensor_slices((df[path_col].values, df[label_col].values))
    ds = ds.map(
        lambda p, l: load_image(p, l, img_size), num_parallel_calls=tf.data.AUTOTUNE
    )
    #caches the decoded/resized images after the first epoch, so subsequent epochs skip disk read and jpeg decode to save time
    #safe here as data augmentation lives inside the model (models.build_model), so it will not reduce augmentation randomness
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
        
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


#CLASS IMBALANCE HELPERS (DATAFRAME / LABEL LEVEL)
def get_class_weights(labels: np.ndarray) -> dict:
    #use sklearn "balanced" class weights {0: w0, 1: w1}, passing into model.fit() later
    labels = np.asarray(labels)
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def oversample_minority_class(
    df: pd.DataFrame, label_col: str = "label", random_state: int = 42
) -> pd.DataFrame:
    #random oversampling of minority class up to parity with the majority class (alternative to class weighting for comparison)
    counts = df[label_col].value_counts()
    majority_label = counts.idxmax()
    majority_count = counts.max()

    frames = [df[df[label_col] == majority_label]]
    for label, count in counts.items():
        if label == majority_label:
            continue
        minority_df = df[df[label_col] == label]
        resampled = minority_df.sample(
            n=majority_count, replace=True, random_state=random_state
        )
        frames.append(resampled)

    #shuffle
    balanced_df = pd.concat(frames, axis=0).sample(
        frac=1.0, random_state=random_state
    )
    return balanced_df.reset_index(drop=True)