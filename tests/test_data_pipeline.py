import numpy as np
import pandas as pd
import pytest

from data_pipeline import (
    add_binary_label,
    build_dataset_df,
    get_class_weights,
    load_image,
    make_tf_dataset,
    oversample_minority_class,
    split_dataset,
)


def test_add_binary_label_marks_only_malignant_as_1():
    df = pd.DataFrame({"pathology": ["MALIGNANT", "BENIGN", "BENIGN_WITHOUT_CALLBACK"]})
    out = add_binary_label(df)
    assert list(out["label"]) == [1, 0, 0]


def test_build_dataset_df_merges_and_filters_cropped_only(synthetic_mass_and_dicom_df):
    mass_df, dicom_df = synthetic_mass_and_dicom_df
    result = build_dataset_df(mass_df, dicom_df)

    #UID_C's DICOM ROW IS "full mammogram images", NOT "cropped images",
    #SO IT SHOULD NEVER MAKE IT INTO THE MERGE
    assert "UID_C" not in result["uid"].values
    assert len(result) == 3
    assert set(result["label"]) == {0, 1}


def test_build_dataset_df_rewrites_jpeg_prefix(synthetic_mass_and_dicom_df):
    mass_df, dicom_df = synthetic_mass_and_dicom_df
    result = build_dataset_df(mass_df, dicom_df)
    assert all(p.startswith("cbis-ddsm/jpeg/") for p in result["real_path"])
    assert not any(p.startswith("CBIS-DDSM/jpeg/") for p in result["real_path"])


def test_split_dataset_produces_expected_sizes_and_no_overlap(synthetic_dataset_df):
    train_df, val_df, test_df = split_dataset(
        synthetic_dataset_df, holdout_size=0.5, test_size_of_holdout=0.5, random_state=0
    )
    n = len(synthetic_dataset_df)
    assert len(train_df) + len(val_df) + len(test_df) == n

    train_idx, val_idx, test_idx = set(train_df.index), set(val_df.index), set(test_df.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_split_dataset_is_stratified(synthetic_dataset_df):
    train_df, val_df, test_df = split_dataset(
        synthetic_dataset_df, holdout_size=0.5, test_size_of_holdout=0.5, random_state=0
    )
    overall_rate = synthetic_dataset_df["label"].mean()
    for part in (train_df, val_df, test_df):
        assert abs(part["label"].mean() - overall_rate) <= 0.5  # loose bound for tiny n


def test_load_image_returns_correct_shape_and_dtype(fake_jpeg_dir):
    import tensorflow as tf

    path = tf.constant(fake_jpeg_dir[0])
    label = tf.constant(1)
    image, out_label = load_image(path, label, img_size=32)

    assert image.shape == (32, 32, 3)
    assert image.dtype == tf.float32
    assert int(out_label.numpy()) == 1


def test_make_tf_dataset_batches_correctly(synthetic_dataset_df):
    ds = make_tf_dataset(synthetic_dataset_df, img_size=32, batch_size=4)
    batch_images, batch_labels = next(iter(ds))
    assert batch_images.shape[1:] == (32, 32, 3)
    assert batch_images.shape[0] == batch_labels.shape[0]
    assert batch_images.shape[0] <= 4


def test_get_class_weights_balances_minority_class():
    labels = np.array([0, 0, 0, 0, 1])  # 4 vs 1 -> imbalanced
    weights = get_class_weights(labels)
    assert weights[1] > weights[0]


def test_get_class_weights_equal_for_balanced_labels():
    labels = np.array([0, 1, 0, 1])
    weights = get_class_weights(labels)
    assert weights[0] == pytest.approx(weights[1])


def test_oversample_minority_class_balances_counts():
    df = pd.DataFrame({"label": [0] * 8 + [1] * 2, "real_path": [f"p{i}" for i in range(10)]})
    balanced = oversample_minority_class(df, random_state=0)
    counts = balanced["label"].value_counts()
    assert counts[0] == counts[1]
    assert counts[0] == 8