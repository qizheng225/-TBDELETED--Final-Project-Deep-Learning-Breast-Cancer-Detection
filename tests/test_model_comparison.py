from data_pipeline import make_tf_dataset
from model_comparison import best_backbone, compare_architectures


def test_compare_architectures_returns_one_row_per_backbone(tiny_config, synthetic_dataset_df):
    tiny_config.backbones = ["EfficientNetB0", "DenseNet121"]  # keep it fast
    train_ds = make_tf_dataset(synthetic_dataset_df, tiny_config.img_size, tiny_config.batch_size, shuffle=True)
    val_ds = make_tf_dataset(synthetic_dataset_df, tiny_config.img_size, tiny_config.batch_size)
    test_ds = make_tf_dataset(synthetic_dataset_df, tiny_config.img_size, tiny_config.batch_size)

    result = compare_architectures(
        train_ds,
        val_ds,
        test_ds,
        synthetic_dataset_df["label"].values,
        synthetic_dataset_df["label"].values,
        tiny_config,
    )

    assert len(result["table"]) == 2
    assert set(result["table"]["backbone"]) == {"EfficientNetB0", "DenseNet121"}
    assert set(result["models"].keys()) == {"EfficientNetB0", "DenseNet121"}
    for col in ["accuracy", "precision_malignant", "recall_malignant", "f1_malignant", "roc_auc"]:
        assert col in result["table"].columns


def test_compare_architectures_table_sorted_by_auc_descending(tiny_config, synthetic_dataset_df):
    tiny_config.backbones = ["EfficientNetB0", "DenseNet121"]
    train_ds = make_tf_dataset(synthetic_dataset_df, tiny_config.img_size, tiny_config.batch_size, shuffle=True)
    val_ds = make_tf_dataset(synthetic_dataset_df, tiny_config.img_size, tiny_config.batch_size)
    test_ds = make_tf_dataset(synthetic_dataset_df, tiny_config.img_size, tiny_config.batch_size)

    result = compare_architectures(
        train_ds,
        val_ds,
        test_ds,
        synthetic_dataset_df["label"].values,
        synthetic_dataset_df["label"].values,
        tiny_config,
    )
    aucs = result["table"]["roc_auc"].tolist()
    assert aucs == sorted(aucs, reverse=True) or all(
        (a == b or (a != a) or (b != b)) for a, b in zip(aucs, sorted(aucs, reverse=True))
    )


def test_best_backbone_picks_top_row_of_table():
    import pandas as pd

    fake_result = {
        "table": pd.DataFrame(
            {"backbone": ["ResNet50", "EfficientNetB0"], "roc_auc": [0.6, 0.9]}
        ).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    }
    assert best_backbone(fake_result) == "EfficientNetB0"