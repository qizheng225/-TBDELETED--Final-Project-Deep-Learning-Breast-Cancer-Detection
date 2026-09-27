import pytest

from data_pipeline import make_tf_dataset
from train import build_callbacks, train_frozen_model


def test_build_callbacks_includes_early_stopping_and_reduce_lr(tiny_config):
    callbacks = build_callbacks(tiny_config)
    names = [type(c).__name__ for c in callbacks]
    assert "EarlyStopping" in names
    assert "ReduceLROnPlateau" in names


def test_build_callbacks_includes_checkpoint_when_path_given(tiny_config, tmp_path):
    ckpt = str(tmp_path / "model.keras")
    callbacks = build_callbacks(tiny_config, checkpoint_path=ckpt)
    names = [type(c).__name__ for c in callbacks]
    assert "ModelCheckpoint" in names


def test_train_frozen_model_with_class_weight_strategy_runs(tiny_config, synthetic_dataset_df):
    tiny_config.imbalance_strategy = "class_weight"
    train_ds = make_tf_dataset(synthetic_dataset_df, tiny_config.img_size, tiny_config.batch_size, shuffle=True)
    val_ds = make_tf_dataset(synthetic_dataset_df, tiny_config.img_size, tiny_config.batch_size)

    model, history = train_frozen_model(
        "EfficientNetB0",
        train_ds,
        val_ds,
        synthetic_dataset_df["label"].values,
        tiny_config,
        epochs=1,
    )
    assert "loss" in history.history
    assert model.output_shape == (None, 1)


def test_train_frozen_model_with_focal_loss_strategy_runs(tiny_config, synthetic_dataset_df):
    tiny_config.imbalance_strategy = "focal_loss"
    train_ds = make_tf_dataset(synthetic_dataset_df, tiny_config.img_size, tiny_config.batch_size, shuffle=True)
    val_ds = make_tf_dataset(synthetic_dataset_df, tiny_config.img_size, tiny_config.batch_size)

    model, history = train_frozen_model(
        "EfficientNetB0",
        train_ds,
        val_ds,
        synthetic_dataset_df["label"].values,
        tiny_config,
        epochs=1,
    )
    assert "loss" in history.history


def test_train_frozen_model_rejects_unknown_imbalance_strategy(tiny_config, synthetic_dataset_df):
    tiny_config.imbalance_strategy = "not_a_real_strategy"
    train_ds = make_tf_dataset(synthetic_dataset_df, tiny_config.img_size, tiny_config.batch_size)
    val_ds = make_tf_dataset(synthetic_dataset_df, tiny_config.img_size, tiny_config.batch_size)

    with pytest.raises(ValueError):
        train_frozen_model(
            "EfficientNetB0", train_ds, val_ds, synthetic_dataset_df["label"].values, tiny_config
        )