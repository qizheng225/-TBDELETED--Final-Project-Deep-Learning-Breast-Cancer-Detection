from data_pipeline import make_tf_dataset
from fine_tune import fine_tune_model
from models import build_model


def test_fine_tune_model_unfreezes_base_and_runs(tiny_config, synthetic_dataset_df):
    model = build_model("EfficientNetB0", img_size=tiny_config.img_size, weights=None)
    train_ds = make_tf_dataset(synthetic_dataset_df, tiny_config.img_size, tiny_config.batch_size, shuffle=True)
    val_ds = make_tf_dataset(synthetic_dataset_df, tiny_config.img_size, tiny_config.batch_size)

    fine_tuned_model, history = fine_tune_model(model, train_ds, val_ds, tiny_config, epochs=1)

    assert fine_tuned_model.base_model.trainable is True
    assert "loss" in history.history


def test_fine_tune_model_uses_lower_learning_rate(tiny_config, synthetic_dataset_df):
    model = build_model("EfficientNetB0", img_size=tiny_config.img_size, weights=None)
    train_ds = make_tf_dataset(synthetic_dataset_df, tiny_config.img_size, tiny_config.batch_size, shuffle=True)
    val_ds = make_tf_dataset(synthetic_dataset_df, tiny_config.img_size, tiny_config.batch_size)

    tiny_config.fine_tune_learning_rate = 1e-6
    fine_tuned_model, _ = fine_tune_model(model, train_ds, val_ds, tiny_config, epochs=1)

    lr = float(fine_tuned_model.optimizer.learning_rate.numpy())
    assert abs(lr - 1e-6) < 1e-9