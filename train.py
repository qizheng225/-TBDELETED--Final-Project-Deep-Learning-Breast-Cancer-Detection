#STAGE 1 OF TRANSFER LEARNING: TRAIN THE CLASSIFICATION HEAD ON TOP OF A FROZEN PRETRAINED BACKBONE, WITH CLASS-IMBALANCE HANDLING APPLIED

from typing import Optional

import numpy as np
from tensorflow import keras

from config import Config
from models import build_model, compile_model, binary_focal_loss
from data_pipeline import get_class_weights


#TRAINING CALLBACKS
def build_callbacks(cfg: Config, checkpoint_path: Optional[str] = None):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=cfg.reduce_lr_patience, factor=0.5, min_lr=1e-7
        ),
    ]
    if checkpoint_path:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor="val_auc", mode="max", save_best_only=True
            )
        )
    return callbacks


#LABELS OF A tf.data DATASET
def labels_from_dataset(ds) -> np.ndarray:
    #collects the labels of a (batched or unbatched) dataset, in iteration order
    #for non-shuffled datasets (val/test) this matches the order of model.predict(ds)
    chunks = [np.atleast_1d(y.numpy()) for y in ds.map(lambda x, y: y)]
    return np.concatenate(chunks) if chunks else np.array([])


# IMBALANCE STRATEGY > (loss, class_weight)
def get_imbalance_settings(cfg: Config, train_labels: Optional[np.ndarray] = None):
    # where cfg.imbalance_strategy is turned into a loss + class weights
    loss = "binary_crossentropy"
    class_weight = None

    if cfg.imbalance_strategy == "class_weight":
        if train_labels is None:
            raise ValueError("train_labels are required for the 'class_weight' strategy")
        class_weight = get_class_weights(train_labels)
    elif cfg.imbalance_strategy == "focal_loss":
        loss = binary_focal_loss(gamma=cfg.focal_loss_gamma, alpha=cfg.focal_loss_alpha)
    elif cfg.imbalance_strategy == "oversample":
         #oversampling is applied prior, at the dataframe level, before train_ds is built
        pass
    else:
        raise ValueError(f"Unknown imbalance_strategy: {cfg.imbalance_strategy}")
    return loss, class_weight


#TRAINING MODEL
def train_frozen_model(
    backbone_name: str,
    train_ds,
    val_ds,
    train_labels: np.ndarray,
    cfg: Config,
    epochs: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
):
    #build, compile and train a frozen backbone classifier, applying whichever imbalance strategy is configured (cfg.imbalance_strategy)
    loss, class_weight = get_imbalance_settings(cfg, train_labels)

    #building classification model
    model = build_model(
        backbone_name,
        img_size=cfg.img_size,
        dropout=cfg.dropout,
        dense_units=cfg.dense_units,
        weights=cfg.weights,
        trainable_base=False,
    )
    compile_model(model, learning_rate=cfg.learning_rate, loss=loss)

    #training model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs or cfg.frozen_epochs,
        class_weight=class_weight,
        callbacks=build_callbacks(cfg, checkpoint_path),
        shuffle=False,  #shuffling is handled prior in the tf.data pipeline
    )
    return model, history