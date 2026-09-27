#STAGE 2 OF TRANSFER LEARNING, UNFREEZES TOP PORTION OF THE PRETRAINED BACKBONE AND CONTINUE TRAINING AT A MUCH LOWER LEARNING RATE.
#KEPT SEPARATE FROM train.py SO THE TWO STAGES CAN BE UNIT TESTED, SWAPPED, OR SKIPPED INDEPENDENTLY

from typing import Optional

from config import Config
from models import compile_model, unfreeze_for_fine_tuning
from train import build_callbacks, get_imbalance_settings, labels_from_dataset


#TRAINING MODEL (FINE-TUNE STAGE)
def fine_tune_model(
    model,
    train_ds,
    val_ds,
    cfg: Config,
    epochs: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    train_labels=None,
):
    #unfreeze the top (1 - fine_tune_at_fraction) of the backbone and continue training the model at a lower learning rate
    #apply the same imbalance handling as the frozen stage (class weights / focal loss), otherwise fine-tuning reverts to cross-entropy
    #train_labels is derived from train_ds when not given (only needed for the class_weight strategy)
    if train_labels is None and cfg.imbalance_strategy == "class_weight":
        train_labels = labels_from_dataset(train_ds)
    loss, class_weight = get_imbalance_settings(cfg, train_labels)

    model = unfreeze_for_fine_tuning(model, cfg.fine_tune_at_fraction)
    compile_model(model, learning_rate=cfg.fine_tune_learning_rate, loss=loss)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs or cfg.fine_tune_epochs,
        class_weight=class_weight,
        callbacks=build_callbacks(cfg, checkpoint_path),
        shuffle=False,  # shuffling is handled prior in the tf.data pipeline
    )
    return model, history