#TRANSFER-LEARNING FRAMEWORK THAT BUILDS/COMPILES A CLASSIFICATION HEAD ON TOP OF ANY OF SEVERAL PRETRAINED BACKBONES,
#SO ARCHITECTURES CAN BE SWAPPED AND COMPARED WITH ONE FUNCTION CALL INSTEAD OF DUPLICATING THE MODEL-BUILDING CODE PER BACKBONE

from typing import Optional

import tensorflow as tf
from tensorflow import keras


#BACKBONE REGISTRY
#each entry maps a human-readable name to the keras applications constructor, and its matching preprocess_input function
BACKBONES = {
    "EfficientNetB0": (
        tf.keras.applications.EfficientNetB0,
        tf.keras.applications.efficientnet.preprocess_input,
    ),
    "ResNet50": (
        tf.keras.applications.ResNet50,
        tf.keras.applications.resnet50.preprocess_input,
    ),
    "DenseNet121": (
        tf.keras.applications.DenseNet121,
        tf.keras.applications.densenet.preprocess_input,
    ),
}


def get_backbone_names():
    return list(BACKBONES.keys())


#DATA AUGMENTATION FOR TRAINING
def build_data_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )


#BUILDING CLASSIFICATION MODEL
def build_model(
    backbone_name: str,
    img_size: int = 224,
    dropout: float = 0.3,
    dense_units: Optional[int] = None,
    weights: Optional[str] = "imagenet",
    trainable_base: bool = False,
    augment: bool = True,
) -> keras.Model:
    #transfer learning classifier: backbone > GAP(global average pooling) > [Dense] > dropout > sigmoid
    #weights: "imagenet" OR None
    #trainable_base: freeze (False) OR unfreeze (True)
    if backbone_name not in BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. Available: {get_backbone_names()}"
        )

    constructor, preprocess_input = BACKBONES[backbone_name]

    #load pretrained backbone
    base_model = constructor(
        include_top=False, weights=weights, input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = trainable_base

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = inputs
    if augment:
        x = build_data_augmentation()(x)
    x = preprocess_input(x)  #pixel normalization
    x = base_model(x, training=trainable_base)

    gap_layer = keras.layers.GlobalAveragePooling2D()
    x = gap_layer(x)

    dense_layer = None
    if dense_units:
        dense_layer = keras.layers.Dense(dense_units, activation="relu")
        x = dense_layer(x)

    dropout_layer = keras.layers.Dropout(dropout)
    x = dropout_layer(x)

    output_layer = keras.layers.Dense(1, activation="sigmoid")
    outputs = output_layer(x)

    model = keras.Model(inputs, outputs, name=f"{backbone_name}_classifier")
    model.base_model = base_model  # stash reference for later fine-tuning

    #store explicit head and preprocessing references for GRAD-CAM
    #the head is replayed on intermediate conv activations to avoid ambiguity when accessing layers inside the nested backbone
    head_layers = [gap_layer]
    if dense_layer is not None:
        head_layers.append(dense_layer)
    head_layers.append(dropout_layer)
    head_layers.append(output_layer)
    model.head_layers = head_layers
    model.preprocess_input = preprocess_input

    return model


#LOSS FUNCTIONS (USED FOR THE "focal_loss" IMBALANCE STRATEGY)
def binary_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    #focal loss for binary classification that focuses on hard examples in imbalanced data
    #alternative to class weighting for imbalanced data

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)

        loss = -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(loss)

    loss_fn.__name__ = "binary_focal_loss"
    return loss_fn


#COMPILE MODEL
def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
    loss: str = "binary_crossentropy",
) -> keras.Model:
    #compile using the standard set of metrics used throughout the project
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


#FINE-TUNING
def unfreeze_for_fine_tuning(
    model: keras.Model, fine_tune_at_fraction: float = 0.7
) -> keras.Model:
    #unfreezes the top proportion of the backbone's layers (1-fine_tune_at_fraction), so they can be fine-tuned at a low learning rate,
    #while keeping the earlier layers frozen. batch-norm layers are kept frozen to avoid destabilising the running statistics
    base_model = getattr(model, "base_model", None)
    if base_model is None:
        raise ValueError(
            "model has no `.base_model` attribute - build it with build_model()"
        )

    base_model.trainable = True
    n_layers = len(base_model.layers)
    fine_tune_at = int(n_layers * fine_tune_at_fraction)

    for i, layer in enumerate(base_model.layers):
        if i < fine_tune_at or isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    return model