import numpy as np
import pytest
import tensorflow as tf

from models import (
    BACKBONES,
    binary_focal_loss,
    build_model,
    compile_model,
    get_backbone_names,
    unfreeze_for_fine_tuning,
)


@pytest.mark.parametrize("backbone_name", get_backbone_names())
def test_build_model_output_shape_and_frozen_by_default(backbone_name):
    model = build_model(backbone_name, img_size=32, weights=None)
    assert model.output_shape == (None, 1)
    assert model.base_model.trainable is False


def test_build_model_rejects_unknown_backbone():
    with pytest.raises(ValueError):
        build_model("NotARealBackbone", img_size=32, weights=None)


def test_build_model_forward_pass_produces_valid_probabilities():
    model = build_model("EfficientNetB0", img_size=32, weights=None)
    batch = np.random.rand(2, 32, 32, 3).astype("float32") * 255
    preds = model(batch, training=False).numpy()
    assert preds.shape == (2, 1)
    assert np.all(preds >= 0) and np.all(preds <= 1)


def test_build_model_with_dense_units_adds_layer():
    model_no_dense = build_model("EfficientNetB0", img_size=32, weights=None, dense_units=None)
    model_with_dense = build_model("EfficientNetB0", img_size=32, weights=None, dense_units=64)
    assert len(model_with_dense.layers) == len(model_no_dense.layers) + 1


def test_compile_model_sets_expected_metrics():
    model = build_model("EfficientNetB0", img_size=32, weights=None)
    compile_model(model, learning_rate=1e-3)

    #READ METRIC NAMES FROM evaluate()'s OWN RETURN DICT RATHER THAN
    #model.metrics - UNDER KERAS 3, model.metrics DOESN'T RELIABLY EXPOSE
    #INDIVIDUAL NAMED METRIC OBJECTS (IT CAN JUST SHOW ['loss', 'compile_metrics']
    #DEPENDING ON VERSION), BUT return_dict=True ALWAYS KEYS RESULTS BY NAME
    x = np.random.rand(2, 32, 32, 3).astype("float32")
    y = np.array([0, 1], dtype="float32")
    results = model.evaluate(x, y, verbose=0, return_dict=True)

    for expected in ["loss", "accuracy", "precision", "recall", "auc"]:
        assert expected in results


def test_unfreeze_for_fine_tuning_unfreezes_top_layers_only():
    model = build_model("EfficientNetB0", img_size=32, weights=None)
    unfreeze_for_fine_tuning(model, fine_tune_at_fraction=0.7)

    base = model.base_model
    assert base.trainable is True

    n_layers = len(base.layers)
    fine_tune_at = int(n_layers * 0.7)

    #EARLIEST LAYERS MUST REMAIN FROZEN
    assert base.layers[0].trainable is False
    #A LATE, NON-BATCHNORM LAYER SHOULD BE TRAINABLE
    late_trainable = [
        l.trainable
        for l in base.layers[fine_tune_at:]
        if not isinstance(l, tf.keras.layers.BatchNormalization)
    ]
    assert any(late_trainable)


def test_unfreeze_for_fine_tuning_keeps_batchnorm_frozen():
    model = build_model("EfficientNetB0", img_size=32, weights=None)
    unfreeze_for_fine_tuning(model, fine_tune_at_fraction=0.0)
    bn_layers = [l for l in model.base_model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    assert bn_layers, "expected at least one BatchNorm layer in the backbone"
    assert all(not l.trainable for l in bn_layers)


def test_unfreeze_raises_without_base_model_attribute():
    plain_model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(4,))])
    with pytest.raises(ValueError):
        unfreeze_for_fine_tuning(plain_model)


def test_binary_focal_loss_penalizes_confident_wrong_predictions_more():
    loss_fn = binary_focal_loss(gamma=2.0, alpha=0.25)
    y_true = tf.constant([[1.0]])

    confident_wrong = loss_fn(y_true, tf.constant([[0.01]]))
    unsure_wrong = loss_fn(y_true, tf.constant([[0.4]]))

    assert confident_wrong.numpy() > unsure_wrong.numpy()


def test_binary_focal_loss_is_near_zero_for_confident_correct_predictions():
    loss_fn = binary_focal_loss(gamma=2.0, alpha=0.25)
    y_true = tf.constant([[1.0]])
    loss = loss_fn(y_true, tf.constant([[0.999]]))
    assert loss.numpy() < 0.01


def test_all_registered_backbones_have_matching_preprocess_fn():
    for name, (constructor, preprocess_fn) in BACKBONES.items():
        assert callable(constructor)
        assert callable(preprocess_fn)