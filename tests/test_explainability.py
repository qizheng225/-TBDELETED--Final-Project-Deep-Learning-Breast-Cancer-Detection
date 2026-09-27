import builtins

import numpy as np
import pytest
import tensorflow as tf

from explainability import (
    _gradient_x_input_fallback,
    compare_gradcam_and_deeplift,
    deeplift_attributions,
    find_last_conv_layer,
    grad_cam,
    normalize_attribution_map,
    overlay_heatmap,
)
from models import build_model


@pytest.fixture
def tiny_trained_model(tiny_config):
    return build_model("EfficientNetB0", img_size=tiny_config.img_size, weights=None)


@pytest.fixture
def sample_image(tiny_config):
    rng = np.random.default_rng(0)
    return (rng.random((tiny_config.img_size, tiny_config.img_size, 3)) * 255).astype("float32")


@pytest.fixture
def background_images(tiny_config):
    rng = np.random.default_rng(1)
    return (rng.random((4, tiny_config.img_size, tiny_config.img_size, 3)) * 255).astype("float32")


def test_find_last_conv_layer_returns_a_4d_output_layer(tiny_trained_model):
    layer_name = find_last_conv_layer(tiny_trained_model, tiny_trained_model.base_model)
    layer = tiny_trained_model.base_model.get_layer(layer_name)
    assert len(layer.output.shape) == 4


def test_grad_cam_output_matches_image_spatial_size(tiny_trained_model, sample_image):
    heatmap = grad_cam(tiny_trained_model, sample_image)
    assert heatmap.shape == sample_image.shape[:2]


def test_grad_cam_output_is_normalized_between_0_and_1(tiny_trained_model, sample_image):
    heatmap = grad_cam(tiny_trained_model, sample_image)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_grad_cam_raises_without_base_model():
    plain_model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(32, 32, 3)), tf.keras.layers.Conv2D(4, 3), tf.keras.layers.Flatten(), tf.keras.layers.Dense(1)]
    )
    image = np.random.rand(32, 32, 3).astype("float32")
    with pytest.raises(ValueError):
        grad_cam(plain_model, image)


def test_overlay_heatmap_output_shape_and_range(sample_image):
    heatmap = np.random.rand(*sample_image.shape[:2]).astype("float32")
    overlay = overlay_heatmap(sample_image, heatmap)
    assert overlay.shape == (*sample_image.shape[:2], 3)
    assert overlay.min() >= 0.0 and overlay.max() <= 1.0


def test_normalize_attribution_map_handles_all_zero_input():
    zeros = np.zeros((8, 8))
    normalized = normalize_attribution_map(zeros)
    assert np.all(normalized == 0)


def test_normalize_attribution_map_scales_to_unit_range():
    attr = np.array([[-4.0, 2.0], [0.0, -8.0]])
    normalized = normalize_attribution_map(attr)
    assert normalized.max() == pytest.approx(1.0)
    assert normalized.min() >= 0.0


def test_gradient_x_input_fallback_output_shape(tiny_trained_model, sample_image, background_images):
    images = np.expand_dims(sample_image, 0)
    attributions = _gradient_x_input_fallback(tiny_trained_model, images, background_images)
    assert attributions.shape == (1, sample_image.shape[0], sample_image.shape[1])


def test_deeplift_attributions_falls_back_when_shap_not_installed(
    tiny_trained_model, sample_image, background_images, monkeypatch
):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("simulated: shap not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    images = np.expand_dims(sample_image, 0)
    attributions = deeplift_attributions(tiny_trained_model, images, background_images)
    assert attributions.shape == (1, sample_image.shape[0], sample_image.shape[1])


def test_compare_gradcam_and_deeplift_returns_two_comparable_maps(
    tiny_trained_model, sample_image, background_images, monkeypatch
):
    #FORCE THE FALLBACK PATH SO THIS TEST DOESN'T REQUIRE SHAP TO BE INSTALLED
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("simulated: shap not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    cam, deeplift_map = compare_gradcam_and_deeplift(tiny_trained_model, sample_image, background_images)
    assert cam.shape == deeplift_map.shape == sample_image.shape[:2]
    assert deeplift_map.min() >= 0.0 and deeplift_map.max() <= 1.0 + 1e-6