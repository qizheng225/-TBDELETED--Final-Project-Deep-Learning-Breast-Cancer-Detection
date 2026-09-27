#SHARED FIXTURES FOR THE WHOLE TEST SUITE
#
#KEY DESIGN CHOICE: EVERY FIXTURE KEEPS THINGS TINY (8x8 IMAGES, ~12 ROWS,
#1-2 EPOCHS) AND USES weights=None WHEN BUILDING MODELS, SO THE FULL TEST
#SUITE RUNS IN SECONDS ON CPU WITH NO NETWORK ACCESS AND NO REAL CBIS-DDSM
#DATA REQUIRED

import numpy as np
import pandas as pd
import pytest

from config import Config


@pytest.fixture
def tiny_config(tmp_path):
    cfg = Config()
    cfg.img_size = 32
    cfg.batch_size = 4
    cfg.frozen_epochs = 1
    cfg.fine_tune_epochs = 1
    cfg.hp_search_epochs = 1
    cfg.hp_search_trials = 2
    cfg.weights = None          # avoid downloading imagenet weights in CI/tests
    cfg.model_outputs_dir = str(tmp_path / "model_outputs")
    cfg.random_state = 0
    return cfg


@pytest.fixture
def fake_jpeg_dir(tmp_path):
    #write a handful of tiny random RGB jpegs to disk and return the paths
    from PIL import Image

    img_dir = tmp_path / "jpeg"
    img_dir.mkdir()
    paths = []
    rng = np.random.default_rng(0)
    for i in range(12):
        arr = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
        path = img_dir / f"img_{i}.jpg"
        Image.fromarray(arr).save(path)
        paths.append(str(path))
    return paths


@pytest.fixture
def synthetic_dataset_df(fake_jpeg_dir):
    #A DATAFRAME SHAPED LIKE THE REAL dataset_df PRODUCED BY build_dataset_df
    n = len(fake_jpeg_dir)
    labels = [0, 1] * (n // 2)  # balanced-ish, deterministic
    return pd.DataFrame(
        {
            "real_path": fake_jpeg_dir,
            "label": labels,
            "pathology": ["MALIGNANT" if l == 1 else "BENIGN" for l in labels],
        }
    )


@pytest.fixture
def synthetic_mass_and_dicom_df():
    #MINIMAL FRAMES EXERCISING THE MERGE LOGIC IN build_dataset_df
    #
    #NOTE: THE UID IS EXTRACTED VIA path.split("/")[2] ON BOTH SIDES (MATCHING
    #THE REAL CBIS-DDSM CSV LAYOUT), SO THESE FAKE PATHS MUST PLACE THE UID AT
    #INDEX 2: "<folder>/<subfolder>/<UID>/<file>"
    mass_df = pd.DataFrame(
        {
            "pathology": ["MALIGNANT", "BENIGN", "BENIGN_WITHOUT_CALLBACK", "MALIGNANT"],
            "cropped image file path": [
                "Mass-Training_P_00001/1.0/UID_A/000000.dcm",
                "Mass-Training_P_00002/1.0/UID_B/000000.dcm",
                "Mass-Training_P_00003/1.0/UID_C/000000.dcm",
                "Mass-Training_P_00004/1.0/UID_D/000000.dcm",
            ],
        }
    )
    dicom_df = pd.DataFrame(
        {
            "SeriesDescription": [
                "cropped images",
                "cropped images",
                "full mammogram images",  # should be filtered out
                "cropped images",
            ],
            "image_path": [
                "CBIS-DDSM/jpeg/UID_A/000000.jpg",
                "CBIS-DDSM/jpeg/UID_B/000000.jpg",
                "CBIS-DDSM/jpeg/UID_C/000000.jpg",
                "CBIS-DDSM/jpeg/UID_D/000000.jpg",
            ],
        }
    )
    return mass_df, dicom_df