#EVERY PATH / HYPERPARAMETER LIVES HERE SO MODULES TAKE A Config OBJECT,
#INSTEAD OF HARD-CODING VALUES (MAKES UNIT TESTS ABLE TO INJECT TINY/FAKE VALUES)

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    #PATHS
    mass_csv_path: str = "cbis-ddsm/csv/mass_case_description_train_set.csv"
    dicom_csv_path: str = "cbis-ddsm/csv/dicom_info.csv"
    jpeg_prefix_old: str = "CBIS-DDSM/jpeg/"
    jpeg_prefix_new: str = "cbis-ddsm/jpeg/"
    model_outputs_dir: str = "model_outputs"

    #IMAGE/DATA
    img_size: int = 224
    batch_size: int = 16
    val_test_split: float = 0.30
    test_split_of_holdout: float = 0.50  #split of the holdout into val/test
    random_state: int = 50               #seed for replicable results
    shuffle_buffer: int = 1000

    #TRANSFER LEARNING
    backbones: List[str] = field(
        default_factory=lambda: ["EfficientNetB0", "ResNet50", "DenseNet121"]
    )
    dropout: float = 0.3
    dense_units: Optional[int] = None  #none = no extra dense layer
    weights: str = "imagenet"

    #TRAINING
    frozen_epochs: int = 20
    fine_tune_epochs: int = 10
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    fine_tune_at_fraction: float = 0.7  # unfreeze last 30% of base layers
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3

    #SEARCH/COMPARISON STAGES (ARCHITECTURE COMPARISON, IMBALANCE-STRATEGY COMPARISON)
    #only needs enough epochs to rank candidates relative to each other, not to fully converge
    #the winning configuration gets the full frozen_epochs/fine_tune_epochs budget later, to save costs
    search_epochs: int = 6

    #CLASS IMBALANCE
    imbalance_strategy: str = "focal_loss"  # "class_weight" | "oversample" | "focal_loss"
    #using focal_loss_alpha corrected to 0.5, as focal loss outperformed others in comparison
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.5   #weight on the malignant (positive) class, benign gets 1 - alpha. 0.5 = neutral. (0.25 down-weighted malignant cases and suppressed recall)

    #DECISION-THRESHOLD CALIBRATION
    #calibrated on the validation set to reach at least this malignant recall, then applied once to the test set
    target_malignant_recall: float = 0.90

    #HYPERPARAMETER SEARCH
    hp_search_trials: int = 8
    hp_search_epochs: int = 5
    hp_learning_rates: List[float] = field(default_factory=lambda: [1e-2, 1e-3, 1e-4])
    hp_dropouts: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])
    hp_dense_units: List[Optional[int]] = field(default_factory=lambda: [None, 128, 256])
    hp_batch_sizes: List[int] = field(default_factory=lambda: [16, 32])

    #EXPLAINABILITY
    gradcam_last_conv_layer: Optional[str] = None  # auto detected on None
    deeplift_background_samples: int = 20