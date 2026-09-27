# DEPENDENCY-FREE RANDOM-SEARCH HYPERPARAMETER TUNING, DESIGNED FOR EASY TESTING AND USE IN ENVIRONMENTS
# WHERE ADDITIONAL PACKAGES CANNOT BE INSTALLED.

import itertools
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd

from config import Config


@dataclass
class TrialResult:
    params: Dict
    val_auc: float
    history: Optional[object] = None


def sample_search_space(cfg: Config, n_trials: int, seed: int = 42) -> List[Dict]:
    #randomly sample  n_trials unique hyperparameters combinatinos from the grid
    grid = list(
        itertools.product(
            cfg.hp_learning_rates, cfg.hp_dropouts, cfg.hp_dense_units, cfg.hp_batch_sizes
        )
    )
    rng = random.Random(seed)
    rng.shuffle(grid)
    n_trials = min(n_trials, len(grid))
    combos = grid[:n_trials]

    return [
        {"learning_rate": lr, "dropout": do, "dense_units": du, "batch_size": bs}
        for (lr, do, du, bs) in combos
    ]


def run_hyperparameter_search(
    cfg: Config,
    train_fn: Callable[[Dict], float],
) -> List[TrialResult]:
    #generic random-search driver with a injected training function.
    #keeps the search logic independent of tensorflow and easy to unit test
    #in production, train_fn wraps model construction and training. In tests, can be replaced with simple func that returns a validation AUC
    param_sets = sample_search_space(cfg, cfg.hp_search_trials)
    results = []
    for params in param_sets:
        val_auc = train_fn(params)
        results.append(TrialResult(params=params, val_auc=val_auc))
    return results


def best_trial(results: List[TrialResult]) -> TrialResult:
    if not results:
        raise ValueError("No trial results to select from.")
    return max(results, key=lambda r: r.val_auc)


#FLATTEN TRIAL RESULTS INTO ONE TABLE, SORTED BEST FIRST
def trials_to_dataframe(results: List[TrialResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = dict(r.params)
        row["val_auc"] = r.val_auc
        rows.append(row)
    return pd.DataFrame(rows).sort_values("val_auc", ascending=False).reset_index(drop=True)


def make_keras_train_fn(
    backbone_name: str,
    train_df,
    val_df,
    cfg: Config,
):
    #returns a train_fn  that builds and trains a keras model.
    #kept separate so the search logic can be tested without tensorflow,
    #tensorflow dependencies are imported only when needed
    from data_pipeline import make_tf_dataset
    from models import build_model, compile_model
    from train import build_callbacks, get_imbalance_settings

    def train_fn(params: Dict) -> float:
        #create tf.data datasets
        train_ds = make_tf_dataset(
            train_df, img_size=cfg.img_size, batch_size=params["batch_size"], shuffle=True
        )
        val_ds = make_tf_dataset(val_df, img_size=cfg.img_size, batch_size=params["batch_size"])

        #build classification model
        model = build_model(
            backbone_name,
            img_size=cfg.img_size,
            dropout=params["dropout"],
            dense_units=params["dense_units"],
            weights=cfg.weights,
            trainable_base=False,
        )
        #translates cfg.imbalance_strategy into a loss + class_weight exactly as train_frozen_model() and fine_tune_model() do,
        #so HP search trains under the configured strategy
        loss, class_weight = get_imbalance_settings(cfg, train_df["label"].values)
        compile_model(model, learning_rate=params["learning_rate"], loss=loss)

        #training model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.hp_search_epochs,
            class_weight=class_weight,
            callbacks=build_callbacks(cfg),
            verbose=0,
            shuffle=False,  # shuffling is already handled prior in the tf.data pipeline
        )
        return float(max(history.history.get("val_auc", [0.0])))

    return train_fn