import pytest

from hyperparameter_tuning import (
    TrialResult,
    best_trial,
    run_hyperparameter_search,
    sample_search_space,
)


def test_sample_search_space_respects_n_trials(tiny_config):
    tiny_config.hp_search_trials = 3
    combos = sample_search_space(tiny_config, n_trials=3, seed=1)
    assert len(combos) == 3
    for c in combos:
        assert set(c.keys()) == {"learning_rate", "dropout", "dense_units", "batch_size"}


def test_sample_search_space_caps_at_grid_size(tiny_config):
    tiny_config.hp_learning_rates = [1e-3]
    tiny_config.hp_dropouts = [0.3]
    tiny_config.hp_dense_units = [None]
    tiny_config.hp_batch_sizes = [16]
    combos = sample_search_space(tiny_config, n_trials=50, seed=1)
    assert len(combos) == 1  # grid only has one combination


def test_sample_search_space_is_deterministic_given_seed(tiny_config):
    a = sample_search_space(tiny_config, n_trials=4, seed=7)
    b = sample_search_space(tiny_config, n_trials=4, seed=7)
    assert a == b


def test_run_hyperparameter_search_calls_train_fn_per_trial(tiny_config):
    tiny_config.hp_search_trials = 3
    calls = []

    def fake_train_fn(params):
        calls.append(params)
        return params["learning_rate"]  # arbitrary deterministic "auc"

    results = run_hyperparameter_search(tiny_config, fake_train_fn)
    assert len(results) == 3
    assert len(calls) == 3
    assert all(isinstance(r, TrialResult) for r in results)


def test_best_trial_picks_highest_val_auc():
    results = [
        TrialResult(params={"a": 1}, val_auc=0.7),
        TrialResult(params={"a": 2}, val_auc=0.9),
        TrialResult(params={"a": 3}, val_auc=0.5),
    ]
    winner = best_trial(results)
    assert winner.params == {"a": 2}


def test_best_trial_raises_on_empty_results():
    with pytest.raises(ValueError):
        best_trial([])