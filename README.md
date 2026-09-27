# CBIS-DDSM Breast Cancer Classification — Extended Pipeline

This extends the original single-script EfficientNetB0 baseline into a modular
pipeline covering: multi-architecture transfer learning, class-imbalance
handling, hyperparameter tuning, fine-tuning, Grad-CAM, DeepLIFT, and a full
unit test suite.

## Project layout

```
config.py                 Central Config dataclass (paths, hyperparams)
data_pipeline.py          CSV loading/merge, train/val/test split, tf.data, class weights, oversampling
models.py                 Transfer-learning framework (EfficientNetB0 / ResNet50 / DenseNet121), focal loss
train.py                  Stage 1: frozen-backbone training + imbalance handling
fine_tune.py              Stage 2: unfreeze top layers, low-LR continued training
hyperparameter_tuning.py  Dependency-free random search (learning rate, dropout, dense units, batch size)
model_comparison.py       Trains/evaluates every backbone under an identical protocol, ranks by test ROC-AUC
evaluate.py               classification_report / confusion_matrix / ROC-AUC / plotting, as pure functions
explainability.py         Grad-CAM + DeepLIFT (via SHAP DeepExplainer, with a labelled fallback)
main.py                   Orchestrates the full pipeline end-to-end (steps 1-8 below)
tests/                    pytest unit tests for every module above
requirements.txt
```

## Pipeline stages (`main.py`)

1. Load & merge the two CBIS-DDSM CSVs into `dataset_df` (same merge logic as the original script).
2. Stratified train/val/test split.
3. Apply the configured class-imbalance strategy (`Config.imbalance_strategy`):
   - `"class_weight"` — `sklearn`-balanced weights passed to `model.fit(class_weight=...)` (default, matches common practice)
   - `"focal_loss"` — replaces binary cross-entropy with a focal loss that down-weights easy examples
   - `"oversample"` — random oversampling of the minority class at the dataframe level
4. **Compare CNN architectures**: EfficientNetB0, ResNet50, DenseNet121 are each trained (frozen backbone) under identical conditions and ranked by test ROC-AUC (`model_comparison.py`).
5. **Hyperparameter tuning**: random search over learning rate / dropout / dense units / batch size on the winning architecture (`hyperparameter_tuning.py`).
6. **Fine-tuning**: unfreeze the top ~30% of the winning, tuned model's backbone and continue training at a low learning rate (`fine_tune.py`).
7. **Evaluation**: classification report, confusion matrix, ROC-AUC/curve, training curves (`evaluate.py`).
8. **Explainability**: Grad-CAM and DeepLIFT attribution maps for sample test images, plotted side by side (`explainability.py`).

Run the whole thing with:

```bash
pip install -r requirements.txt
python main.py
```

(This expects the CBIS-DDSM CSVs/jpegs at the paths configured in `config.py`,
same layout as the original script — adjust `Config.mass_csv_path` /
`Config.dicom_csv_path` / `Config.jpeg_prefix_new` if your data lives
elsewhere.)

## Testing

```bash
pip install -r requirements.txt
pytest
```

All tests use `weights=None` (random init, no internet/ImageNet download
needed) and tiny synthetic data (8–12 fake 32×32 jpegs generated on the fly),
so the whole suite runs in well under a minute on CPU. Coverage:

| Module | What's tested |
|---|---|
| `data_pipeline.py` | label creation, CSV merge/filter logic, prefix rewriting, stratified split (sizes, no leakage, class ratios), image decoding, `tf.data` batching, class weights, oversampling |
| `models.py` | model construction for every backbone, output shape/range, frozen-by-default, focal loss numerically penalizing confident-wrong > unsure-wrong and ~0 for confident-correct, fine-tune unfreezing (including BatchNorm staying frozen), error handling |
| `train.py` | callback construction, training runs under each imbalance strategy, invalid-strategy error |
| `fine_tune.py` | backbone becomes trainable, optimizer learning rate matches config |
| `hyperparameter_tuning.py` | search-space sampling (size, determinism, grid cap), search driver calls the injected train function once per trial, best-trial selection |
| `evaluate.py` | thresholding, perfect/worst-case/mixed metric computation, single-class edge case |
| `model_comparison.py` | one result row per backbone, table sorted by AUC, best-backbone selection |
| `explainability.py` | Grad-CAM shape/range/error-handling, attribution normalization, gradientxinput fallback shape, SHAP-unavailable fallback path (via mocked import), Grad-CAM/DeepLIFT comparison output |