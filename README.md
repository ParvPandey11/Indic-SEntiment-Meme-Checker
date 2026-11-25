# IndicSent-M: Multimodal Meme Sentiment

This repository contains runnable text-only, image-only, and text-plus-image sentiment baselines for Memotion 3. The classes are **negative**, **neutral**, and **positive**. Text is encoded with `xlm-roberta-base`; images use ImageNet-pretrained ResNet-50. Fusion supports concatenation and a learned gate.

## Data and split

Keep the downloaded dataset out of Git. Expected layout:

```text
data/
  csv/train.csv
  csv/test.csv
  trainImages/
  testImages/
```

Run preparation to produce `data/processed/train.csv`, `val.csv`, and `test.csv`. The validation rows are a reproducible, stratified 15% split of the labeled `train.csv` (seed 42); they are **not** the organizer's official validation split. Test labels are unavailable, so test is used only for predictions and no test score is reported. The preparation script matches the CSV image identifiers to the released image files and leaves the original CSV files unchanged.

## Setup and run

From the repository root on macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/00_inspect_data.py
python3 scripts/01_prepare_data.py
```

Start with a small smoke run:

```bash
python3 -m src.train --mode fusion --limit 8 --bs 2 --epochs 1 --freeze --workers 0 --name smoke
```

Run the frozen encoder baselines:

```bash
python3 -m src.train --mode text   --epochs 3 --bs 8 --freeze --workers 0 --seed 42 --name text_frozen_e3
python3 -m src.train --mode image  --epochs 3 --bs 8 --freeze --workers 0 --seed 42 --name image_frozen_e3
python3 -m src.train --mode fusion --fusion concat --epochs 3 --bs 8 --freeze --workers 0 --seed 42 --name fusion_frozen_e3
python3 -m src.train --mode fusion --fusion gated  --epochs 3 --bs 8 --freeze --workers 0 --seed 42 --name fusion_gated_frozen_e3
```

On Apple Silicon, training uses MPS when available. Keep `--workers 0` for compatibility with macOS multiprocessing. Full encoder fine-tuning is more demanding; the commands above freeze the pretrained encoders and train the classification head.

## Explain one prediction

After a run has produced `best.pt` and `metrics.json`, generate a local explanation for a validation row:

```bash
python3 -m src.explain --run runs/fusion_frozen_e3 --split val --row 0
```

The JSON output is saved under that run. Text scores measure the change in predicted-class probability after removing one whitespace token. Image scores measure the change after masking each cell of a coarse grid. Image and fusion runs also save a Grad-CAM overlay from the final ResNet-50 convolution block. These are model sensitivity/attention indicators, not causal explanations. See [xai/README.md](xai/README.md).

## Local demo

The optional Gradio demo lets you choose a saved seed-42 baseline, enter meme OCR text, upload an image, and view class probabilities with a Grad-CAM overlay for models that use images.

```bash
python3 -m pip install -r requirements-demo.txt
python3 app/demo.py
```

Open the local URL printed in the terminal. See [app/README.md](app/README.md).

## Three-seed validation results

All four models were trained for three epochs with frozen encoders, batch size 8, and the same stratified validation split. For each seed, the checkpoint with the highest validation macro-F1 was selected. The table reports mean and sample standard deviation across seeds 42, 43, and 44; per-seed scores are in [reports/multiseed_validation.md](reports/multiseed_validation.md). Recreate the report with `python3 scripts/02_summarize_runs.py`.

| Model | Validation accuracy (mean ± SD) | Validation macro-F1 (mean ± SD) |
|---|---:|---:|
| Text only | 0.3994 ± 0.0281 | 0.3147 ± 0.0440 |
| Image only | 0.3765 ± 0.0157 | 0.3665 ± 0.0088 |
| Concatenation fusion | 0.3730 ± 0.0214 | 0.3689 ± 0.0191 |
| Gated fusion | 0.3717 ± 0.0071 | 0.3698 ± 0.0045 |

Gated fusion has the highest mean validation macro-F1, but its advantage is small (about 0.001 over concatenation and 0.003 over image-only), and three seeds are still a limited comparison. The unlabeled test split is used only for predictions; no test scores are available. These metrics are from a validation split derived from the labeled training CSV, not the organizer's official validation set.

## Outputs

Each `runs/<name>/` contains `best.pt`, `last.pt`, `metrics.json`, a validation confusion matrix, and (when test labels are absent) `test_predictions.csv`. Do not interpret those test predictions as measured test performance.

## Scope

Implemented: data inspection/preparation, stratified train/validation split, text/image/fusion baselines, validation metrics and checkpoint selection, unlabeled test prediction, and perturbation-based explanation output.

Still planned: the organizer's labeled validation data if it becomes available, further seeds or statistical testing, and LIME/SHAP text explanations. No dataset annotations or performance values are fabricated.
