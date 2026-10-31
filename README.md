# Railway Track Defect Classifier

Explainable, calibrated, priority-ranked binary defect classifier. Everything runs in Google Colab — nothing needs to run on your own machine.

## What to run where

**Everything is in `colab/RailDefect_Classifier_Colab.ipynb`. Run it entirely in Colab, top to bottom.**

Steps:
1. Get a Kaggle API key (one-time): kaggle.com → your profile picture → **Settings** → **API** → **Create New Token**. This downloads `kaggle.json` to your computer. Don't upload it anywhere else or commit it to git — it's your personal API credential.
2. Open `colab/RailDefect_Classifier_Colab.ipynb` in Google Colab (upload it, or open from Drive).
3. **Runtime → Change runtime type → GPU.**
4. Run every cell top to bottom, in order. When the notebook asks you to upload `kaggle.json` (Section 2), do that.
5. Training takes roughly 10–20 minutes. Everything after that (Grad-CAM, calibration, uncertainty, severity, priority ranking, selective prediction, sanity check) is fast — seconds to a couple minutes total, since it's all inference on the one trained model, no extra training.
6. At the end, the notebook downloads a results zip to your browser (`raildefect_export.zip`) and also saves everything to `MyDrive/RailDefectClassifier/outputs/` on your Drive.

## What's in the notebook

| Section | What it does |
|---|---|
| 1–2 | Setup, Kaggle credentials, dataset download |
| 3–5 | Auto-detect dataset structure, build train/val/test split, dataloaders |
| 6–8 | Model (ResNet18 transfer learning), training, test evaluation |
| 9 | Grad-CAM — visual heatmap of where the model "looked" |
| 10 | Confidence calibration (temperature scaling) + Expected Calibration Error |
| 11 | Monte Carlo Dropout uncertainty |
| 11b | Aleatoric vs. epistemic uncertainty decomposition |
| 11c | Out-of-distribution detection (Mahalanobis distance on embeddings) |
| 12 | Grad-CAM-derived severity proxy + combined inspection-priority score, with per-image test photo + Grad-CAM overlay export |
| 13 | Selective prediction (reject/flag low-confidence cases) |
| 14 | Grad-CAM sanity check (proves explanations are real, not decoration) |
| 15 | Export model + metrics + visualizations |

## What you'll get in the export
- `best_model.pt` — trained model weights
- `metrics.json` — test accuracy, ROC-AUC, confusion matrix, ROC curve points, full training history, ECE before/after calibration, reliability-diagram bin data, MC-Dropout uncertainty (summary + per-image values), selective-prediction coverage/accuracy (including a full threshold sweep), per-split dataset composition (`dataset_stats`), and whether uncertainty-decomposition/OOD plots were generated
- `priority_ranked_results.csv` — every test image ranked by inspection priority (severity × uncertainty), including per-image aleatoric/epistemic uncertainty and OOD (Mahalanobis) distance, plus `image_file`/`gradcam_file` columns pointing at the exported photo and overlay for that row
- `test_images/` — the actual test-set photos, copied out with collision-free filenames (so the dashboard can show you the real image, not just a path)
- `gradcam_overlays/` — the matching Grad-CAM heatmap overlay for each test image, same filenames as `test_images/`
- `visualizations/` — `gradcam_examples.png`, `gradcam_sanity_check.png`, `training_curves.png`, `confusion_matrix_roc.png`, `reliability_diagram.png`, `mc_dropout_uncertainty_hist.png`, `severity_priority_distributions.png`, `selective_prediction_sweep.png`, `uncertainty_decomposition.png`, `ood_mahalanobis.png`

## Local dashboard (optional, runs entirely on your machine)

A static HTML/JS dashboard that visualizes everything above — no backend, no install beyond Python (which you already have).

1. Download `raildefect_export.zip` from the Colab notebook (or copy it from your Drive at `MyDrive/RailDefectClassifier/outputs/`).
2. Extract its contents into `dashboard/data/` in this project (so you end up with `dashboard/data/metrics.json`, `dashboard/data/priority_ranked_results.csv`, `dashboard/data/test_images/`, `dashboard/data/gradcam_overlays/`, `dashboard/data/visualizations/*.png`).
3. From inside the `dashboard/` folder, run a simple local server (needed because browsers block `fetch()` of local files opened directly — this is not a real backend, just a one-line static file server):
   ```
   cd RailDefectClassifier/dashboard
   python -m http.server 8000
   ```
4. Open `http://localhost:8000` in your browser.

The dashboard shows: summary metric cards, a dataset-composition chart (per split, defective vs. non-defective), training curves, confusion matrix + ROC curve, calibration reliability diagrams (before/after), MC-Dropout uncertainty distribution, aleatoric-vs-epistemic uncertainty decomposition, the OOD (Mahalanobis) signal, the coverage-vs-accuracy selective-prediction trade-off, Grad-CAM examples, severity/priority distributions, the sanity-check comparison, and a sortable/searchable/clickable table of every test image ranked by inspection priority — clicking a row opens the real test photo next to its Grad-CAM overlay with full trust-signal stats (severity, calibrated confidence, MC-Dropout std, aleatoric/epistemic uncertainty, OOD distance, priority score). Themed in the Indian tricolor (saffron/white/green, navy Ashoka Chakra accent).

## If the dataset auto-detection fails (Section 3)
The notebook tries to automatically find the two class folders (defective / non-defective) after downloading. If it can't, run `!find /content/railway_data -maxdepth 3` in a new cell to see the actual folder layout, then manually set `CLASS_DIRS` to the two correct paths before continuing.

## Research-integrity note
Only report numbers this notebook actually prints — don't round up, don't estimate missing ones. If something fails or looks off, that's worth reporting honestly too, not papering over.
