# Railway Track Defect Classifier: Project Summary and Novelty

## 1. What this project is

A binary image classifier (**defective** vs. **non-defective**) for railway track photos, built as a transfer-learning fine-tune of a pretrained ResNet18 backbone, trained on the Kaggle dataset `salmaneunus/railway-track-fault-detection`. Training itself — data loading, the backbone, the classification head, the optimizer loop — is a standard, well-established recipe; it is not where this project's contribution lies, and this write-up doesn't claim otherwise.

What the project actually adds is a layer of **explainability, calibration, and uncertainty-awareness** wrapped around that trained classifier, entirely at inference time (no extra training cost): the model doesn't just output "defective" or "not," it shows *where* it's looking, *how much to trust* that confidence number, *how sure it really is*, and *which images most need a human to double-check them*. That combination — not any single piece of it — is the actual novelty claim, and it's deliberately scoped to be citable against real literature rather than asserted as original on its own.

## 2. The pipeline, in order

| Step | What it does | Technique / reference |
|---|---|---|
| Classification | ResNet18 (ImageNet-pretrained), fine-tuned end-to-end, binary output | Standard transfer learning |
| Visual explanation | Grad-CAM heatmap over the input image, showing which pixels drove the prediction | Selvaraju et al., 2017 |
| Confidence calibration | Temperature scaling (single scalar fit on validation logits), reported as Expected Calibration Error (ECE) before/after | Guo et al., 2017 |
| Epistemic uncertainty | Monte Carlo Dropout — dropout left active at inference, variance across stochastic passes | Gal & Ghahramani, 2016 |
| Uncertainty decomposition | Splitting total predictive uncertainty into aleatoric (irreducible, data noise) vs. epistemic (reducible, model's own ignorance) components | Kendall & Gal, 2017; Houlsby et al., 2011 (BALD) |
| Out-of-distribution flagging | Mahalanobis distance of a test image's penultimate-layer embedding from the fitted training-embedding distribution | Lee et al., 2018 |
| Severity + inspection priority | A Grad-CAM-derived severity proxy (fraction of the heatmap strongly activated), combined with the uncertainty signals above into one ranking score | Novel combination (see §3) — components are cited individually, this specific scoring function is this project's own construction |
| Selective prediction | Below-threshold predictions are flagged "needs manual review" instead of forced to a yes/no, evaluated as a coverage-vs-accuracy trade-off | Geifman & El-Yaniv, 2017 |
| Explanation sanity check | Randomizing the model's final layer and showing the Grad-CAM heatmap degrades, as evidence the explanations are tied to learned weights rather than generic edge detection | Adebayo et al., 2018 |

## 3. What's actually novel here, honestly stated

None of the individual techniques above is new — each is a published, well-established method, cited above. The honest novelty claim is narrower and has two parts:

**a) The combination is not standard practice in published rail-defect-classification work.** Section 4 below checks this directly against two very recent, closely comparable papers: neither combines calibration + uncertainty quantification + explainability + inspection-priority ranking the way this project does. Most published rail-defect models report accuracy and, at most, one add-on (usually just Grad-CAM) — not the full stack.

**b) The inspection-priority score is this project's own construction**, not lifted from a paper: it combines the Grad-CAM severity proxy with a weighted blend of calibrated-confidence uncertainty and MC-Dropout uncertainty into a single ranking number, meant to answer a concrete operational question — "if a human can only manually check N images today, which N should they look at first?" That specific scoring formula is original engineering built on top of cited, non-original components. This is flagged explicitly rather than implied to be a peer-reviewed contribution.

## 4. Literature positioning — what recent published work does and doesn't do

Two recent, directly comparable papers were checked specifically to see whether this combination already exists in the literature:

**Gungor, M. A., & Gencol, K. (2026). "Optimized Transfer Learning Models for Rail Surface Defect Classification with Explainable AI Validation." *Engineering, Technology & Applied Science Research*, 16(4), 38037–38044.**
Compares seven pretrained CNN backbones (Inception-V3, MobileNet-V1/V2/V3, NasNetMobile, ResNet50, EfficientNet-B0) on a rail surface defect dataset (flaking/spalling/squat), with Optuna-based hyperparameter search and an accuracy-vs-compute Pareto analysis. Uses Grad-CAM for explainability (applied only to the Pareto-optimal models). Best accuracy: MobileNet-V2 at 81.6%. **Does not** include confidence calibration, uncertainty quantification, or any inspection-priority ranking — the Pareto analysis is an efficiency trade-off, not a priority score for human review.

**Phaphuangwittayakul, A., Harnpornchai, N., Ying, F., & Zhang, J. (2024). "RailTrack-DaViT: A Vision Transformer-Based Approach for Automated Railway Track Defect Detection." *Journal of Imaging*.**
Uses a pretrained Dual Attention Vision Transformer (DaViT) backbone, two-stage training (frozen backbone, then full fine-tune), tested across four datasets including a private ThaiRailTrack set. Reports strong accuracy (96.9%–99.2% depending on dataset). **Does not** report confidence calibration, does not perform uncertainty estimation at inference (dropout is used only as training regularization, not MC-Dropout-at-inference), does not do severity or inspection-priority ranking, and does not report any explainability method.

For broader context on why uncertainty quantification matters in railway AI specifically (not rail-defect classification, but the same underlying concern — a wrong confident prediction in railway infrastructure has real safety cost):

**Andéol, L., Fel, T., de Grancey, F., & Mossina, L. (2023). "Conformal Prediction for Trustworthy Detection of Railway Signals." AAAI 2023 workshop.** (SNCF/Thales/IRT Saint Exupéry-affiliated.)
Applies conformal prediction to railway signal (traffic-light) bounding-box detection, using detector-output-widening to guarantee ~90% true-box coverage. A different task (signal detection, not track-defect classification) and a different uncertainty method (conformal prediction vs. this project's MC-Dropout + calibration), but it's evidence that the railway-AI research community treats "don't just report a raw confidence number" as a live, serious problem — this project applies that same concern to track-defect classification specifically, where (as of this check) it doesn't yet appear addressed.

**Honest caveat:** this is not an exhaustive literature review — it's a targeted check against the closest comparable papers findable in the time available, done to avoid an unsupported novelty claim rather than to survey the whole field. If a reviewer or teacher points to a paper that already does this combination, that should be taken seriously and the claim narrowed accordingly, not defended reflexively.

## 5. Scope and honesty notes (for the write-up/presentation)

- This is a **binary image classifier**, not object-detection-level defect localization (no bounding boxes around defects) — Grad-CAM gives an approximate heatmap of attention, not a precise segmentation or detection box. Say this plainly if asked how precise the "localization" is.
- The dataset is a public Kaggle dataset of real railway track photos, not self-collected field data.
- Test accuracy and all other numbers should be reported exactly as the notebook prints them for whatever run is being submitted — no rounding up, no cherry-picking without disclosure. If model selection across multiple training seeds is used (picking the run with the best *validation* accuracy, not test accuracy), that should be stated explicitly, since it's a legitimate but non-default choice worth being upfront about.
- The inspection-priority score (§3b) should not be presented as validated against real inspector judgments — it hasn't been. It's a defensible, citable construction, not an evaluated operational tool.

## 6. Full reference list

1. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017*.
2. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.
3. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *ICML 2016*.
4. Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? *NeurIPS 2017*.
5. Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011). Bayesian Active Learning for Classification and Preference Learning (BALD). *arXiv:1112.5745*.
6. Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. *NeurIPS 2018*.
7. Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. *NeurIPS 2017*.
8. Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity Checks for Saliency Maps. *NeurIPS 2018*.
9. Gungor, M. A., & Gencol, K. (2026). Optimized Transfer Learning Models for Rail Surface Defect Classification with Explainable AI Validation. *Engineering, Technology & Applied Science Research*, 16(4), 38037–38044.
10. Phaphuangwittayakul, A., Harnpornchai, N., Ying, F., & Zhang, J. (2024). RailTrack-DaViT: A Vision Transformer-Based Approach for Automated Railway Track Defect Detection. *Journal of Imaging*.
11. Andéol, L., Fel, T., de Grancey, F., & Mossina, L. (2023). Conformal Prediction for Trustworthy Detection of Railway Signals. *AAAI 2023 Workshop*.

*All citations above were checked directly (abstract/full-text where accessible) at the time this document was written, rather than taken from memory alone, specifically to avoid presenting an unverified or fabricated reference. If any citation details above later turn out to be wrong, that's worth re-checking before final submission rather than trusting this document blindly.*
