## Explainable AI for Autism Diagnosis: Identifying Critical Brain Regions Using fMRI Data

#### Abstract

Early diagnosis and intervention for Autism Spectrum Disorder (ASD) has been shown to significantly improve the quality of life of autistic individuals. However, diagnostics methods for ASD rely on assessments based on clinical presentation that are prone to bias and can be challenging to arrive at an early diagnosis. There is a need for objective biomarkers of ASD which can help improve diagnostic accuracy. Deep learning (DL) has achieved outstanding performance in diagnosing diseases and conditions from medical imaging data. Extensive research has been conducted on creating models that classify ASD using resting-state functional Magnetic Resonance Imaging (fMRI) data. However, existing models lack interpretability. This research aims to improve the accuracy and interpretability of ASD diagnosis by creating a DL model that can not only accurately classify ASD but also provide explainable insights into its working. The dataset used is a preprocessed version of the Autism Brain Imaging Data Exchange (ABIDE) with 884 samples. Our findings show a model that can accurately classify ASD and highlight critical brain regions differing between ASD and typical controls, with potential implications for early diagnosis and understanding of the neural basis of ASD. These findings are validated by studies in the literature that use different datasets and modalities, confirming that the model actually learned characteristics of ASD and not just the dataset. This study advances the field of explainable AI in medical imaging by providing a robust and interpretable model, thereby contributing to a future with objective and reliable ASD diagnostics.

#### Preprint

https://arxiv.org/abs/2409.15374

#### Paper

The Lancet eClinicalMedicine, Aug 2025 - [DOI](https://doi.org/10.1016/j.eclinm.2025.103452)

## Summary of changes

- 2026-02-27 — Commit 9af713d: "leak consider" — noted and considered potential data-leakage issues in preprocessing and model evaluation; follow-up review recommended.

**Details for `app/main.py` (commit 9af713d)**

- Purpose: Address potential data-leakage and numerical-stability issues in training/evaluation.
- Prevented data leakage:
	- Feature extraction is performed per cross-validation fold (train vs test) instead of globally.
	- RFE is fitted only on the training fold via `fit_rfe_on_fold` and applied to validation and test sets.
	- `StandardScaler` is fit on training features and then applied to validation/test sets.
	- Validation splits are drawn from the training fold (no use of test data for validation).
- Numerical stability:
	- Fisher Z transform clipping added (clip coefficients to ±0.9999) to avoid infinities during transformation.
- Training/evaluation improvements:
	- `rfe_step` parameter exposed to configure RFE step-size.
	- `EarlyStopping` records and restores the best model state when triggered.
	- Encoding and fine-tuning stages use explicit encode/decode loaders with early stopping per stage.

## Results

### Model Performance (Without Data Leakage)
The following results are from models trained with proper data leakage prevention measures (per-fold feature extraction, RFE, and scaling):

| Metric | Mean ± Std |
|--------|-----------|
| Accuracy | 65.02% ± 2.51% |
| Sensitivity | 0.6585 ± 0.0393 |
| Specificity | 0.6416 ± 0.0503 |
| Precision | 0.6597 ± 0.0304 |
| F1 Score | 0.6582 ± 0.0255 |

### Explainability Results

#### CCS Pipeline - Top Important Brain Regions

**SHAP Method:**
- Top: Vermis_9 ↔ Cerebelum_8_L (Cerebellar-sensorimotor function) - Importance: 10.0
- Angular_R ↔ Angular_L (Complex language processing) - Importance: 7.65

**LIME Method:**
- Top: Temporal_Sup_R ↔ Thalamus_R (Auditory-sensory relay) - Importance: 10.0
- Cerebelum_3_L ↔ Temporal_Inf_L (Cerebellar-auditory processing) - Importance: 7.14

**GuidedBackprop Method:**
- Top: Temporal_Pole_Sup_R ↔ Caudate_L (Emotional-learning) - Importance: 10.0
- Temporal_Mid_R ↔ Putamen_R (Auditory-movement regulation) - Importance: 7.38

**Integrated Gradients Method:**
- Top: Frontal_Mid_L ↔ Frontal_Sup_L (Planning-cognition) - Importance: 10.0
- Occipital_Sup_R ↔ Calcarine_L (Visual processing) - Importance: 8.37

#### DPARSF Pipeline - Top Important Brain Regions

**SHAP Method:**
- Top: Temporal_Pole_Sup_R ↔ Rolandic_Oper_L (Emotional-sensory integration) - Importance: 10.0
- Cingulum_Ant_R ↔ Frontal_Mid_Orb_R (Decision-emotion regulation) - Importance: 7.97

**LIME Method:**
- Top: Temporal_Mid_R ↔ Cuneus_L (Auditory-visual perception) - Importance: 10.0
- Calcarine_R ↔ Rectus_L (Visual-emotion processing) - Importance: 9.34

**GuidedBackprop Method:**
- Top: Cerebelum_10_R ↔ Parietal_Inf_R (Cerebellar-perception integration) - Importance: 10.0
- Temporal_Mid_R ↔ Pallidum_L (Auditory-movement execution) - Importance: 9.44

**Integrated Gradients Method:**
- Top: Frontal_Mid_L ↔ Frontal_Sup_L (Planning-cognition) - Importance: 10.0
- Cuneus_L ↔ Calcarine_L (Visual perception) - Importance: 9.88

### Comparison: Data Leakage vs No Leakage
Models trained with data leakage showed artificially inflated performance (~97-98% accuracy) compared to properly validated models (~65% accuracy), highlighting the critical importance of proper cross-validation methodology in medical imaging studies.
