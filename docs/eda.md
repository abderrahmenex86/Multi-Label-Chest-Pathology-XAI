# Dataset and exploratory analysis

Naming, class balance, co-occurrence, and demographics for NIH ChestX-ray14.


The pipeline trains on the **NIH ChestX-ray14** dataset (112,120 chest radiographs across 30,805 unique patients). Patient overlap between train and validation splits is strictly prevented by splitting on unique Patient IDs.

### Class Distribution

<!-- | Pathology Class | Positive Count | Prevalence (%) | -->

<!-- | :--- | :--- | :--- | -->

<!-- | **Infiltration** | 19,894 | 17.7% | -->

<!-- | **Effusion** | 13,317 | 11.9% | -->

<!-- | **Atelectasis** | 11,559 | 10.3% | -->

<!-- | **Nodule** | 6,331 | 5.6% | -->

<!-- | **Mass** | 5,782 | 5.2% | -->

<!-- | **Pneumothorax** | 5,302 | 4.7% | -->

<!-- | **Consolidation** | 4,667 | 4.2% | -->

<!-- | **Pleural Thickening** | 3,385 | 3.0% | -->

<!-- | **Cardiomegaly** | 2,776 | 2.5% | -->

<!-- | **Emphysema** | 2,516 | 2.2% | -->

<!-- | **Edema** | 2,303 | 2.1% | -->

<!-- | **Fibrosis** | 1,686 | 1.5% | -->

<!-- | **Pneumonia** | 1,431 | 1.3% | -->

<!-- | **Hernia** | 227 | 0.2% | -->

![Class Distribution](figs/eda_class_distribution.png)

### Co-Occurrence Matrix

The diagonal of the $14 \\times 14$ co-occurrence matrix isolates **solitary diagnoses** (pathologies diagnosed alone), while off-diagonal elements measure overlapping multi-label co-occurrences.

![Co-Occurrence Heatmap](figs/eda_co_occurrence_heatmap.png)

### Demographics & View Positions

![Demographics and Views](figs/eda_demographics_and_views.png)

______________________________________________________________________
