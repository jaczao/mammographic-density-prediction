# Mammographic Density Prediction

Classifies breast tissue density into the four BI-RADS classes (A to D) from screening mammograms, then tests how performance holds across patient and scanner subgroups and inspects the learned features.

Coursework for Machine Learning for Imaging, MSc Computing (AI & ML), Imperial College London, February 2026. Joint work with Zachary Alfalahi.

## Results

The final model, an ImageNet-pretrained ConvNeXt-Tiny with an MLP head, reaches a macro AUC of 0.921 on the 5,865 test images (0.934 on validation) and an accuracy of 0.72. Recall is between 0.67 and 0.77 for every class, including the rare extremes: A and D are 10% and 5% of the training images.

Backbones compared, each with an MLP head:

| Backbone | AUC |
|---|---|
| ResNet18 | 0.91 |
| EfficientNet-B1 | 0.915 |
| EfficientNet-V2-S | 0.916 |
| DenseNet121 | 0.917 |
| **ConvNeXt-Tiny** | **0.921** |

## Subgroup analysis

Test-set macro AUC by subgroup:

- **Stable** across laterality (left 0.920, right 0.923), view (CC 0.923, MLO 0.920) and the five most common scanners (0.907 to 0.925).
- **Lowest** with breast implants: 0.822 (287 images), against 0.924 without.
- **Highest** for patients aged 70 and over: 0.943 (904 images); the other age bands score 0.911 to 0.917.

## Model inspection

The 768-dimensional ConvNeXt features of the test images are reduced with PCA (84 components keep 95% of the variance) and projected to 2D with t-SNE and UMAP. The projections are coloured by true and predicted density, prediction correctness, view, laterality, age, scanner, implants and cancer status, to look for confounders.

## Method

- **Data:** RSNA Screening Mammography Breast Cancer Detection, downsampled by the course to 128×96 pixels: 21,204 training, 2,368 validation and 5,865 test images.
- **Preprocessing:** the breast is segmented as the largest connected component.
- **Augmentation:** brightness (±15%) and contrast (±20%) jitter, Gaussian blur, random affine (rotation ±12°, translation ±8%, scale ±10%) and horizontal flips.
- **Training:** AdamW with the backbone at 5% of the head's learning rate, a one-cycle cosine schedule, label smoothing 0.1, class-balanced sampling, and early stopping and checkpoint selection on validation AUC.

## Usage

```bash
pip install -r requirements.txt
jupyter lab notebook.ipynb
```

The notebook downloads the preprocessed dataset (about 30,000 images) on first run. The saved outputs come from a run on one NVIDIA RTX 4080.
