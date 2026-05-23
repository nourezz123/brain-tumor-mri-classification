# 🧠 Brain Tumor Extraction and Classification from MRI Images

> A complete end-to-end automated system combining deep learning and classical image processing for brain tumor detection and classification from MRI scans — achieving **92.14% accuracy**, **AUC-ROC of 0.9888**, and **Kappa coefficient of 0.8952** on a 4-class dataset of 7,023 MRI images.

---

## 👥 Author
Nour Ezz 
---

## 🏆 Key Results

| Classifier | Accuracy | Sensitivity | Specificity | Precision | F1-Score | AUC-ROC | Kappa |
|-----------|---------|------------|------------|---------|---------|--------|------|
| **SVM (RBF) 🏆** | **92.14%** | **92.09%** | **97.38%** | **92.21%** | **92.09%** | **0.9888** | **0.8952** |
| KNN (k=5) | 88.84% | 88.76% | 96.28% | 88.91% | 88.76% | 0.9749 | 0.8512 |
| Random Forest | 86.34% | 86.13% | 95.38% | 86.55% | 86.13% | 0.9731 | 0.8179 |
| KNN (Manual ⭐) | ~85.0% | — | — | — | — | — | — |

---

## 📋 Project Overview

Brain tumors are among the most dangerous neurological conditions worldwide. Early and accurate detection is critical for improving patient survival rates. Magnetic Resonance Imaging (MRI) is the gold-standard non-invasive modality for brain tumor diagnosis, but raw MRI images suffer from low contrast, noise, and intensity non-uniformity — making manual interpretation time-consuming and subject to inter-observer variability.

This project presents a complete two-phase automated Computer-Aided Detection (CAD) system:

**Phase I** develops a 9-stage preprocessing pipeline that enhances MRI image quality through spatial domain enhancement, frequency domain filtering, morphological operations, and edge detection — with two algorithms implemented entirely from scratch without any built-in functions.

**Phase II** implements a hybrid feature extraction framework combining deep CNN features from four pre-trained architectures (VGG16, VGG19, ResNet50, EfficientNetB0) with three manually implemented handcrafted descriptors (LBP, GLCM, HOG). Features are fused, reduced using PCA, and classified using SVM, Random Forest, and KNN — with a fully manual KNN implementation from scratch.

The system classifies MRI scans into 4 categories:

- 🔴 **Glioma** — Most aggressive malignant brain tumor (Grade III–IV)
- 🟠 **Meningioma** — Slower growing, usually benign (Grade I–II)
- 🔵 **Pituitary Tumor** — Benign tumor on pituitary gland (Grade I)
- 🟢 **No Tumor** — Healthy brain scan (control class)

---

## 🗂️ Repository Structure

```
brain-tumor-mri-classification/
│
├── Full_BrainTumor_Project.ipynb            # Complete notebook — Phase I & Phase II
├── Brain_Tumor_Research_Paper.pdf
└── README.md                               
```

---

## 📁 Dataset

We use the **Brain Tumor MRI Dataset** published by Masoud Nickparvar on Kaggle (2021).

🔗 **Download:** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The dataset contains 7,023 MRI brain images in JPEG format across 4 classes, pre-organized into Training and Testing subsets. All images are coronal or axial T1-weighted MRI scans.

| Class | Training Images | Testing Images | Tumor Type | Severity |
|-------|----------------|---------------|-----------|---------|
| Glioma | 1,321 | 300 | Primary malignant CNS | High |
| Meningioma | 1,339 | 306 | Meningeal | Moderate |
| Pituitary | 1,457 | 300 | Pituitary gland | Low |
| No Tumor | 1,595 | 405 | Healthy brain | — |
| **Total** | **5,712** | **1,311** | — | — |

After downloading, extract and organize the dataset in this structure:

```
Dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Place the `Dataset/` folder in the same directory as the notebook.

---

## ⚙️ Installation & Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
```

### Step 2 — Install required libraries

```bash
pip install tensorflow opencv-python scikit-learn scikit-image matplotlib numpy seaborn pandas
```

Individual installs if needed:

```bash
pip install tensorflow
pip install opencv-python
pip install scikit-learn
pip install scikit-image
pip install matplotlib
pip install numpy
pip install seaborn
pip install pandas
```

### Step 3 — Download and place the dataset

Download the dataset from Kaggle using the link above and place it in the `Dataset/` folder as shown in the structure above.

### Step 4 — Open the notebook

```bash
jupyter notebook Full_BrainTumor_Project.ipynb
```

### Step 5 — Update the dataset path

In **Cell 1** of the notebook, update the path to match your local setup:

```python
DATASET_PATH = "Dataset/Training"
```

### Step 6 — Run all cells

Run all cells from top to bottom in order. Phase I cells (1–14) run quickly. Phase II cells (15–29) take longer due to CNN feature extraction — Cell 18 may take 10–20 minutes depending on your hardware.

---

## 🔬 Phase I — Preprocessing Pipeline

A 9-stage preprocessing pipeline was designed and implemented to maximize tumor region visibility, reduce acquisition noise, and enhance structural boundaries before feature extraction. Two algorithms were implemented entirely from scratch without any built-in image processing functions, as required.

| Stage | Technique | Implementation |
|-------|-----------|---------------|
| 1 | Grayscale Conversion | OpenCV |
| 2 | Histogram Equalization | ⭐ Manual — from scratch |
| 3 | CLAHE (Adaptive Local Enhancement) | ⭐ Manual — from scratch |
| 4 | Contrast Stretching | NumPy |
| 5 | Gaussian & Median Noise Removal | OpenCV |
| 6 | Image Sharpening (Unsharp Masking) | NumPy + OpenCV |
| 7 | Morphological Operations | OpenCV |
| 8 | Edge Detection (Sobel, Canny, Laplacian) | OpenCV |
| 9 | FFT Low-pass & High-pass Filtering | NumPy FFT |

> ⭐ = Implemented manually without cv2.equalizeHist, cv2.createCLAHE, or any equivalent built-in function

### Manual Histogram Equalization

Implemented using 4 steps from scratch: compute 256-bin histogram → compute CDF → normalize CDF using the formula `T(r) = round((CDF(r) - CDF_min) / (N - CDF_min) × 255)` → apply mapping to remap every pixel. Redistributes pixel intensities across the full dynamic range to dramatically improve contrast in low-contrast MRI images.

### Manual CLAHE

Divides the image into 8×8 non-overlapping tiles. Applies histogram equalization independently to each tile. A clip limit of 2.0 caps the histogram in each tile before equalization, redistributing excess uniformly to prevent noise over-amplification. Produces superior local contrast adaptation compared to global equalization, particularly effective for MRI images with spatially varying intensity distributions.

### FFT Frequency Domain Filtering

Applies 2D Fast Fourier Transform, centers the spectrum, and applies a circular mask. Low-pass filter (radius=40px) keeps the center — preserving smooth brain structure and removing high-frequency noise. High-pass filter (inverted mask) keeps only the edges and fine structural detail. Both filtered images are reconstructed via Inverse FFT.

---

## 🤖 Phase II — Feature Extraction, Fusion & Classification

### CNN Feature Extraction — Transfer Learning

Four pre-trained CNN architectures were used as feature extractors. Each model was loaded with ImageNet weights, the classification head was removed, and a Global Average Pooling layer was added to produce a compact fixed-length feature vector per image.

| CNN Model | Feature Dimensions | SVM Accuracy | Notes |
|-----------|-------------------|-------------|-------|
| VGG16 | 512-D | 88.3% | Reliable baseline |
| VGG19 | 512-D | 89.1% | Slight improvement over VGG16 |
| EfficientNetB0 | 1,280-D | 90.4% | Parameter-efficient |
| **ResNet50** | **2,048-D** | **91.2%** | **Best — selected for fusion** |

ResNet50 was selected for feature fusion due to its highest feature variance and accuracy. Its residual (skip) connections allow very deep feature learning without vanishing gradients, producing the richest and most informative feature representations.

### Handcrafted Feature Extraction — Manual Implementation ⭐

Three handcrafted feature descriptors were implemented and extracted:

**Manual LBP (Local Binary Pattern) ⭐**
Implemented entirely from scratch without skimage.feature.local_binary_pattern. For each pixel, compares its 8 surrounding neighbors clockwise — if neighbor ≥ center pixel the bit is 1, otherwise 0 — producing an 8-bit binary code (0–255) per pixel. The normalized 256-bin histogram of LBP codes is the feature vector. Captures micro-texture patterns that distinguish tumor tissue from healthy brain matter.

**GLCM (Gray Level Co-occurrence Matrix)**
Computed at 4 angles (0°, 45°, 90°, 135°) with displacement distance d=1. Extracts 4 statistical properties per angle: contrast, correlation, energy, and homogeneity — producing a 16-dimensional feature vector characterizing spatial pixel intensity relationships.

**HOG (Histogram of Oriented Gradients)**
Computed using 8×8 pixel cells, 2×2 cell blocks, 9 orientation bins spanning 0–180°. Gradient magnitudes and orientations computed via Sobel derivatives. Block-normalized using L2 normalization. Captures local shape and edge structure of tumor regions.

### Feature Fusion

All features are normalized with StandardScaler then concatenated:

```
ResNet50 (2,048-D) + LBP (256-D) + GLCM (16-D) + HOG (500-D) = ~2,820-D fused vector
```

Normalization ensures no single feature group dominates due to scale differences. CNN features capture high-level semantic content while handcrafted features capture low-level texture and shape — they are complementary and together achieve higher accuracy than either alone.

### PCA Dimensionality Reduction

Principal Component Analysis (PCA) reduces the ~2,820-D fused vector to 573 dimensions while retaining 95% of the total explained variance — a 79.7% reduction. Removes redundant and noisy dimensions while preserving the most discriminative information.

```
Input:  ~2,820 dimensions
Output:   573 dimensions
Variance retained: 95%
Reduction: 79.7%
```

### Manual KNN Implementation ⭐

K-Nearest Neighbors implemented entirely from scratch without sklearn.neighbors.KNeighborsClassifier:

```python
For each test image:
1. Compute Euclidean distance to every training image
   distance = sqrt(sum((f_test - f_train)²))
2. Sort all distances ascending
3. Take the k=5 nearest neighbors
4. Majority vote of their labels → predicted class
```

### Classification Models

Three classifiers trained and evaluated on PCA-reduced features with 80/20 stratified train/test split:

**SVM with RBF Kernel (C=10, gamma='scale')** — Finds the optimal maximum-margin hyperplane in the 573-dimensional feature space using the RBF kernel to map data to a higher-dimensional space where classes become linearly separable. Best performer at 92.14% accuracy.

**Random Forest (200 trees)** — Ensemble of 200 decision trees using bagging and random feature selection. Robust to overfitting through averaging but slightly weaker in very high-dimensional spaces.

**KNN (k=5, Euclidean distance)** — Non-parametric classifier based on majority vote of 5 nearest neighbors. Competitive at 88.84% accuracy.

---

## 🏗️ System Architecture

```
MRI Input (224×224 px)
         │
         ▼
┌─────────────────────────────────────────┐
│         Phase I — Preprocessing         │
│  CLAHE → Median Filter → Sharpening    │
│  → Morphological Ops → FFT Low-pass   │
└─────────────────────────────────────────┘
         │
         ▼
┌──────────────────────┐  ┌──────────────────────┐
│  CNN Feature         │  │  Handcrafted Features │
│  Extraction          │  │  (Manual ⭐)          │
│                      │  │                      │
│  VGG16   → 512-D    │  │  LBP    →  256-D     │
│  VGG19   → 512-D    │  │  GLCM   →   16-D     │
│  ResNet50 → 2048-D  │  │  HOG    →  500-D     │
│  EffNet  → 1280-D   │  │                      │
│                      │  │                      │
│  Best: ResNet50 ✓   │  │  All manual ⭐        │
└──────────────────────┘  └──────────────────────┘
         │                         │
         └──────────┬──────────────┘
                    ▼
         Feature Fusion (~2,820-D)
         StandardScaler normalization
                    │
                    ▼
         PCA Reduction → 573-D
         (95% variance retained)
                    │
                    ▼
    ┌───────────────────────────────┐
    │        Classification         │
    │  SVM (RBF)    → 92.14% 🏆   │
    │  KNN (k=5)    → 88.84%      │
    │  RandomForest → 86.34%      │
    └───────────────────────────────┘
                    │
                    ▼
         Tumor Class Prediction
    Glioma / Meningioma / Pituitary / No Tumor
```

---

## 📓 Notebook Structure

The notebook `Full_BrainTumor_Project.ipynb` is organized as follows:

### Phase I Cells

| Cell | Title | Description |
|------|-------|------------|
| 1 | Imports & Loading | Load libraries and read one image per class |
| 2 | Raw Visualization | Display baseline images before processing |
| 3 | Manual Histogram Equalization ⭐ | Contrast enhancement from scratch |
| 4 | Histogram Analysis | Before/after histogram plots |
| 5 | Contrast Stretching | Linear intensity normalization |
| 6 | Noise Removal | Gaussian and median filter comparison |
| 7 | FFT Low-pass Filter | Frequency domain noise removal |
| 9 | CLAHE ⭐ | Adaptive local enhancement from scratch |
| 10 | Morphological Operations | Erosion, dilation, opening, closing |
| 11 | Edge Detection | Sobel, Canny, Laplacian comparison |
| 12 | FFT High-pass Filter | Frequency domain edge enhancement |
| 13 | Image Sharpening | Unsharp masking |
| 14 | Pipeline Summary | All 12 techniques in one figure |

### Phase II Cells

| Cell | Title | Description |
|------|-------|------------|
| 15 | Install Libraries | TensorFlow, sklearn, scikit-image |
| 16 | Load Full Dataset | All 5,712 training images |
| 17 | Preprocess All Images | Apply Phase I pipeline to full dataset |
| 18 | CNN Feature Extraction | VGG16, VGG19, ResNet50, EfficientNetB0 |
| 19 | Handcrafted Features ⭐ | Manual LBP + GLCM + HOG |
| 20 | Feature Fusion | Normalize and concatenate all features |
| 21 | PCA Reduction | Compress to 573 dimensions |
| 22 | Train/Test Split | 80/20 stratified split |
| 23 | Manual KNN ⭐ | KNN from scratch |
| 24 | SVM + RF + KNN | Train and evaluate all classifiers |
| 25 | Results Table | All 8 metrics for all classifiers |
| 26 | Confusion Matrices | Visual confusion matrix per classifier |
| 27 | CNN Comparison | Bar chart comparing 4 CNN architectures |
| 28 | Classifier Comparison | Accuracy vs F1 grouped bar chart |
| 29 | Final Summary | Complete results printout |

---

## 📚 Related Work

This project is informed by 15 published studies in the field of brain tumor MRI analysis:

1. Deepak & Ameer (2019) — GoogleNet + SVM — 98% accuracy (3-class)
2. Cheng et al. (2015) — Bag-of-words + saliency features — 91.28%
3. Afshar et al. (2018) — Capsule Networks (CapsNet) for tumor classification
4. Sultan et al. (2019) — Deep CNN multi-grade classification — 96.13%
5. Sajjad et al. (2019) — CNN + data augmentation — 94.58%
6. Swati et al. (2019) — VGG19 block-wise fine-tuning — 94.82%
7. Zhao et al. (2018) — FCNNs + CRFs for segmentation
8. Pereira et al. (2016) — Deep CNNs for BraTS segmentation
9. Havaei et al. (2017) — Two-phase CNN with local and global context
10. Bakas et al. (2018) — BraTS 2018 benchmark dataset and challenge
11. Tandel et al. (2019) — Systematic review of ML/DL for brain tumor classification
12. Kumar et al. (2021) — ResNet50 + LBP + GLCM hybrid — 93.5% (4-class)
13. Ari & Hanbay (2018) — Deep CNN comparison for brain disease classification
14. Ghassemi et al. (2020) — CNN + SVM with GAN pre-training
15. Rehman et al. (2020) — VGG16/VGG19 evaluation — up to 98.69%

Full references with DOIs are available in the research paper.

---

## 🔮 Future Work

- Fine-tune CNN architectures end-to-end on the target MRI dataset to surpass 95% accuracy
- Incorporate multi-modal MRI inputs (T1, T1ce, T2, FLAIR) using 3D volumetric CNNs such as 3D-ResNet or 3D U-Net
- Apply Vision Transformer (ViT) and Swin Transformer architectures for long-range spatial dependency modeling
- Develop pixel-level tumor segmentation using U-Net or Mask R-CNN for precise boundary delineation
- Implement Grad-CAM explainability to highlight image regions driving classification decisions for clinical validation

---

## 📄 Documents

| Document | Description |
|----------|------------|
| `Full_BrainTumor_Project.ipynb` | Complete implementation code — Phase I and Phase II |
| `Complete_Brain_Tumor_Research_Paper.docx` | Full research paper covering both phases with methodology, results, 15 related studies, and references |
| `Brain_Tumor_Presentation.pptx` | 15-slide professional presentation for 10-minute project defense |
| `Phase1_BrainTumor_Report.docx` | Standalone Phase I preprocessing report with visual results |

---

## 📄 License

This project is submitted as academic coursework at Egypt-Japan University of Science and Technology (E-JUST).
Intended for educational and research purposes only.

---


> **Best Result: SVM with RBF Kernel — 92.14% Accuracy | 0.9888 AUC-ROC | 0.8952 Kappa**
