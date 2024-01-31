# Lightweight Neural Network for Bone Metastasis Detection
Lightweight neural network for binary classification of cancer metastasis in bone scan images using PyTorch. Code repository for the study entitled "A Lightweight Convolutional Neural Network for Detection of Osseous Metastasis using Feature Fusion and Attention Strategies".

### [Important]
This study is currently ongoing. Documentation and notebooks may be incomplete at this point in time.

### MobileLookNet: A Novel Lightweight CNN
This study introduces a novel lightweight architecture called MobileLookNet, which achieves better performance than existing state-of-the-art CNNs despite having fewer FLOPs and parameters. MobileLookNet exhibits ~1/3 of MobileNet's FLOPs, ~1/30 of ResNet's FLOPS, ~1/10 of MobileNet's parameters, and ~1/100 of ResNet's parameters. MobileLookNet surpasses MobileNet and ResNet in accuracy, precision, F1-score, and sensitivity.

---

### Models
This repository contains the following:
- MobileLookNet - variations of MobileLookNet with feature fusion and attention strategies
- Base CNN - A base/vanilla CNN proposed in [Analysis of Batch Size in the Assessment of Bone Metastasis from Bone Scans in Various Convolutional Neural Networks](https://link.springer.com/chapter/10.1007/978-981-99-3068-5_20). Compared against the performance and computational load of MobileLookNet.
- LB-FCNN - A lightweight CNN proposed in . Compared against the performance and computational load of MobileLookNet.
- MobileNetv2 - A lightweight state-of-the-art CNN developed for mobile systems proposed in [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381). Compared against the performance and computational load of MobileLookNet.
- ResNet - A residual neural network proposed in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). Compared against the performance and computational load of MobileLookNet.

---

### Setup
1. Create your virtual environment using `environment.yml`.
2. Unzip and place your dataset as specified in the Dataset section below.

---

### Dataset
For privacy reasons, the dataset used in this study is limited to research collaborators. To use the notebooks here with your own dataset, unzip and place your dataset/s in your folder as follows:
```
root
├── .gitignore  
├── README.md  
├── dataset-sample
├── dataset
├── base-cnn.ipynb
└── ...
```
Your dataset should be a folder containing .tif images, where each .tif image corresponds to one bone scan. Each image filename should follow the format `XXXX-Y-Z`: 
- `XXXX` corresponds to the image index*
- `Y` corresponds to either 0 or 1, with 0 indicating no metastasis and 1 indicating with metastasis
- `Z` corresponds to either A or P, with A indicating anterior bone scan and P indicating posterior bone scan*
*These parts are ignored in the code, but are helpful for holding additional information about specific images.

Alternatively, if you would like to request access to the dataset used in this study, please contact alive.sose@ateneo.edu.

---

### Steps for Training
When training a model for the first time, do the following:
1. Change `data_dir = r"dataset-sample-flat"` to the name of your dataset. It is recommended to have two datasets, one being your actual dataset and another being a smaller sample that you can use to first test if the notebook is set up correctly on your machine.
2. Run the notebook on your sample dataset and see if it runs to the very end.
3. If it runs to the very end, change `data_dir = r"dataset-sample-flat"` to the filename of your actual dataset (ex. `data_dir = r"dataset-augmented-flat"`)
4. Change hyperparameters, number of folds for crossfold validation (`k_folds = 3`), and number of epochs (`epochs = 2`) as you see fit.
5. Run the model on your full dataset.

**Note**: If using CUDA and you have already run the notebook once then try to run again, it may error on the cell with `summary(model, (C, H, W))`. To fix this, simply restart the kernel and run all cells to reset the tensor state from the previous run.
