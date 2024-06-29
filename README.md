# Lightweight Neural Network for Bone Metastasis Detection
Lightweight neural network for binary classification of cancer metastasis in bone scan images using PyTorch. Code repository for ["A Lightweight Convolutional Neural Network for Detection of Osseous Metastasis using Feature Fusion and Attention Strategies"](https://dl.acm.org/doi/10.1145/3663976.3664235).

### MobileLookNet
MobileLookNet is a novel lightweight CNN for osseous metastasis detection in bone scintigrams on resource-constrained devices. The top-performing MobileLookNet variant exhibits superior accuracy, precision, sensitivity, and F1 score to ResNet while achieving a 95.54% reduction in parameters. This is achieved through employing depthwise separable convolutions in parallel, utilizing inverted residuals, and integrating low-level and high-level features, allowing the model to capture diverse levels of abstraction and extract more individually expressive features. MobileLookNet also outperforms traditional bone scintigraphy methods and state-of-the-art networks in metastasis detection while requiring significantly fewer floating-point operations (FLOPs) and parameters.

MobileLookNet is applied to cancer detection through bone metastasis in the hopes of assisting understaffed and underfunded healthcare facilties, which often lack the computing power needed to run existing state-of-the-art models.

![Results](https://github.com/IrishMorales/lightweight-cnn-for-bone-metastasis/raw/main/results/all_scatter.svg "Results")

---

### Models
This repository contains the following:
- MobileLookNet - variations of MobileLookNet with feature fusion and attention strategies
- Base CNN - A base/vanilla CNN proposed in [Analysis of Batch Size in the Assessment of Bone Metastasis from Bone Scans in Various Convolutional Neural Networks](https://link.springer.com/chapter/10.1007/978-981-99-3068-5_20). Compared against the performance and computational load of MobileLookNet.
- LB-FCNN - (Outdated) A lightweight CNN containing early, outdated versions of the model. Kept for documentation.
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

---

### Citation
```
@inproceedings{10.1145/3663976.3664235,
  author = {Morales, Irish Danielle and Echon, Carlo Joseph and Tea\~{n}o, Angelico Ruiz and Alampay, Raphael and Abu, Patricia Angela},
  title = {MobileLookNet: A Lightweight Convolutional Neural Network for Detection of Osseous Metastasis Using Feature Fusion and Attention Strategies},
  year = {2024},
  isbn = {9798400716607},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3663976.3664235},
  doi = {10.1145/3663976.3664235},
  abstract = {This study introduces MobileLookNet, a novel lightweight architecture designed for detecting osseous metastasis in bone scintigrams on resource-constrained devices. By employing depthwise separable convolutions in parallel, utilizing inverted residuals, and integrating low-level and high-level features, MobileLookNet captures diverse levels of abstraction and extracts more individually expressive features. It outperforms traditional bone scintigraphy methods and state-of-the-art networks in metastasis detection while requiring significantly fewer floating-point operations (FLOPs) and parameters. Ablation studies reveal that feature fusion yields superior results compared to transformer-based attention strategies, highlighting the informative nature of low-level features in metastasis detection. Moreover, MobileLookNet demonstrates a trade-off between high accuracy, low FLOPs, and low parameters, where at most two can be achieved at a time. Overall, MobileLookNet shows promise in assisting nuclear medicine practitioners and enhancing metastasis detection in resource-constrained settings.},
  booktitle = {Proceedings of the 2024 2nd Asia Conference on Computer Vision, Image Processing and Pattern Recognition},
  articleno = {55},
  numpages = {6},
  keywords = {bone scintigrams, convolutional neural networks, lightweight networks, metastasis},
  location = {Xiamen, China},
  series = {CVIPPR '24}
}
```
For questions, please contact [asemorales.tech@gmail.com](mailto:asemorales.tech@gmail.com).
