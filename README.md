# Bone Metastasis LB-FCNN

### If setting up the project for the first time on your local machine
1. Open Anaconda Prompt and cd to the repo
2. Run `conda env create -f environment.yml` to create virtual environment from the environment.yml file
3. Run `conda activate conv_env` to activate the virtual environment
4. Run `jupyter notebook` to open Jupyter Notebook
5. Unzip and place the dataset in your folder as follows:
```
root
├── .gitignore  
├── README.md  
├── dataset-sample-flat
├── dataset-augmented-flat
├── base-cnn.ipynb
└── ...
```

### Training Notes
The following notebooks are the models for training:
```
root
├── base-cnn.ipynb
├── lb-fcnn.ipynb
├── mobilenet-pretrained-frozen.ipynb
├── mobilenet-pretrained-unfrozen.ipynb
├── mobilenet-trained-from-scratch.ipynb
├── resnet-pretrained-frozen.ipynb
├── resnet-pretrained-unfrozen.ipynb
└── resnet-trained-from-scratch.ipynb
```
When training a model, please do the following:
1. Run it first and see if it runs to the very end. It is initially set to run on sample data `(dataset-sample-flat)`
2. If it runs to the very end, change `data_dir = r"dataset-sample-flat"` to `data_dir = r"dataset-augmented-flat"`
3. Change `k_folds = 3` to `k_folds = 10`
4. Change `epochs = 2` to `epochs = 100`
5. Run the model again. This will run the model on the full dataset, amount of folds, and amount of epochs.
