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
├── dataset  
│   ├── test 
│   ├── train
│   ├── val  
│   └── GroundTruth.xlsx
└── ...
```