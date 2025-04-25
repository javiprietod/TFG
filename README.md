# TFG
Neural networks are increasingly employed in the banking sector for tasks ranging from credit scoring to fraud detection. Despite their powerful predictive capabilities, the opacity of neural network models poses significant challenges for their interpretability. This project aims to bridge the gap between complex neural network architectures and their practical interpretation by developing a comprehensive methodology utilizing state-of-the-art Explainable AI (XAI) techniques. 

## Algorithm requirements
To be able to use this algorithm, datasets and models need to fulfill some requirements:
* Datasets
    - Cleaned beforehand. A few cleaning security measures are put in place, like droping duplicates, rows that have missing values, ID columns and columns that only have one value; just in case
    - Include the target column

* Model (if necessary)
    - Differentiable model, therefore no trees are accepted.

## How to run the programs
To run the programs, you will need to install the libraries provided in the requirements.txt file. You can do this by running the following command:
``` cmd
pip install -r requirements.txt
```

There are different programs that you can run:
- `train.py`: This file trains a model with the settings in `train.yaml`. This model is composed of layers with activations in between them. To execute it, run the command:
```cmd
python -m src.train
```
- `counterfactual.py`: This file runs the counterfactual system on an specific index of the dataset. To execute it, run the command:
```cmd
python -m src.counterfactual
```
- `interface.py`: This file contains the code for a user interface in which the user can interact with many of the functionalities of the counterfactual system. You can use a model you have previously trained or you can provide the dataset and we will train a basic model for you. To execute it, run the command:
```cmd
streamlit run interface.py
```
and follow the instructions in the terminal.

## Datasets
The datasets used in this repository are:
1. [Loan Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default):

