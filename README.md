# French Web Domain Classification Challenge

First place solution for the French Web Domain classification challenge of the Altegrad 2020 course
> https://www.kaggle.com/c/fr-domain-classification

- **Author :** Théo Viel
- **Team :** Logistic Regression Baseline 
- **Hardware :** Nvidia RTX 2080 Ti 

## Quick Overview

My solution consists on finetuning the CamemBert architecture and then using the graph structure for ensembling. Read the report `Report.pdf` for more information.


## Data

The input data expects to be put in an `input` folder at the root.
You can change the paths to it in the `src/params.py` script


## Reproducing Results

If you do not wish to retrain models, you can directly go to 3. as the extracted features were provided

#### 1. Run the notebook `Prepare Data.ipynb` :
 - with `TRANSLATE = True`
 - with `TRANSLATE = False`
This will generate the `df_texts.csv` and `df_texts_trans.csv` dataframes needed for modelling

#### 2. Run the notebook `Texts Modelling.ipynb`
 - with extract_ft = True 
 - Change the `augment`, `translate` and `avg_pool` parameters depeding on the models you want to train
 - Change the "name" of the `.npy` files to save predictions to
This will generate the `.npy` features to use for the graph approach

#### 3. Run the notebook `Graph.ipynb`
 - You may want to specify the `models` to indicate which features to use
This will generate the submission file 

Although everything was seeded, depending on the device, you might observe about 0.001 CV variability because of PyTorch's weird behaviour. 
This should not affect the final results that much. 

Check the report to see which `augment`, `translate` and `avg_pool` configuration to use to retrain the same models as I did.
Results can be improved by going for the 8 possibilities.


