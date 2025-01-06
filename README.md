# Olist Machine Learning Project

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

The model built is a churn propensity model, which means it will provide a score indicating the probability of an Olist seller stopping their sales in the next six months.

## Overview

This is a Kedro project with Kedro-Viz setup, which was generated using `kedro 0.19.10`. The project consists of six pipelines, each performing specific functions. Below is a list of the pipelines and their respective objectives:

- **ingesting**: Handles data ingestion and schema normalization. 
- **preprocessing**: Performs data cleaning and preprocessing for the datasets used in the construction of model variables.
- **audience_building**: it separates the scoring or modeling audience considering the adopted business rules.
- **feature_engineering**: Builds the model variables. The datasets are at the model's granularity, i.e., `seller id` in the reference `cohort`. Furthermore, it constructs the inference table with the calculated variables and the spine table with the target variable.
- **modeling**: Creates the ABT (Analytical Base Table) and also trains and validates the model.
- **scoring**: Executes scoring with the trained model based on the inference table created in the feature_engineering pipeline.
- **monitoring**: Creates data and visualizations for monitoring the machine learning model. This monitoring is divided into three types: monitoring the model's input and output variables (lag M-0); monitoring the impact of variables using SHAP values (lag M-0); and monitoring modeling metrics and the relationship between input and output variables and target (lag M-6).
## Datasets
The datasets used are those publicly available at [olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). More information about this data can be obtained through the previously indicated link.

## Pre modeling definitions
- **Audience**: considering a reference month, the model's audience was defined as sellers who had at least one valid order (`order_status="delivered"`) in the last 9 months.
- **Feature engineering**: for each dataset, the variables were created considering a historical period of 9 months with time windows varying in 3-month intervals. In other words, variables were created with time windows of 3, 6, and 9 months. The definition on this 9-months historical peridos basically rely on the historic available on the `order` dataset and the volume necessary to proceed model development and validation.
- **Target definition**: Churn was defined as a seller who, starting from a given reference month, fails to make any sales over the next 6 months, this is, the defined performance period is 6 months. The target variable was marked when this event occurred for any seller within the audience. More details about the target definition can be found at `notebooks/target_definition.ipynb`.

## Modeling details

### ABT and division of training, validation and test sets:
- The created Analytical Base Table (ABT) had predictor variables and target variables from the months 10-2017 to 03-2018.
- The historical period from 10-2017 to 02-2018 (~80% of the ABT's volume) was used for the training/validation set and the stratified cross-validation technique with 5 folds was used for validation.
- The last month for which the target was available, i.e. 03-2018, was used for the test set (~20% of the ABT's volume). This test set was not used to make decisions about choosing models, resources or optimizing hyperparameters, but only for the final evaluation of the models. The most recent month available was purposely used because it is understood that this is the set closest to the one we will have when the model is put into production.

### Model development steps:
- **Model selection**: in this step, the machine learning algorithm was selected from experiments with different models and preprocessing techniques. The file `notebooks/Modeling/model_selection.ipynb` contains more detail about the experiments performed.
- **Feature selection**: in this step, a selection of variables was performed. The initial set had 370 features and several different successive filters were applied to obtain 10 restricted sets ranging from 7 to 30 variables, thus altering the complexity of the model. More details about the processes used can be found in the file `notebooks/Modeling/feature_selection.ipynb`.
- **Hyperparameter optimization**: from the sets of variables obtained in the previous step, the hyperparameter optimization of each of these sets was performed. The hyperparameter search process was performed with Optuna. For details about the procedure used, see the file `notebooks/Modeling/hyperparameters_tuning.ipynb`.

### Experiment Tracking 
MLflow was used as the experiment tracking tool of this project. To consult the results of the experiments performed at each modeling stage, simply start the local server using the command `make start_mlflow_server`.

# Getting Started
## Pré requesitos
To execute the scoring process completely, it is necessary to have the access credentials to the data lake saved in `conf/local/credentials.yml` to access the raw datasets used in the scoring process.

## Dependências
The dependencies required for executing the scoring process are listed in the `requirements` folder and can be installed using the command.

```console
pip install -r requirements/dev
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```
There are two types of scoring that can be performed:

1) The first type is batch scoring, which executes the process for the cohort immediately following the most recent cohort in the billing dataset. For example, if the latest cohort available in the billing dataset is 202312, features can be created up to cohort 202401 since the model's features use information from cohorts prior to the scoring cohort. Additionally, scoring and target creation are performed for the last cohort where this is possible, considering the performance period defined in the modeling stage. For example, if the performance period is three months, the last cohort available for target creation would be 202309. In summary, batch scoring always processes two cohorts: the most recent cohort available in the billing dataset and the last cohort available for target creation, with the latter being used to calculate the model's performance. This is the default behavior of scoring in Kedro and can be executed using the following command: 
    ```console
    kedro run --tags=scoring
    ```

    If the data ingested from the data lake has not changed since the last scoring and you wish to re-run the scoring for any reason, it can be executed without ingestion and preprocessing using the following command:
    ```console
    kedro run --tags=scoring-without-preprocess
    ```

2) The second type is backfill scoring, which processes a range of cohorts defined by the parameters start_cohort and final_cohort. This can be done by setting values for the parameters scoring.start_cohort and scoring.final_cohort, which are expected to be integers in the YYYYmm format to indicate the start and end cohorts for scoring. The command used for this purpose is: 
    ```console
    kedro run --tags=scoring --params=scoring.start_cohort=YYYYmm,scoring.final_cohort=YYYYmm
    ```

    Alternatively:
    ```console
    kedro run --tags=scoring-without-preprocess --params=scoring.start_cohort=YYYYmm,scoring.final_cohort=YYYYmm
    ```

    The modeling pipeline can be executed using a command similar to those for scoring, but with the --tag changed to modeling. It is worth emphasizing once again that this pipeline should not be used for automatic retraining, for the reasons mentioned earlier. The command for execution is: 
    ```console
    kedro run --tags=modeling
    ```

    Like scoring, modeling can be executed without the data ingestion and preprocessing steps:
    ```console
    kedro run --tags=modeling-without-preprocess
    ```

    Additionally, modeling can be executed without updating the ABT, meaning the ingestion, preprocessing, and feature construction steps are also skipped. In practice, this will execute the train-validation split, model training, and the creation of the training and validation datasets with the results. This procedure can be executed using the command:
    ```console
    kedro run --tags=modeling-without-abt-update
    ```

# Future enhancements
- Create feature_selection, hyperparameters optimization and binning optmization nodes
- Put data in S3
- MLflow artifact store in S3 and backend store in RDS or other
- Containerize the application
- Use of docstring in functions nodes to document the code
- refactorate monitoring codes