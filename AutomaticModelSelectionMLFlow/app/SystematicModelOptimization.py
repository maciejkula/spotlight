# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 10:40:20 2022

@author: Thorsten Kalb

This program will download the dataset, systematically search for the best model 
architecture and parameters, save its test metrics and save the best, trained model
"""
import sys
# sys.path.append("./spotlight")
from spotlight.cross_validation import user_based_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import sequence_mrr_score
from spotlight.sequence.implicit import ImplicitSequenceModel
import numpy as np
import mlflow
from mlflow import log_metric, log_param, log_artifact
import torch
from mlflow.tracking import MlflowClient



def get_best_params(experiment_id):
    """requires an experiment_id and gives the parameters of the best run of this experiment as a dictionary"""
    client = MlflowClient()
    runs = mlflow.search_runs([experiment_id], order_by=["metrics.mean_invmrr ASC","metrics.mean_mrr DESC"])
    best_run_id = runs.loc[0,'run_id']
    best_run = client.get_run(best_run_id)
    params=best_run.data.params
    return params

def report_test_mrr(experiment_id, train_valid, test):
    """requires an existing experiment_id, and train_valid and test SequenceInteraction
    instances of spotlight"""
    params = get_best_params(experiment_id)
    print("Optimal parameters obtained succesfully. Start training now.")
    n_iter, model_type, loss = params["n_iter"], params["model_type"], params["loss"]
    n_iter = int(n_iter)
    model = ImplicitSequenceModel(n_iter=n_iter, #10 #5 #3 #50
                                  representation=model_type,
                                  loss=loss)
    # train the model with given parameters on train and validation
    model.fit(train_valid)
    test_mrr = sequence_mrr_score(model, test)
    return test_mrr

def invert_mrr(mrr):
    """inverts the array of mrr scores. Can be interpreted as:
        Index of first relevant item"""
    invmrr = np.zeros(mrr.shape)
    invmrr[mrr>1e-12] = 1/mrr[mrr>1e-12]
    return invmrr

dataset_name = '100K'     ### update this line to say from WHEN the data is!
dataset = get_movielens_dataset(dataset_name)
mlflow.set_tracking_uri('./mlruns')

# split data
train_valid, test = user_based_train_test_split(dataset)
train, validation = user_based_train_test_split(train_valid)

train = train.to_sequence()
validation = validation.to_sequence()
test = test.to_sequence()
train_valid = train_valid.to_sequence()
dataset = dataset.to_sequence()

# create new experiment or continue the previous one.
experiment_name = "Spotlight_Movielense_Sqeuence_Validation1"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
    print("A new experiment was created.")
except:
    current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
    experiment_id = current_experiment['experiment_id']
    print("The experiment already exists. Continue with the existing experiment.")

sequmods = ['pooling','cnn','lstm']
loss='bpr'
iters = [1,5,10,20,30,40,50,60,70,80,90,100]
for repres in sequmods:
    for n_iter in iters:
        with mlflow.start_run(experiment_id=experiment_id):
            model = ImplicitSequenceModel(n_iter=n_iter,
                                      representation=repres,
                                      loss=loss)
            model.fit(train)
            # training metrics
            mrr_train = sequence_mrr_score(model, train) # the training metrics, useful for overfitting!
            invmrr_train = invert_mrr(mrr_train)
            mean_mrr_train = np.mean(mrr_train)
            mean_invmrr_train = np.mean(invmrr_train)
            # validation metrics
            mrr = sequence_mrr_score(model, validation)
            invmrr = invert_mrr(mrr)
            mean_mrr = np.mean(mrr)
            mean_invmrr = np.mean(invmrr)
            
            # log the model's parameters in mlflow
            log_param("dataset_name",dataset_name)
            log_param("n_iter",n_iter)
            log_param("loss",loss)
            log_param("model_type",repres)
            # log the model's metrics in mlflow
            log_metric("mean_mrr",mean_mrr)
            log_metric("mean_invmrr",mean_invmrr)
            log_metric("mean_mrr_train",mean_mrr_train)
            log_metric("mean_invmrr_train",mean_invmrr_train)

test_mrr = report_test_mrr(experiment_id, train_valid, test)
test_invmrr = invert_mrr(test_mrr)
best_params = get_best_params(experiment_id)
best_params["test_mrr"] = np.mean(test_mrr)
best_params["average_first_relevant_movie"] = np.mean(test_invmrr)
with open('saved_model.txt', 'w') as f:
    for key,value in best_params.items():
        f.write('%s:%s\n' % (key, value))

# Retrain the model and save it
n_iter, model_type, loss = best_params["n_iter"], best_params["model_type"], best_params["loss"]
n_iter = int(n_iter)
best_model = ImplicitSequenceModel(n_iter=n_iter,
                                  representation=model_type,
                                  loss=loss)
best_model.fit(dataset)
torch.save(best_model, "./saved_model.pt")