#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 07:16:55 2022

@author: hannah
"""
import numpy as np
import pandas as pd

from spotlight.datasets.movielens import get_movielens_dataset

dataset = get_movielens_dataset(variant='100K')
import torch

from spotlight.factorization.explicit import ExplicitFactorizationModel

model = ExplicitFactorizationModel(loss='regression',
                                   embedding_dim=128,  # latent dimensionality
                                   n_iter=10,  # number of epochs of training
                                   batch_size=1024,  # minibatch size
                                   l2=1e-9,  # strength of L2 regularization
                                   learning_rate=1e-3,
                                   use_cuda=torch.cuda.is_available())
from spotlight.cross_validation import random_train_test_split

train, test = random_train_test_split(dataset, random_state=np.random.RandomState(42))

print('Split into \n {} and \n {}.'.format(train, test))
model.fit(train, verbose=True)
from spotlight.evaluation import rmse_score

train_rmse = rmse_score(model, train)
test_rmse = rmse_score(model, test)

print('Train RMSE {:.3f}, test RMSE {:.3f}'.format(train_rmse, test_rmse))

dataset_path= "movies.csv"

outdict={}
with open(dataset_path,'r') as fh:
    for line in fh:
        if line.startswith('mov'): continue
        pline=line.strip().split(',')
        movie_id, title =pline[0], pline[1]
        if movie_id in outdict:continue
        outdict[movie_id]=title
        
def recommend_movies(user_id, dataset, model, n_movies=5):
    
    ratings=model.predict(user_ids=user_id)
    indices=np.argpartition(ratings,-n_movies)[-n_movies:]
    best_movie_ids=indices[np.argsort(ratings[indices])]
    movie_id= [dataset[i] for i in best_movie_ids]

    return [outdict[str(v)] for v in list(movie_id)]

recommend_movies(1, dataset.item_ids, model)
