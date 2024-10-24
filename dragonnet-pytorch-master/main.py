# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:18:24 2024

@author: x.liu
"""

from dragonnet.dragonnet import DragonNet
import numpy as np
import pandas as pd
import torch
with open('./ihdp_npci_1-100.train.npz','rb') as trf, open('./ihdp_npci_1-100.test.npz','rb') as tef:
        train_data=np.load(trf); test_data=np.load(tef)
        y=np.concatenate(   (train_data['yf'][:,7],   test_data['yf'][:,7])).astype('float32').squeeze() #most GPUs only compute 32-bit floats
        t=np.concatenate(   (train_data['t'][:,7],    test_data['t'][:,7])).astype('float32').squeeze()
        X=np.concatenate(   (train_data['x'][:,:,7],  test_data['x'][:,:,7]),axis=0).astype('float32')
        mu_0=np.concatenate((train_data['mu0'][:,7],  test_data['mu0'][:,7])).astype('float32').squeeze()
        mu_1=np.concatenate((train_data['mu1'][:,7],  test_data['mu1'][:,7])).astype('float32').squeeze()

#X=pd.DataFrame(X)
'''X.columns=['birth weight', 'weeks preterm', 'days in hospital','child age at treatment', 'age at birth', 'head circumference', 'male', 'first born', 'black', 'hispanic',
                'unmarried at birth', 'less than high school', 'high school graduate', 'some college', 'college graduate',
                'worked during pregnancy', 'had no prenatal care',
                'Arkansas', 'Oklahoma', 'Connecticut', 'Florida', 'Maryland', 'Pennsylvania','Texas', 'Washington']'''
torch.manual_seed(1)
np.random.seed(1)
# initialize model and train
model = DragonNet(X.shape[1])
model.fit(X,y,t)
#Plot propensity score
df_x=pd.DataFrame(X)
y0_pred, y1_pred, t_pred, eps = model.predict(df_x.values)
pd.Series(t_pred[t==1].squeeze()).plot.kde()
pd.Series(t_pred[t==0].squeeze()).plot.kde()