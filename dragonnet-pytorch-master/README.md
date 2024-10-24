
Pytorch implementation of DragonNet from the paper:

Shi, C., Blei, D. and Veitch, V., 2019. Adapting neural networks for the estimation of treatment effects. Advances in neural information processing systems, 32.
[arxiv link](https://arxiv.org/abs/1906.02120)

Author's original Tensorflow [implementation](https://github.com/claudiashi57/dragonnet)

### Installation

```shell
python setup.py bdist_wheel
pip install dist/dragonnet-0.1-py3-none-any.whl
```

### Usage

```python
# import the module
from dragonnet.dragonnet import DragonNet

# initialize model and train
model = DragonNet(X.shape[1])
model.fit(X_train, y_train, t_train)

# predict
y0_pred, y1_pred, t_pred, _ = model.predict(X_test)
```

### Parameters
```text
class dragon.DragonNet(input_dim, shared_hidden=200, outcome_hidden=100, alpha=1.0, beta=1.0, epochs=200, batch_size=64, learning_rate=1e-5, data_loader_num_workers=4, loss='tarreg')

input_dim: int
    input dimension for covariates
shared_hidden: int, default=200
    layer size for hidden shared representation layers
outcome_hidden: int, default=100
    layer size for conditional outcome layers
alpha: float, default=1.0
    loss component weighting hyperparameter between 0 and 1
beta: float, default=1.0
    targeted regularization hyperparameter between 0 and 1
epochs: int, default=200
    Number training epochs
batch_size: int, default=64
    Training batch size
learning_rate: float, default=1e-3
    Learning rate
data_loader_num_workers: int, default=4
    Number of workers for data loader
loss: str, {'tarreg', 'default'}, default='tarreg'
    Loss function to use
```

### To do:
1) Replicate experiments on IHDP and ACIC data
import numpy as np
import pandas as pd
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