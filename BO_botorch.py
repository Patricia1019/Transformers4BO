import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound,ExpectedImprovement
from botorch.optim import optimize_acqf
from time import time
import pdb
import numpy as np

# data
num_features = 5
train_X = torch.rand(50, num_features)
# train_X = torch.tensor(np.array([np.array([0.23180091, 0.94422106, 0.42501595, 0.26483917, 0.23192972]), \
#         np.array([0.79809764, 0.11301675, 0.63663546, 0.62484511, 0.83708263]), \
#         np.array([0.03780123, 0.47780602, 0.60053366, 0.56114713, 0.47477184]),\
#          np.array([0.85959314, 0.80438238, 0.44738586, 0.97268328, 0.27733487]), \
#          np.array([0.95092983, 0.67406061, 0.12863479, 0.65752681, 0.44891038])]))
def function(x):
    y = 1 - torch.norm(x - 0.5, dim=-1, keepdim=True)
    y = y + 0.1 * torch.randn_like(y)  # add some noise
    return y
train_Y = function(train_X)
train_Y = standardize(train_Y)

# gp mdel
gp = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

# define acquisition function
UCB = UpperConfidenceBound(gp, beta=0.1)
EI = ExpectedImprovement(model=gp, best_f=train_Y.max())

# optimize the ac
bounds = torch.stack([torch.zeros(num_features), torch.ones(num_features)])
max_sum = 0
iter_num = 10
for _ in range(iter_num):
    t1 = time()
    candidate, acq_value = optimize_acqf(
        EI, bounds=bounds, q=1, num_restarts=10, raw_samples=500,
    )
    t2 = time()
    # print("Input values: ")
    # print(candidate)  # argmax
    print("maximum value: ")
    print(function(candidate).item())  # max value
    # print(f"time: {t2-t1}")
    max_sum += function(candidate).item()
print(f"avg:{max_sum/iter_num}")