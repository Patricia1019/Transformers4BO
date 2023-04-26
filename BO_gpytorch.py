import torch
import gpytorch
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.test_functions import Rastrigin
import pdb

# Define the objective function
objective_function = Rastrigin()

# Define the GP model
class GPModel(ExactGP):
    def __init__(self, train_X, train_Y, likelihood):
        super(GPModel, self).__init__(train_X, train_Y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# Set up the optimization loop
bounds = torch.tensor(objective_function.bounds)
train_X = torch.rand(5, 2) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]  # initialize with 5 random points
train_Y = objective_function(train_X).unsqueeze(-1)
likelihood = GaussianLikelihood()
model = GPModel(train_X, train_Y, likelihood)
mll = ExactMarginalLogLikelihood(likelihood, model)

# Perform Bayesian optimization
num_iterations = 20
for i in range(num_iterations):
    # Update the GP model with the current data
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    mll.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    output = model(train_X)
    loss = -mll(output, train_Y)
    loss.backward()
    optimizer.step()
    model.eval()
    mll.eval()
    
    # Define the acquisition function
    ucb = UpperConfidenceBound(model, beta=0.1)
    
    # Optimize the acquisition function to get the next point to sample
    next_x = optimize_acqf(
        ucb, bounds=bounds, q=1, num_restarts=5, raw_samples=20
    )
    
    # Evaluate the objective function at the next point
    next_y = objective_function(next_x)
    
    # Add the new data point to the training set
    train_X = torch.cat([train_X, next_x])
    train_Y = torch.cat([train_Y, next_y])

# Get the optimal solution and its value
optimal_x = train_X[train_Y.argmax()]
optimal_y = train_Y.max()

print("Optimal solution: ", optimal_x)
print("Optimal value: ", optimal_y)

