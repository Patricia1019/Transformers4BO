import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import torch
import torch.nn as nn
from time import time
import pdb
import sys,os
from transformer import TransformerModel, MyTransformerModel
sys.path.append('/home/ypq/TransformersCanDoBayesianInference')
import encoders
import positional_encodings
import bar_distribution
import priors


class BayesianOptimization:
    def __init__(self, objective_function, bounds):
        self.objective_function = objective_function
        self.bounds = bounds
        self.n_dims = len(bounds)
        self.X = []
        self.y = []
        self.gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=0.01)

    def initialize(self, n_init):
        for i in range(n_init):
            x = np.array([np.random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.n_dims)])
            y = self.objective_function(x)
            self.X.append(x)
            self.y.append(y)
        self.gpr.fit(self.X, self.y)

    def _acquisition(self, X):
        y_hat, sigma = self.gpr.predict(X.reshape(1, -1), return_std=True)
        best_y = np.min(self.y)
        if sigma == 0:
            return 0
        u = (best_y - y_hat) / sigma
        return (best_y - y_hat) * norm.cdf(u) + sigma * norm.pdf(u)

    def _next_sample(self):
        x_next = None
        max_ei = -np.inf
        bounds = np.array(self.bounds)
        for i in range(2000):
            x = np.random.uniform(bounds[:, 0], bounds[:, 1])
            ei = self._acquisition(x)
            if ei > max_ei:
                x_next = x
                max_ei = ei
        return x_next

    def optimize(self, n_iter, n_init=5):
        self.initialize(n_init)
        min_y = np.inf
        for i in range(n_iter):
            x_next = self._next_sample()
            y_next = self.objective_function(x_next)
            self.X.append(x_next)
            self.y.append(y_next)
            self.gpr.fit(self.X, self.y)
            if np.min(self.y) < min_y:
                min_y = np.min(self.y)
        return self.X[np.argmin(self.y)]

class PTBayesianOptimization:
    def __init__(self, objective_function, bounds, model):
        self.objective_function = objective_function
        self.bounds = bounds
        self.n_dims = len(bounds)
        self.X = []
        self.y = []
        self.model = model

    def initialize(self, n_init):
        # self.X = [np.array([0.23180091, 0.94422106, 0.42501595, 0.26483917, 0.23192972]), \
        # np.array([0.79809764, 0.11301675, 0.63663546, 0.62484511, 0.83708263]), \
        # np.array([0.03780123, 0.47780602, 0.60053366, 0.56114713, 0.47477184]),\
        #  np.array([0.85959314, 0.80438238, 0.44738586, 0.97268328, 0.27733487]), \
        #  np.array([0.95092983, 0.67406061, 0.12863479, 0.65752681, 0.44891038])]
        for i in range(n_init):
            x = np.array([np.random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.n_dims)])
            y = self.objective_function(x)
            self.X.append(x)
            self.y.append(y)

    def _acquisition(self, X):
        # y_hat, sigma = self.gpr.predict(X.reshape(1, -1), return_std=True)
        train_X = torch.cat((torch.repeat_interleave(torch.tensor(np.array(self.X)).unsqueeze(0),repeats=X.shape[0],dim=0),torch.tensor(X)),1)
        train_Y = torch.cat((torch.repeat_interleave(torch.tensor(np.array(self.y)).unsqueeze(0),repeats=X.shape[0],dim=0).squeeze(-1),torch.zeros(X.shape[0],X.shape[1])),1)
        output,_ = self.model((train_X.float(),train_Y.float()),single_eval_pos=len(self.X))
        output = output.reshape(-1,1,output.shape[-1])
        y_hat = model.criterion.mean(output.to('cuda')).cpu().detach()[:,0]
        sigma = (model.criterion.quantile(output)[:,0,1]-model.criterion.quantile(output)[:,0,0])/2
        best_y = np.min(self.y)
        if sigma.equal(torch.zeros(sigma.shape)):
            return torch.zeros(sigma.shape)
        u = (best_y - y_hat) / sigma
        return (best_y - y_hat) * norm.cdf(u) + sigma * norm.pdf(u)

    def _next_sample(self):
        x_next = None
        max_ei = -np.inf
        bounds = np.array(self.bounds)
        # for i in range(100):
        #     x = np.random.uniform(bounds[:, 0], bounds[:, 1])
        #     ei = self._acquisition(x)
        #     if ei > max_ei:
        #         x_next = x
        #         max_ei = ei
        eval_points = 100
        batch_num = 20
        x = np.random.uniform(bounds[:, 0], bounds[:, 1],(batch_num,eval_points,bounds[:,0].shape[0]))
        ei = self._acquisition(x)
        x = x.reshape(-1,x.shape[-1])
        if ei.max() > max_ei:
            x_next = x[np.argmax(ei)]
            max_ei = ei.max()
        return x_next

    def optimize(self, n_iter, n_init=5):
        self.initialize(n_init)
        min_y = np.inf
        for i in range(n_iter):
            x_next = self._next_sample()
            y_next = self.objective_function(x_next)
            self.X.append(x_next)
            self.y.append(y_next)
            # self.gpr.fit(self.X, self.y)
            if np.min(self.y) < min_y:
                min_y = np.min(self.y)
        return self.X[np.argmin(self.y)]

if __name__ == '__main__':
    def function(x):
        x = torch.tensor(x)
        y = 1 - torch.norm(x - 0.5, dim=-1, keepdim=True)
        # y = y + 0.1 * torch.randn_like(y)  # add some noise
        y = np.array(y)
        return -y
        # return x[0]**2 + x[1]**2
    PT = True
    num_features = 5
    bounds = [(0,1) for _ in range(num_features)]
    if PT:
        emsize = 512
        encoder = encoders.Linear(num_features,emsize)
        bptt = 2010
        hps = {'noise': 1e-4, 'outputscale': 1., 'lengthscale': .6, 'fast_computations': (False,False,False)}
        ys = priors.fast_gp.get_batch_first(100000,20,num_features, hyperparameters=hps)[1]
        # num_border_list = [1000,10000]
        num_borders = 1000
        # epoch_list = [50,100,200,400]
        epoch_list = [200]
        batch_fraction = 8
        draw_flag = False
        data_augment = False
        lr = 0.0008
        epochs = 625
        root_dir = '/home/ypq/TransformersCanDoBayesianInference/myresults/GPfitting_parallel'
        model = MyTransformerModel(encoder, num_borders, emsize, 4, 2*emsize, 6, 0.0,
                        y_encoder=encoders.Linear(1, emsize), input_normalization=False,
                        pos_encoder=positional_encodings.NoPositionalEncoding(emsize, bptt*2),
                        decoder=None
                        )
        model.criterion = bar_distribution.FullSupportBarDistribution(bar_distribution.get_bucket_limits(num_borders, ys=ys))
        model_path = f'{root_dir}/numborder{num_borders}_lr{lr}_epoch{epochs}_GPfitting.pth'
        checkpoint = torch.load(model_path)
        model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
        model.eval()
    max_sum = 0
    iter_num = 10
    for _ in range(iter_num):
        if PT:
            bo = PTBayesianOptimization(function, bounds, model)
        else:
            bo = BayesianOptimization(function, bounds)
        t1 = time()
        x_min = bo.optimize(n_iter=20,n_init=50)
        t2 = time()
        # print("Input values: ")
        # print(x_min)
        print("maximum value: ")
        print(-function(x_min))
        print(f"time: {t2-t1}")
        max_sum += -function(x_min)
    print(f"avg:{max_sum/iter_num}")
    