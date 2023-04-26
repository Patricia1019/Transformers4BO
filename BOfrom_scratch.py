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
import pandas as pd


class BayesianOptimization:
    def __init__(self, objective_function, bounds, n_init, ac, k=None, v=None,init_point=None):
        self.objective_function = objective_function
        self.bounds = bounds
        self.n_dims = len(bounds)
        self.X = []
        self.y = []
        self.ac = ac
        self.k = k
        self.v = v
        self.init_point = init_point
        self.gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=0.01)
        self.initialize(n_init)

    def initialize(self, n_init):
        if self.init_point == None:
            for i in range(n_init):
                x = np.array([np.random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.n_dims)])
                y = self.objective_function(x)
                self.X.append(x)
                self.y.append(y)
        else:
            self.X,self.y = self.init_point
        self.gpr.fit(self.X, self.y)

    def _acquisition(self, X):
        y_hat, sigma = self.gpr.predict(X.reshape(1, -1), return_std=True)
        if self.ac == 'EI':
            best_y = np.max(self.y)
            if sigma == 0:
                return 0
            u = (y_hat - best_y) / sigma
            if self.v == None:
                ac_value =  (y_hat - best_y) * norm.cdf(u) + sigma * norm.pdf(u)
            else:
                ac_value =  (y_hat - best_y - self.v) * norm.cdf(u) + sigma * norm.pdf(u)
        elif self.ac == 'UCB':
            assert self.k != None
            ac_value = y_hat +self. k * sigma
        elif self.ac == 'PI':
            assert self.v != None
            best_y = np.max(self.y)
            ac_value = norm.cdf((y_hat - best_y - self.v) / sigma)
        return ac_value

    def _next_sample(self):
        x_next = None
        max_ei = -np.inf
        bounds = np.array(self.bounds)
        for i in range(200):
            x = np.random.uniform(bounds[:, 0], bounds[:, 1])
            ei = self._acquisition(x)
            if ei > max_ei:
                x_next = x
                max_ei = ei
        return x_next

    def optimize(self, n_iter, n_init=5):
        # self.initialize(n_init)
        max_y = -np.inf
        for i in range(n_iter):
            x_next = self._next_sample()
            y_next = self.objective_function(x_next)
            self.X.append(x_next)
            self.y.append(y_next)
            self.gpr.fit(self.X, self.y)
            if np.max(self.y) > max_y:
                max_y = np.max(self.y)
        return self.X[np.argmax(self.y)]

class PTBayesianOptimization:
    def __init__(self, objective_function, bounds, model, n_init, ac, k=None, v=None,init_point=None):
        self.objective_function = objective_function
        self.bounds = bounds
        self.n_dims = len(bounds)
        self.X = []
        self.y = []
        self.ac = ac
        self.k = k
        self.v = v
        self.init_point = init_point
        self.model = model
        self.initialize(n_init)

    def initialize(self, n_init):
        # self.X = [np.array([0.23180091, 0.94422106, 0.42501595, 0.26483917, 0.23192972]), \
        # np.array([0.79809764, 0.11301675, 0.63663546, 0.62484511, 0.83708263]), \
        # np.array([0.03780123, 0.47780602, 0.60053366, 0.56114713, 0.47477184]),\
        #  np.array([0.85959314, 0.80438238, 0.44738586, 0.97268328, 0.27733487]), \
        #  np.array([0.95092983, 0.67406061, 0.12863479, 0.65752681, 0.44891038])]
        if self.init_point == None:
            for i in range(n_init):
                x = np.array([np.random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.n_dims)])
                y = self.objective_function(x)
                self.X.append(x)
                self.y.append(y)
        else:
            self.X,self.y = self.init_point

    def _acquisition(self, X):
        train_X = torch.cat((torch.repeat_interleave(torch.tensor(np.array(self.X)).unsqueeze(0),repeats=X.shape[0],dim=0),torch.tensor(X)),1)
        train_Y = torch.cat((torch.repeat_interleave(torch.tensor(np.array(self.y)).unsqueeze(0),repeats=X.shape[0],dim=0).squeeze(-1),torch.zeros(X.shape[0],X.shape[1])),1)
        output,_ = self.model((train_X.float(),train_Y.float()),single_eval_pos=len(self.X))
        output = output.reshape(-1,1,output.shape[-1])
        y_hat = self.model.criterion.mean(output.to('cuda')).cpu().detach()[:,0]
        sigma = (self.model.criterion.quantile(output)[:,0,1]-self.model.criterion.quantile(output)[:,0,0])/2
        if self.ac == 'EI':
            # y_hat, sigma = self.gpr.predict(X.reshape(1, -1), return_std=True)
            best_y = np.max(self.y)
            if sigma.equal(torch.zeros(sigma.shape)):
                return torch.zeros(sigma.shape)
            u = (y_hat - best_y) / sigma
            if self.v == None:
                ac_value = (y_hat - best_y) * norm.cdf(u) + sigma * norm.pdf(u)
            else:
                ac_value = (y_card - best_y - self.v) * norm.cdf(u) + sigma * norm.pdf
        elif self.ac == 'UCB':
            assert self.k != None
            ac_value = y_hat + self.k * sigma
        elif self.ac == 'PI':
            assert self.v != None
            best_y = np.max(self.y)
            ac_value = norm.cdf((y_hat - best_y - self.v) / sigma)
        return ac_value

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
        batch_num = 2
        x = np.random.uniform(bounds[:, 0], bounds[:, 1],(batch_num,eval_points,bounds[:,0].shape[0]))
        ei = self._acquisition(x)
        x = x.reshape(-1,x.shape[-1])
        if ei.max() > max_ei:
            x_next = x[np.argmax(ei)]
            max_ei = ei.max()
        return x_next

    def optimize(self, n_iter, n_init=5):
        # self.initialize(n_init)
        max_y = -np.inf
        for i in range(n_iter):
            x_next = self._next_sample()
            y_next = self.objective_function(x_next)
            self.X.append(x_next)
            self.y.append(y_next)
            # self.gpr.fit(self.X, self.y)
            if np.max(self.y) > max_y:
                max_y = np.max(self.y)
        return self.X[np.argmax(self.y)]
        
def function(x):
    x = torch.tensor(x)
    y = 1 - torch.norm(x - 0.5, dim=-1, keepdim=True)
    # y = y + 0.1 * torch.randn_like(y)  # add some noise
    y = np.array(y)
    return y
    # return x[0]**2 + x[1]**2

class Function:
    def __init__(self,type,noisy=False,return_value=False):
        self.type = type
        self.noisy = noisy
        self.return_value = return_value

    def quadratic(self,x): # max point: all 0.5, max value: 0, min value: -0.25*num_feature
        x = np.array(x)
        y = -np.sum((x-0.5)**2,-1)
        if self.noisy:
            y += 0.1 * np.random.randn(*y.shape)
        y = np.array([y])
        return y
    def exponential(self,x): # max point: all 1, max value: np.e * num_feature, min value: num_feature
        x = np.array(x)
        y = np.sum(np.exp(x),-1)
        if self.noisy:
            y += 0.1 * np.random.randn(*y.shape)
        y = np.array([y])
        return y
    def log(self,x): # max point: all 1, max value: np.log(2) * num_feature, min value: 0
        x = np.array(x)
        y = np.sum(np.log(x+1),-1)
        if self.noisy:
            y += 0.1 * np.random.randn(*y.shape)
        y = np.array([y])
        return y
    def rosenbrock(self,x): # max point: all 1-0.1, max value: 0, min value: -((num_feature-1)*100+num_feature//2)
        x = np.array(x)
        x = x + 0.1
        assert x.shape[0] > 1, "Rosenbrock function: Input must have at least 2 dimension"
        y = -np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0,-1)
        if self.noisy:
            y += 0.1 * np.random.randn(*y.shape)
        y = np.array([y])
        return y
    def rastrigin(self,x): # max point: all 0+0.1, max value:0, min value≈-20.25*num_feature
        x = np.array(x)
        x = x - 0.1
        A = 10
        n_dim = x.shape[0]
        y = -(A * n_dim + np.sum(x**2 - A * np.cos(2 * np.pi * x),-1))
        if self.noisy:  
            y += 0.1 * np.random.randn(*y.shape)
        y = np.array([y])   
        return y
    def ackley(self,x): # max point: all 0+0.1, max value: 0, min value > 20*np.exp(-0.2)+np.exp(-1)-20-np.e
        x = np.array(x)
        x = x - 0.1
        a = 20
        b = 0.2
        d = x.shape[0]
        y = -(-a * np.exp(-b * np.sqrt(1.0/d * np.sum(x**2,-1))) - np.exp(1.0/d * np.sum(np.cos(2 * np.pi * x),-1)) + a + np.e)
        if self.noisy:
            y += 0.1 * np.random.randn(*y.shape)
        y = np.array([y])
        return y

    def __call__(self,x=None):
        if self.return_value:
            if self.type == 'unimodel':
                y_quadratic = self.quadratic(x)
                y_exponential = self.exponential(x)
                y_log = self.log(x)
                y = {"quadratic":y_quadratic,"exponential":y_exponential,"log":y_log}
            elif self.type == 'multimodel':
                y_rosenbrock = self.rosenbrock(x)
                y_rastrigin = self.rastrigin(x)
                y_ackley = self.ackley(x)
                y = {"rosenbrock":y_rosenbrock,"rastrigin":y_rastrigin,"ackley":y_ackley}
            return y 
        else:
            if self.type == 'unimodel':
                return {"quadratic":self.quadratic,"exponential":self.exponential,"log":self.log}
            elif self.type =='multimodel':
                return {"rosenbrock":self.rosenbrock,"rastrigin":self.rastrigin,"ackley":self.ackley}

if __name__ == '__main__':
    PT = True
    num_features = 40
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
        data_augment = True
        lr = 0.0008
        epochs = 625
        root_dir = f'/home/ypq/TransformersCanDoBayesianInference/myresults/GPfitting_augment_{num_features}feature'
        model = MyTransformerModel(encoder, num_borders, emsize, 4, 2*emsize, 6, 0.0,
                        y_encoder=encoders.Linear(1, emsize), input_normalization=False,
                        # pos_encoder=positional_encodings.NoPositionalEncoding(emsize, bptt*2),
                        pos_encoder=positional_encodings.NoPositionalEncoding(emsize, bptt*2),
                        decoder=None
                        )
        model.criterion = bar_distribution.FullSupportBarDistribution(bar_distribution.get_bucket_limits(num_borders, ys=ys))
        model_path = f'{root_dir}/numborder{num_borders}_lr{lr}_epoch{epochs}_GPfitting.pth'
        checkpoint = torch.load(model_path)
        model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
        model.eval()

    # max value
    # max_sum = 0
    # iter_num = 100
    # n_init = 50
    # if PT:
    #     out_path = f'./PT_iter{iter_num}_init{n_init}.xlsx'
    # else:
    #     out_path = f'./GP_iter{iter_num}_init{n_init}.xlsx'
    # results = {}
    # results['max value'] = []
    # results['time'] = []
    # for _ in range(iter_num):
    #     if PT:
    #         bo = PTBayesianOptimization(function, bounds, model,n_init=n_init,ac='EI')
    #     else:
    #         bo = BayesianOptimization(function, bounds,n_init=n_init,ac='EI')
    #     t1 = time()
    #     x_max = bo.optimize(n_iter=20)
    #     t2 = time()
    #     # print("Input values: ")
    #     # print(x_max)
    #     print("maximum value: ")
    #     print(function(x_max))
    #     print(f"time: {t2-t1}")
    #     max_sum += function(x_max)
    #     results['max value'].append(function(x_max)[0])
    #     results['time'].append(t2-t1)
    # # df = pd.DataFrame(results)
    # # df.to_excel(out_path,index=False)
    # print(f"avg:{max_sum/iter_num}")


    # simple regret
    iter_num = 200
    n_init = 5000
    results = {}
    results['iter_num'] = []
    results['PT regret value'] = []
    results['GP regret value'] = []
    out_path = f'./simple_regret_total{iter_num}_init{n_init}.xlsx'
    PTBO = PTBayesianOptimization(function, bounds, model,n_init=n_init,ac='EI')
    GPBO = BayesianOptimization(function, bounds,n_init=2000,ac='EI')
    for i in range(1,iter_num+1):
        PT_x_max = PTBO.optimize(n_iter=1)
        GP_x_max = GPBO.optimize(n_iter=1)
        print("PT regret value: ")
        print(1-function(PT_x_max))
        print("GP regret value: ")
        print(1-function(GP_x_max))
        results['iter_num'].append(i)
        results['PT regret value'].append(1-function(PT_x_max)[0])
        results['GP regret value'].append(1-function(GP_x_max)[0])
    df = pd.DataFrame(results)
    df.to_excel(out_path,index=False)
