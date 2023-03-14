from sklearn.gaussian_process import GaussianProcessRegressor
self.GP = GaussianProcessRegressor(...)

# 定义acquisition function
def PI(x, gp, y_max, xi):
    mean, std = gp.predict(x, return_std=True)
    z = (mean - y_max - xi)/std
    return norm.cdf(z)
def EI(x, gp, y_max, xi):
    mean, std = gp.predict(x, return_std=True)
    a = (mean - y_max - xi)
    z = a / std
    return a * norm.cdf(z) + std * norm.pdf(z)
def UCB(x, gp, kappa):
    mean, std = gp.predict(x, return_std=True)
    return mean + kappa * std

# 寻找acquisition function最大的对应解
def acq_max(ac, gp, y_max, bounds, random_state, n_warmup=10000):
    # 随机采样选择最大值
    x_tries = np.random.RandomState(random_state).uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))
    ys = ac(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()
    return x_max

if __name__ == '__main__':
    
while iteration < n_iter:
    # 更新高斯过程的后验分布
    self.GP.fit(X, y)
    # 根据acquisition函数计算下一个试验点
    suggestion = acq_max(
            ac=utility_function,
            gp=self.GP,
            y_max=y.max(),
            bounds=self.bounds,
            random_state=self.random_state
        )
    # 进行试验（采样），更新观测点集合
    X.append(suggestion)
    y.append(target_func(suggestion))
    iteration += 1
