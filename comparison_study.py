from BOfrom_scratch import *
import pdb
import math
from tqdm import tqdm

if __name__ == '__main__':
    PT = True
    num_features = 5
    bounds = [(0,1) for _ in range(num_features)]
    max_values = {"quadratic":0,"exponential":np.e * num_features,"log":np.log(2) * num_features,"rosenbrock":0,"rastrigin":0,"ackley":0}
    scale_factors = {"quadratic":0.25*num_features,"exponential":(np.e-1)*num_features,"log":np.log(2) * num_features,\
                    "rosenbrock":((num_features-1)*100+num_features//2),"rastrigin":20.25*num_features,"ackley":4.7}
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
        root_dir = f'/home/ypq/TransformersCanDoBayesianInference/myresults/GPfitting_augment{data_augment}_{num_features}feature'
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
    # simple regret: different ac params for PT model and GP model(fixed)
    # UCB
    iter_num = 100
    n_init = 50
    ac = 'UCB'
    results = {}
    # results['PT regret value'] = {}
    results['GP regret value'] = {}
    iter_step = 5
    k_list = [0.1,0.5,1.0,1.5,2.0,2.5]
    for k in k_list:
        # results['PT regret value'][f'k={k}'] = []
        results['GP regret value'][f'k={k}'] = []
    # results['PT regret value']['iter_num'] = []
    results['GP regret value']['iter_num'] = []
    for i in range(1,iter_num+1,iter_step):
        # results['PT regret value']['iter_num'].append(i)
        results['GP regret value']['iter_num'].append(i)
        for k in k_list:
            # results['PT regret value'][f'k={k}'].append(0)
            results['GP regret value'][f'k={k}'].append(0)
    
    repeat_num = 100
    func_type = 'multimodel'
    index = 1
    functions = Function(type=func_type)
    functions = functions()
    function = list(functions.values())[index]
    max_value = max_values[list(functions.keys())[index]]
    scale_factor = scale_factors[list(functions.keys())[index]]
    for n in range(repeat_num):
        for k in k_list:
            # PTBO = PTBayesianOptimization(function, bounds, model,n_init=n_init,ac='UCB',k=k)
            GPBO = BayesianOptimization(function, bounds,n_init=n_init,ac='UCB',k=k)
            for i in range(1,iter_num+1,iter_step):
                # PT_x_max = PTBO.optimize(n_iter=iter_step)
                GP_x_max = GPBO.optimize(n_iter=iter_step)
                # print("PT regret value: ")
                # print(1-function(PT_x_max))
                print("GP regret value: ")
                print((max_value-function(GP_x_max))/scale_factor)
                # results['PT regret value'][f'k={k}'][i//iter_step] += (1-function(PT_x_max)[0])
                results['GP regret value'][f'k={k}'][i//iter_step] += (max_value-function(GP_x_max)[0])/scale_factor
    for k in k_list:
        # tmp1 = [results['PT regret value'][f'k={k}'][i]/repeat_num for i in range(iter_num//iter_step)]
        tmp2 = [results['GP regret value'][f'k={k}'][i]/repeat_num for i in range(iter_num//iter_step)]
        # results['PT regret value'][f'k={k}'] = tmp1
        results['GP regret value'][f'k={k}'] = tmp2
    out_root_dir = './numerical_results/ac_comparison'
    # df_PT = pd.DataFrame(results['PT regret value'])
    # out_PT_path = f'./simple_regret_UCB_params_total{iter_num}_PT_init{n_init}_repeat{repeat_num}.xlsx'
    # df_PT.to_excel(out_PT_path,index=False)
    df_GP = pd.DataFrame(results['GP regret value'])
    out_GP_path = f'{out_root_dir}/simple_regret_UCB_params_{func_type}{index}_total{iter_num}_GP_init{n_init}_repeat{repeat_num}.xlsx'
    df_GP.to_excel(out_GP_path,index=False)

    # PI
    # iter_num = 100
    # n_init = 50
    # ac = 'PI'
    # results = {}
    # # results['PT regret value'] = {}
    # results['GP regret value'] = {}
    # iter_step = 5
    # v_list = [0.02,0.04,0.06,0.08,0.1,0.12]
    # for v in v_list:
    #     # results['PT regret value'][f'v={v}'] = []
    #     results['GP regret value'][f'v={v}'] = []
    # # results['PT regret value']['iter_num'] = []
    # results['GP regret value']['iter_num'] = []
    # for i in range(1,iter_num+1,iter_step):
    #     # results['PT regret value']['iter_num'].append(i)
    #     results['GP regret value']['iter_num'].append(i)
    #     for v in v_list:
    #         # results['PT regret value'][f'v={v}'].append(0)
    #         results['GP regret value'][f'v={v}'].append(0)

    # repeat_num = 100
    # func_type = 'multimodel'
    # index = 1
    # functions = Function(type=func_type)
    # functions = functions()
    # function = list(functions.values())[index]
    # max_value = max_values[list(functions.keys())[index]]
    # scale_factor = scale_factors[list(functions.keys())[index]]
    # for n in tqdm(range(repeat_num)):
    #     for v in v_list:
    #         # PTBO.v = v
    #         GPBO = BayesianOptimization(function, bounds,n_init=n_init,ac=ac,v=v)
    #         for i in range(1,iter_num+1,iter_step):
    #             # PT_x_max = PTBO.optimize(n_iter=iter_step)
    #             GP_x_max = GPBO.optimize(n_iter=iter_step)
    #             # print("PT regret value: ")
    #             # print(1-function(PT_x_max))
    #             print("GP regret value: ")
    #             print((max_value-function(GP_x_max))/scale_factor)
    #             # results['PT regret value'][f'v={v}'][i//iter_step] += (1-function(PT_x_max)[0])
    #             results['GP regret value'][f'v={v}'][i//iter_step] += (max_value-function(GP_x_max)[0])/scale_factor
    # for v in v_list:
    #     # tmp1 = [results['PT regret value'][f'v={v}'][i]/repeat_num for i in range(iter_num//iter_step)]
    #     tmp2 = [results['GP regret value'][f'v={v}'][i]/repeat_num for i in range(iter_num//iter_step)]
    #     # results['PT regret value'][f'v={v}'] = tmp1
    #     results['GP regret value'][f'v={v}'] = tmp2
    # out_root_dir = './numerical_results/ac_comparison'
    # # df_PT = pd.DataFrame(results['PT regret value'])
    # # out_PT_path = f'{out_root_dir}/simple_regret_PI_params2_total{iter_num}_PT_init{n_init}_repeat{repeat_num}.xlsx'
    # # df_PT.to_excel(out_PT_path,index=False)
    # df_GP = pd.DataFrame(results['GP regret value'])
    # out_GP_path = f'{out_root_dir}/simple_regret_PI_params_{func_type}{index}_total{iter_num}_GP_init{n_init}_repeat{repeat_num}.xlsx'
    # df_GP.to_excel(out_GP_path,index=False)

    # EI with no parameters 
    # iter_num = 100
    # n_init = 50
    # ac = 'EI'
    # iter_step = 5
    # results = {}
    # results['iter_num'] = [i for i in range(1,iter_num+1,iter_step)]
    # results['GP regret value'] = [0]*(iter_num//iter_step)

    # repeat_num = 100
    # for n in tqdm(range(repeat_num)):
    #     GPBO = BayesianOptimization(function, bounds,n_init=n_init,ac=ac)
    #     for i in range(1,iter_num+1,iter_step):
    #         # PT_x_max = PTBO.optimize(n_iter=iter_step)
    #         GP_x_max = GPBO.optimize(n_iter=iter_step)
    #         # print("PT regret value: ")
    #         # print(1-function(PT_x_max))
    #         print("GP regret value: ")
    #         print(1-function(GP_x_max))
    #         # results['PT regret value'][f'v={v}'][i//iter_step] += (1-function(PT_x_max)[0])
    #         results['GP regret value'][i//iter_step] += (1-function(GP_x_max)[0])
    # # tmp1 = [results['PT regret value'][f'v={v}'][i]/repeat_num for i in range(iter_num//iter_step)]
    # tmp2 = [results['GP regret value'][i]/repeat_num for i in range(iter_num//iter_step)]
    # # results['PT regret value'][f'v={v}'] = tmp1
    # results['GP regret value'] = tmp2
    # out_root_dir = './numerical_results/ac_comparison'
    # # df_PT = pd.DataFrame(results['PT regret value'])
    # # out_PT_path = f'{out_root_dir}/simple_regret_PI_params2_total{iter_num}_PT_init{n_init}_repeat{repeat_num}.xlsx'
    # # df_PT.to_excel(out_PT_path,index=False)
    # df_GP = pd.DataFrame(results)
    # out_GP_path = f'{out_root_dir}/simple_regret_{ac}_params_total{iter_num}_GP_init{n_init}_repeat{repeat_num}.xlsx'
    # df_GP.to_excel(out_GP_path,index=False)

    # EI with parameters 
    # iter_num = 100
    # n_init = 50
    # ac = 'EI'
    # results = {}
    # # results['PT regret value'] = {}
    # results['GP regret value'] = {}
    # iter_step = 5
    # v_list = [0.02,0.04,0.06,0.08,0.1,0.12]
    # for v in v_list:
    #     # results['PT regret value'][f'v={v}'] = []
    #     results['GP regret value'][f'v={v}'] = []
    # # results['PT regret value']['iter_num'] = []
    # results['GP regret value']['iter_num'] = []
    # for i in range(1,iter_num+1,iter_step):
    #     # results['PT regret value']['iter_num'].append(i)
    #     results['GP regret value']['iter_num'].append(i)
    #     for v in v_list:
    #         # results['PT regret value'][f'v={v}'].append(0)
    #         results['GP regret value'][f'v={v}'].append(0)

    # repeat_num = 100
    # for n in tqdm(range(repeat_num)):
    #     x = [np.array([np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(num_features)]) for _ in range(n_init)]
    #     y = [function(i) for i in x]
    #     init_point = (x,y)
    #     # PTBO = PTBayesianOptimization(function, bounds, model,n_init=n_init,ac=ac,init_point=init_point)
    #     GPBO = BayesianOptimization(function, bounds,n_init=n_init,ac=ac,init_point=init_point)
    #     for v in v_list:
    #         # PTBO.v = v
    #         GPBO.v = v
    #         for i in range(1,iter_num+1,iter_step):
    #             # PT_x_max = PTBO.optimize(n_iter=iter_step)
    #             GP_x_max = GPBO.optimize(n_iter=iter_step)
    #             # print("PT regret value: ")
    #             # print(1-function(PT_x_max))
    #             print(f"{v} GP regret value: ")
    #             print(1-function(GP_x_max))
    #             # results['PT regret value'][f'v={v}'][i//iter_step] += (1-function(PT_x_max)[0])
    #             results['GP regret value'][f'v={v}'][i//iter_step] += (1-function(GP_x_max)[0])
    # for v in v_list:
    #     # tmp1 = [results['PT regret value'][f'v={v}'][i]/repeat_num for i in range(iter_num//iter_step)]
    #     tmp2 = [results['GP regret value'][f'v={v}'][i]/repeat_num for i in range(iter_num//iter_step)]
    #     # results['PT regret value'][f'v={v}'] = tmp1
    #     results['GP regret value'][f'v={v}'] = tmp2
    # out_root_dir = './numerical_results/ac_comparison'
    # # df_PT = pd.DataFrame(results['PT regret value'])
    # # out_PT_path = f'{out_root_dir}/simple_regret_PI_params2_total{iter_num}_PT_init{n_init}_repeat{repeat_num}.xlsx'
    # # df_PT.to_excel(out_PT_path,index=False)
    # df_GP = pd.DataFrame(results['GP regret value'])
    # out_GP_path = f'{out_root_dir}/simple_regret_{ac}_params_total{iter_num}_GP_init{n_init}_repeat{repeat_num}.xlsx'
    # df_GP.to_excel(out_GP_path,index=False)

    # simple regret: auto ac params for PT model and GP model
    # UCB
    # iter_num = 100
    # n_init = 50
    # ac = 'UCB'
    # iter_step = 5
    # repeat_num = 100
    # func_type = 'multimodel'
    # index = 1
    # functions = Function(type=func_type)
    # functions = functions()
    # function = list(functions.values())[index]
    # max_value = max_values[list(functions.keys())[index]]
    # scale_factor = scale_factors[list(functions.keys())[index]]

    # results = {}
    # results['iter_num'] = []
    # # results['PT regret value'] = [0]*(iter_num//iter_step)
    # results['GP regret value'] = [0]*(iter_num//iter_step)
    # for n in range(repeat_num):
    #     # PTBO = PTBayesianOptimization(function, bounds, model,n_init=n_init,ac=ac)
    #     GPBO = BayesianOptimization(function, bounds,n_init=n_init,ac=ac)
    #     for i in range(1,iter_num+1,iter_step):
    #         # PTBO.k = math.sqrt(2*math.log(i)/iter_num)
    #         GPBO.k = math.sqrt(2*math.log(i)/iter_num)
    #         # PT_x_max = PTBO.optimize(n_iter=iter_step)
    #         GP_x_max = GPBO.optimize(n_iter=iter_step)
    #         # print("PT regret value: ")
    #         # print(1-function(PT_x_max))
    #         print("GP regret value: ")
    #         print((max_value-function(GP_x_max))/scale_factor)
    #         results['iter_num'].append(i)
    #         # results['PT regret value'][i//iter_step] += (1-function(PT_x_max)[0])
    #         results['GP regret value'][i//iter_step] += (max_value-function(GP_x_max)[0])/scale_factor

    # # tmp1 = [results['PT regret value'][i]/repeat_num for i in range(iter_num//iter_step)]
    # tmp2 = [results['GP regret value'][i]/repeat_num for i in range(iter_num//iter_step)]
    # # results['PT regret value'] = tmp1
    # results['GP regret value'] = tmp2
    # out_root_dir = './numerical_results/ac_comparison'
    # # df_PT = pd.DataFrame(results['PT regret value'])
    # # out_PT_path = f'./simple_regret_UCB_autoparams_total{iter_num}_PT_init{n_init}_repeat{repeat_num}.xlsx'
    # # df_PT.to_excel(out_PT_path,index=False)
    # df_GP = pd.DataFrame(results['GP regret value'])
    # out_GP_path = f'{out_root_dir}/simple_regret_UCB_autoparams_{func_type}{index}_total{iter_num}_GP_init{n_init}_repeat{repeat_num}.xlsx'
    # df_GP.to_excel(out_GP_path,index=False)

    # # PI
    # iter_num = 100
    # n_init = 50
    # ac = 'PI'
    # iter_step = 5
    # repeat_num = 100
    # func_type = 'multimodel'
    # index = 1
    # functions = Function(type=func_type)
    # functions = functions()
    # function = list(functions.values())[index]
    # max_value = max_values[list(functions.keys())[index]]
    # scale_factor = scale_factors[list(functions.keys())[index]]

    # results = {}
    # results['iter_num'] = []
    # # results['PT regret value'] = [0]*(iter_num//iter_step)
    # results['GP regret value'] = [0]*(iter_num//iter_step)

    # repeat_num = 100
    # for n in range(repeat_num):
    #     # PTBO = PTBayesianOptimization(function, bounds, model,n_init=n_init,ac=ac)
    #     GPBO = BayesianOptimization(function, bounds,n_init=n_init,ac=ac)
    #     for i in range(1,iter_num+1,iter_step):
    #         # PTBO.v = 0.1*np.std(np.array(PTBO.y))
    #         GPBO.v = 0.1*np.std(np.array(GPBO.y))
    #         # PT_x_max = PTBO.optimize(n_iter=iter_step)
    #         GP_x_max = GPBO.optimize(n_iter=iter_step)
    #         # print("PT regret value: ")
    #         # print(1-function(PT_x_max))
    #         print("GP regret value: ")
    #         print((max_value-function(GP_x_max))/scale_factor)
    #         results['iter_num'].append(i)
    #         # results['PT regret value'][i//iter_step] += (1-function(PT_x_max)[0])
    #         results['GP regret value'][i//iter_step] += (max_value-function(GP_x_max)[0])/scale_factor
    
    # # tmp1 = [results['PT regret value'][i]/repeat_num for i in range(iter_num//iter_step)]
    # tmp2 = [results['GP regret value'][i]/repeat_num for i in range(iter_num//iter_step)]
    # # results['PT regret value'] = tmp1
    # results['GP regret value'] = tmp2
    # out_root_dir = './numerical_results/ac_comparison'
    # # df_PT = pd.DataFrame(results['PT regret value'])
    # # out_PT_path = f'{out_root_dir}/simple_regret_PI_autoparams_total{iter_num}_PT_init{n_init}_repeat{repeat_num}.xlsx'
    # # df_PT.to_excel(out_PT_path,index=False)
    # df_GP = pd.DataFrame(results['GP regret value'])
    # out_GP_path = f'{out_root_dir}/simple_regret_PI_autoparams_{func_type}{index}_total{iter_num}_GP_init{n_init}_repeat{repeat_num}.xlsx'
    # df_GP.to_excel(out_GP_path,index=False)