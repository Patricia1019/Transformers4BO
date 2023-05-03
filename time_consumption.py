from BOfrom_scratch import *
import pdb
from tqdm import tqdm

if __name__ == '__main__':
    PT = True
    num_features = 20
    bounds = [(0,1) for _ in range(num_features)]
    if PT:
        emsize = 512
        encoder = encoders.Linear(num_features,emsize)
        bptt = 2010
        hps = {'noise': 1e-4, 'outputscale': 1., 'lengthscale': .6, 'fast_computations': (False,False,False)}
        ys = priors.fast_gp.get_batch_first(100000,20,num_features, hyperparameters=hps)[1]
        # num_border_list = [1000,10000]
        num_borders = 1000
        batch_fraction = 8
        draw_flag = False
        data_augment = True
        lr = 0.0008
        epochs = 625
        root_dir = f'/home/ypq/TransformersCanDoBayesianInference/myresults/GPfitting_augmentTrue_{num_features}feature'
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
    
    # time consumption in one iteration
    # n_init_list = [50,100,200,400,800,1600,3200,6400,12800]
    # # n_init_list = [3200]
    # ac = 'EI'
    # iter_step = 1
    # repeat_num = 10
    # out_root_path = "./numerical_results/time_comparison"
    # out_path = f"{out_root_path}/time_per_it_feature{num_features}_repeat{repeat_num}.xlsx"

    # results = {}
    # results['n_init'] = n_init_list
    # results['PT time/it mean'] = [0]*(len(n_init_list))
    # results['PT time/it min'] = [np.inf]*(len(n_init_list))
    # results['PT time/it max'] = [-np.inf]*(len(n_init_list))
    # results['GP time/it mean'] = [0]*(len(n_init_list))
    # results['GP time/it min'] = [np.inf]*(len(n_init_list))
    # results['GP time/it max'] = [-np.inf]*(len(n_init_list))
    # for i in range(len(n_init_list)):
    #     n_init = n_init_list[i]
    #     for n in tqdm(range(repeat_num)):
    #         PTBO = PTBayesianOptimization(function, bounds, model,n_init=n_init,ac=ac)
    #         GPBO = BayesianOptimization(function, bounds,n_init=n_init,ac=ac)
    #         t1 = time()
    #         PT_x_max = PTBO.optimize(n_iter=iter_step)
    #         t2 = time()
    #         print(f"{n_init} PT time/it: {t2-t1}s")
    #         t3 = time()
    #         GP_x_max = GPBO.optimize(n_iter=iter_step)
    #         t4 = time()
    #         print(f"{n_init} GP time/it: {t4-t3}s")
    #         results['PT time/it mean'][i] += (t2 - t1)
    #         results['GP time/it mean'][i] += (t4 - t3)
    #         results['PT time/it min'][i] = min(results['PT time/it min'][i], t2-t1)
    #         results['GP time/it min'][i] = min(results['GP time/it min'][i], t4-t3)
    #         results['PT time/it max'][i] = max(results['PT time/it max'][i], t2-t1)
    #         results['GP time/it max'][i] = max(results['GP time/it max'][i], t4-t3)
    #     results['PT time/it mean'][i] /= repeat_num
    #     results['GP time/it mean'][i] /= repeat_num
    # df = pd.DataFrame(results)
    # df.to_excel(out_path,index=False)
    
    
    # simple regret: regret value versus time
    # 不同维度(5,20,40)，不同初始点数量(800,1600)，不同函数(rastrigin,ackley) 
    n_init = 3200
    ac = 'EI'
    iter_step = 1
    repeat_num = 1

    time_step = 10 # 1s for 800 and 4s for 1600
    total_time = 100 # s
    time_length = total_time // time_step
    func_type="unimodel"
    out_root_path = "./numerical_results/time_comparison"
    out_path = f"{out_root_path}/simple_regret_vstime_{func_type}_feature{num_features}_init{n_init}_time{total_time}_step{time_step}_repeat{repeat_num}.xlsx"
    writer = pd.ExcelWriter(out_path)
    functions = Function(func_type)
    functions = functions()
    max_values = {"quadratic":0,"exponential":np.e * num_features,"log":np.log(2) * num_features,"rosenbrock":0,"rastrigin":0,"ackley":0}
    scale_factors = {"quadratic":0.25*num_features,"exponential":(np.e-1)*num_features,"log":np.log(2) * num_features,\
                    "rosenbrock":((num_features-1)*100+num_features//2),"rastrigin":20.25*num_features,"ackley":4.7}
    for function_index in range(0,3):
        function = list(functions.values())[function_index]
        function_name = list(functions.keys())[function_index]
        max_value = max_values[function_name]
        scale_factor = scale_factors[function_name]
        results = {}
        results['time'] = [i for i in range(time_step,total_time+time_step,time_step)]
        results['PT mean'] = [0]*(time_length)
        results['PT min'] = [np.inf]*(time_length)
        results['PT max'] = [-np.inf]*(time_length)
        results['GP mean'] = [0]*(time_length)
        results['GP min'] = [np.inf]*(time_length)
        results['GP max'] = [-np.inf]*(time_length)
        for n in tqdm(range(repeat_num)):
            x = [np.array([np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(num_features)]) for _ in range(n_init)]
            y = [function(i) for i in x]
            init_point = (x[:],y[:])
            print(len(y))

            PTBO = PTBayesianOptimization(function, bounds, model,n_init=n_init,ac=ac,init_point=init_point)
            t1 = time()
            t2 = time()
            i = 0
            iter_num = 0
            while (t2-t1) < total_time:
                PT_x_max = PTBO.optimize(n_iter=iter_step)
                t2 = time()
                iter_num += 1
                if i < time_length and \
                 (t2-t1) >= results['time'][i]:
                    while i < time_length and \
                    (t2-t1) >= results['time'][i]:
                        i += 1
                    results['PT mean'][i-1] += (max_value-function(PT_x_max)[0])/scale_factor
                    results['PT min'][i-1] = min(results['PT min'][i-1], (max_value-function(PT_x_max)[0])/scale_factor)
                    results['PT max'][i-1] = max(results['PT max'][i-1], (max_value-function(PT_x_max)[0])/scale_factor)
                    print(f"{function_name} PT: {t2-t1}s; regret value: {(max_value-function(PT_x_max)[0])/scale_factor}")
                    print(f"iter_num:{iter_num}")
            init_point = (x[:],y[:])
            print(len(y))
            GPBO = BayesianOptimization(function, bounds,n_init=n_init,ac=ac,init_point=init_point)
            t3 = time()
            t4 = time()
            i = 0
            iter_num = 0
            while (t4-t3) < total_time:
                GP_x_max = GPBO.optimize(n_iter=iter_step)
                t4 = time()
                iter_num += 1
                if i < time_length and \
                (t4-t3) >= results['time'][i]:
                    while i < time_length and \
                    (t4-t3) >= results['time'][i]:
                        i += 1
                    results['GP mean'][i-1] += (max_value-function(GP_x_max)[0])/scale_factor
                    results['GP min'][i-1] = min(results['GP min'][i-1], (max_value-function(GP_x_max)[0])/scale_factor)
                    results['GP max'][i-1] = max(results['GP max'][i-1], (max_value-function(GP_x_max)[0])/scale_factor)
                    print(f"{function_name} GP : {t4-t3}s; regret value: {(max_value-function(GP_x_max)[0])/scale_factor}")
                    print(f"iter_num:{iter_num}")

        tmp1 = [results['PT mean'][i] / repeat_num for i in range(total_time//time_step)]
        tmp2 = [results['GP mean'][i] / repeat_num for i in range(total_time//time_step)]
        results['PT mean'] = tmp1
        results['GP mean'] = tmp2
        df = pd.DataFrame(results)
        df = df.replace(0,np.nan)
        df = df.fillna(method='ffill')
        df_name = pd.DataFrame({function_name:[]})
        df_name.to_excel(writer,index=False,startrow=(function_index)*(total_time//time_step + 3))
        df.to_excel(writer,index=False,startrow=(function_index)*(total_time//time_step + 3)+1)
    writer.save()