from BOfrom_scratch import *
import pdb
from tqdm import tqdm

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
    
    # simple regret: different objective functions for PT model and GP model
    iter_num = 100
    n_init = 50
    ac = 'EI'
    iter_step = 5
    repeat_num = 10
    objective_function = 'multimodel'
    noisy = False
    out_root_path = f'./numerical_results/time_results_comparison'
    if not os.path.exists(out_root_path):
        os.makedirs(out_root_path)

    out_path = f'{out_root_path}/simple_regret_{objective_function}_feature{num_features}_noisy{noisy}_total{iter_num}_init3200_repeat{repeat_num}.xlsx'
    writer = pd.ExcelWriter(out_path)
    functions = Function(type=objective_function,noisy=noisy)
    functions = functions()
    max_values = {"quadratic":0,"exponential":np.e * num_features,"log":np.log(2) * num_features,"rosenbrock":0,"rastrigin":0,"ackley":0}
    scale_factors = {"quadratic":0.25*num_features,"exponential":(np.e-1)*num_features,"log":np.log(2) * num_features,\
                    "rosenbrock":((num_features-1)*100+num_features//2),"rastrigin":20.25*num_features,"ackley":4.7}
    for function_index in range(0,3):
        function = list(functions.values())[function_index]
        function_name = list(functions.keys())[function_index]
        results = {}
        results['time'] = [i for i in range(1,iter_num+1,iter_step)]
        results['PT regret value'] = [0]*(iter_num//iter_step)
        results['GP regret value'] = [0]*(iter_num//iter_step)
        max_value = max_values[function_name]
        scale_factor = scale_factors[function_name]
        for n in tqdm(range(repeat_num)):
            x = [np.array([np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(num_features)]) for _ in range(n_init)]
            y = [function(i) for i in x]
            init_point = (x[:],y[:])
            PTBO = BayesianOptimization(function, bounds,n_init=n_init,ac=ac,init_point=init_point)
            GPBO = BayesianOptimization(function, bounds,n_init=n_init,ac=ac,init_point=init_point)
            for i in range(1,iter_num+1,iter_step):
                PT_x_max = PTBO.optimize(n_iter=iter_step*10)
                GP_x_max = GPBO.optimize(n_iter=iter_step)
                print(f"{function_name} PT regret value: ")
                print((max_value-function(PT_x_max))/scale_factor)
                print(f"{function_name} GP regret value: ")
                print((max_value-function(GP_x_max))/scale_factor)
                results['PT regret value'][i//iter_step] += (max_value-function(PT_x_max)[0])/scale_factor
                results['GP regret value'][i//iter_step] += (max_value-function(GP_x_max)[0])/scale_factor
        tmp1 = [results['PT regret value'][i]/repeat_num for i in range(iter_num//iter_step)]
        tmp2 = [results['GP regret value'][i]/repeat_num for i in range(iter_num//iter_step)]
        results['PT regret value'] = tmp1
        results['GP regret value'] = tmp2
        df = pd.DataFrame(results)
        df_name = pd.DataFrame({function_name:[]})
        df_name.to_excel(writer,index=False,startrow=function_index*(iter_num//iter_step + 3))
        df.to_excel(writer,index=False,startrow=function_index*(iter_num//iter_step + 3)+1)
    writer.save()