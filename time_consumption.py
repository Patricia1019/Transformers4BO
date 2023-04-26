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
    
    # time consumption in one iteration
    n_init_list = [50,100,200,400,800,1600,3200,6400,12800]
    # n_init_list = [50,100]
    ac = 'EI'
    iter_step = 1
    repeat_num = 10
    out_root_path = "./numerical_results/time_comparison"
    out_path = f"{out_root_path}/time_per_it_feature{num_features}_repeat{repeat_num}.xlsx"

    results = {}
    results['n_init'] = n_init_list
    results['PT time/it mean'] = [0]*(len(n_init_list))
    results['PT time/it min'] = [np.inf]*(len(n_init_list))
    results['PT time/it max'] = [-np.inf]*(len(n_init_list))
    results['GP time/it mean'] = [0]*(len(n_init_list))
    results['GP time/it min'] = [np.inf]*(len(n_init_list))
    results['GP time/it max'] = [-np.inf]*(len(n_init_list))
    for i in range(len(n_init_list)):
        n_init = n_init_list[i]
        for n in tqdm(range(repeat_num)):
            PTBO = PTBayesianOptimization(function, bounds, model,n_init=n_init,ac=ac)
            GPBO = BayesianOptimization(function, bounds,n_init=n_init,ac=ac)
            t1 = time()
            PT_x_max = PTBO.optimize(n_iter=iter_step)
            t2 = time()
            print(f"{n_init} PT time/it: {t2-t1}s")
            t3 = time()
            GP_x_max = GPBO.optimize(n_iter=iter_step)
            t4 = time()
            print(f"{n_init} GP time/it: {t4-t3}s")
            results['PT time/it mean'][i] += (t2 - t1)
            results['GP time/it mean'][i] += (t4 - t3)
            results['PT time/it min'][i] = min(results['PT time/it min'][i], t2-t1)
            results['GP time/it min'][i] = min(results['GP time/it min'][i], t4-t3)
            results['PT time/it max'][i] = max(results['PT time/it max'][i], t2-t1)
            results['GP time/it max'][i] = max(results['GP time/it max'][i], t4-t3)
        results['PT time/it mean'][i] /= repeat_num
        results['GP time/it mean'][i] /= repeat_num
    df = pd.DataFrame(results)
    df.to_excel(out_path,index=False)
    
    
    # simple regret: regret value versus time
