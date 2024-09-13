import torch
from skopt import gp_minimize
from skopt.space import Real, Integer
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence
from functools import partial
import time

from nlp_engineer_assignment import BERT, train_classifier, print_line

def objective(params_list, params, param_names, dataset_train, dataset_test, counter):

    params_dict = dict(zip(param_names, params_list))
    params_dict_formatted = {k:f"{v:.2e}" if isinstance(v, float) else v for k,v in params_dict.items()} # round all floats in the dictionary
    params.update(params_dict)

    counter['iteration'] += 1
    print(f"{counter['iteration']}/{20}: {params_dict_formatted}")
    start_time = time.time()

    
    try:
        model = BERT(
            embed_dim=params['embed_dim'],
            dropout=params['dropout'],
            attention_heads=params['attention_heads'],
            layers=params['layers']
        )
        
        _, loss = train_classifier(
            model=model,
            dataset_train=dataset_train,
            dataset_test=dataset_test,
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            warmup_ratio=params['warmup_ratio'],
            eval_every=params['eval_every'],
            allow_print=False,
            plot=False
        )
    except ValueError as e:
        print(f"Invalid parameter combination: {e}")
        return float('inf')
    
    print(f"{counter['iteration']}/{20} loss: {round(loss,2)}, Time taken: {(time.time()-start_time):.2f} seconds.")
    return loss


def tune_hyperparameters(
        search_space,
        params,
        dataset_train,
        dataset_test,
        seed,
        iterations,
        plot
    ):
    counter = {'iteration': 0, 'total': iterations} # must store iterations counter as mutable data type

    param_names = list(search_space.keys())
    dimensions = list(search_space.values())
    
    objective_data = partial(
        objective,
        params=params,
        param_names=param_names,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        counter=counter
    )
    
    start_time = time.time()
    print("Beginning hyperparameter tuning.")
    print_line()
    
    result = gp_minimize(
        func=objective_data,
        dimensions=dimensions,
        n_calls=iterations,
        random_state=seed
    )
    
    params_dict = dict(zip(param_names, result.x))
    params_dict_formatted = {k:f"{v:.2e}" if isinstance(v, float) else v for k,v in params_dict.items()} # round all floats in the dictionary
    print_line()
    print(f"Finishing hyperparameter tuning. Total time taken: {(time.time()-start_time):.2f} seconds.")
    print(f"Optimal hyperparameters: {params_dict_formatted}\nLoss {round(result.fun,2)}")
    print_line()

    if plot:
        plot_convergence(result)
        plt.show()

    params.update(params_dict) # update params with optimals and return
    return params