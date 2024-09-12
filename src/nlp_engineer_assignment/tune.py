import torch
from skopt import gp_minimize
from skopt.space import Real, Integer
from functools import partial

from nlp_engineer_assignment import BERT, train_classifier, print_line

def objective(params_list, param_names, dataset_train, dataset_test):
    params = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'seed': 0,
        'batch_size': 4,
        'learning_rate': 1e-5,
        'epochs': 1,
        'warmup_ratio': 0.1,
        'eval_every': 250,
        'embed_dim': 768,
        'dropout': 0.1,
        'attention_heads': 2,
        'layers': 2
    }
    params_dict = dict(zip(param_names, params_list))
    params.update(params_dict)

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
            print_train=False,
            plot=False
        )
    except ValueError as e:
        print(f"Invalid parameter combination: {e}")
        return float('inf')

    return loss


def tune_hyperparameters(dataset_train, dataset_test):
    search_space = {
    'learning_rate': Real(1e-6, 1e-3, prior='log-uniform'),
    'dropout': Real(0.0, 0.5),
    'layers': Integer(1, 8)
    }
    param_names = list(search_space.keys())
    dimensions = list(search_space.values())
    
    objective_data = partial(objective, param_names=param_names, dataset_train=dataset_train, dataset_test=dataset_test)
    
    print("Beginning hyperparameter tuning.")
    print_line()
    
    result = gp_minimize(
        func=objective_data,
        dimensions=dimensions,
        n_calls=20,
        random_state=0
    )
    print(result)