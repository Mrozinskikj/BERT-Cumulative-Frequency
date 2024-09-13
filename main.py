import os
import uvicorn
import random
import torch
from skopt.space import Real, Integer, Categorical

from nlp_engineer_assignment import test_accuracy,\
train_classifier, BERT, load_model, save_model, tune_hyperparameters, load_data


def train_model():
    
    should_load = True
    should_save = True
    
    model_path = 'data/model.pth'
    
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
    
    should_tune = True
    iterations = 10
    sample_space = {
        'learning_rate': Real(1e-6, 1e-2, prior='log-uniform'),
        'dropout': Real(0.0, 0.5),
        'layers': Integer(1, 2),
        'attention_heads': Categorical([1,2]),
        'embedding_dim': Categorical([(i+1)*12 for i in range(3)]) # sample all factors of 12, divisible by attenion_heads
    }
    
    random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    
    path_train = "data/train.txt"
    path_test = "data/test.txt"
    dataset_train, dataset_test = load_data(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), path_train),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), path_test),
        params['batch_size'],
        params['device']
    )
    
    if should_tune:
        params = tune_hyperparameters(
            sample_space,
            params,
            dataset_train,
            dataset_test,
            params['seed'],
            iterations,
            plot=True,
        )
    
    model = BERT(
        params['embed_dim'],
        params['dropout'],
        params['attention_heads'],
        params['layers'],
    ).to(params['device']) # initialise model

    if should_load:
        model = load_model(model, os.path.join(cur_dir, model_path), params['device'])
    else:
        model, _ = train_classifier(
            model,
            dataset_train,
            dataset_test,
            params['learning_rate'],
            params['epochs'],
            params['warmup_ratio'],
            params['eval_every'],
            allow_print=True,
            plot=True,
        )
        if should_save:
            save_model(model, os.path.join(cur_dir, model_path))
            

    test_accuracy(model, dataset_test)


if __name__ == "__main__":
    train_model()
    uvicorn.run(
        "nlp_engineer_assignment.api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1
    )