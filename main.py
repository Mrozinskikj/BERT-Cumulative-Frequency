import os
import uvicorn
import random
import torch
import yaml

from nlp_engineer_assignment import (
    test_accuracy,
    train_classifier,
    BERT,
    load_model,
    save_model,
    tune_hyperparameters,
    load_data,
    create_app
)


def get_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_model(config):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    
    params = config['train']
    
    random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_train, dataset_test = load_data(
        os.path.join(cur_dir, config['data']['train_path']),
        os.path.join(cur_dir, config['data']['test_path']),
        params['batch_size'],
        device
    )
    
    if config['tune']['tune']:
        params = tune_hyperparameters(
            config['tune']['sample_space'],
            params,
            dataset_train,
            dataset_test,
            config['tune']['iterations'],
            device,
            config['plot']['tune'],
        )
    
    model = BERT(
        params['embed_dim'],
        params['dropout'],
        params['attention_heads'],
        params['layers'],
    ).to(device) # initialise model

    if config['model']['load']:
        model = load_model(model, os.path.join(cur_dir, config['model']['path']), device)
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
            plot=config['plot']['train'],
        )
        if config['model']['save']:
            save_model(model, os.path.join(cur_dir, config['model']['path']))

    test_accuracy(model, dataset_test)
    return model  


if __name__ == "__main__":
    config = get_config("config.yaml")

    model = train_model(config)

    app = create_app(model)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1
    )