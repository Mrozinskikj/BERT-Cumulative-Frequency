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
    """
    Loads configuration parameters from the YAML file.
    
    Parameters
    ----------
    config_path : str
        File path to configuration YAML file.

    Returns
    -------
    dict
        Dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_model(config) -> BERT:
    """
    Loads dataset, creates a BERT model, optionally tunes hyperparameters, trains, and finally tests model accuracy.

    Parameters
    ----------
    config : dict
        Dictionary containing configuration parameters.

    Returns
    -------
    BERT
        The trained BERT model, ready for serving with API.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__)) # get current directory
    
    params = config['train'] # extract training parameters
    
    random.seed(params['seed']) # set random seed for reproducibility
    torch.manual_seed(params['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set device to GPU if available, else CPU

    dataset_train, dataset_test = load_data(
        os.path.join(cur_dir, config['data']['train_path']),
        os.path.join(cur_dir, config['data']['test_path']),
        params['batch_size'],
        device
    ) # load train and test datasets from file into tensor format
    
    if config['tune']['tune']: # perform hyperparameter tuning if specified by config
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
    ).to(device) # initialise model with specified parameters

    if config['model']['load']: # load model weights from local file if specified by config
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
        ) # train the model
        if config['model']['save']:
            save_model(model, os.path.join(cur_dir, config['model']['path'])) # save the model to local file if specified by config

    test_accuracy(model, dataset_test) # evaluate the model to get final percentage accuracy
    return model  


if __name__ == "__main__":
    config = get_config("config.yaml") # load configuration from config.yaml

    model = prepare_model(config) # train or load the model based on the configuration

    app = create_app(model) # create the FastAPI application to serve the model
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1
    ) # run the application using Uvicorn