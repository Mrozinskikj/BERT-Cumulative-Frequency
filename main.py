import os
import uvicorn
import random
import torch

from nlp_engineer_assignment import read_inputs, test_accuracy,\
train_classifier, Tokeniser, process_dataset, BERT, load_model, save_model


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
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    
    random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    

    tokeniser = Tokeniser()

    inputs_train = read_inputs(os.path.join(cur_dir, "data", "train.txt"))
    dataset_train = process_dataset(
        inputs_train,
        tokeniser,
        params['batch_size'],
        params['device']
    )
    
    inputs_test = read_inputs(os.path.join(cur_dir, "data", "test.txt"))
    dataset_test = process_dataset(
        inputs_test,
        tokeniser,
        params['batch_size'],
        params['device']
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
        model = train_classifier(
            model,
            dataset_train,
            dataset_test,
            params['learning_rate'],
            params['epochs'],
            params['warmup_ratio'],
            params['eval_every'],
            print_train=False,
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