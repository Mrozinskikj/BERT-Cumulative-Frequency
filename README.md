# BERT Cumulative Character Frequency Classification

This project implements and serves a BERT-based language model for predicting cumulative character frequencies. The model classifies each character in a given input string into one of three classes (0,1,2) based on how many times it has previously occurred in the string, capped at 2.
```
Input string: yaraku is a japanese
Labels:       00010000012202020011
```
Emphasis was placed on implementing the model from scratch. The BERT model architecture is comprised of only basic PyTorch units such as ``nn.Linear`` and ``nn.Embedding``. The model implementation is provided in ``src\model.py``, including custom implementations of layer normalisation, multi-headed attention, and transformer encoder.
A model was trained to achieve **99.6% accuracy** on validation data, and is provided in ``model/model.pth``. 

# Overview

The project is split into three sections:
1. Training
2. Hyperparameter Tuning
3. Serving the model

For the sake of simplicity, the model is constrained to only work with training data examples which are exactly 20 characters long, only using lowercase characters and space. Example data is provided in ``data/train.txt`` and ``data/test.txt``, comprising of 10000 and 1000 examples respectively.

There is an optional hyperparameter tuning stage, where a bayesian optimisation process is performed to find the optimal hyperparameter combination such that the validation loss is minimised.

Once the model is trained, it is deployed using a FastAPI application, where a RESTful endpoint is exposed for inference.

Unit tests are provided in ``tests`` to prove the functionality of various parts of the project.

## Requirements

- Python (created and tested with version `3.10.5`)
- Poetry (created and tested with version `1.6.1`)
- (Optional) Docker

## Setup

1. Start by cloning the repository into your local environment.
2. Install poetry in your local environment by running: `pip install poetry`
3. Create the virtual environment for the project by running: `poetry install`
4. Initialise the virtual environment by running: `poetry shell`
5. Run the entrypoint script with: `python main.py`

This train the model and deploy a FastAPI application, which is hosted on ``0.0.0.0:8000``. Swagger UI provides a visual interface to use the API. The API was verified to work by navigating to ``http://localhost:8000/docs`` and using the ``/prediction`` POST method, where a request body of this format is sent:
```
{
  "text": "yaraku is a japanaese"
}
```
The API returns a response body of this format:
```
{
  "prediction": "00010000012202020011"
}
```

A ``config.yaml`` file specifies every argument. It may be used to control hyperparameters, specify whether to perform training or load a local model, and specify whether to perform hyperparameter tuning, among others.

Although CUDA support is included, enabling CUDA requires extra steps as it is not natively supported by the poetry package manager. By default, a CPU-only version of torch is specified as a dependency within ``pyproject.toml``. CUDA may be enabled by removing this, and using your local machine's CUDA-enabled PyTorch version instead. This is not important for training the model as it should only take a few minutes, but hyperparameter tuning may take long without it.