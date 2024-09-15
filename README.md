# BERT Cumulative Character Frequency Classification

This project implements and serves a BERT-based language model for classifying cumulative character frequencies. The model assigns one of three classes (0, 1, 2) to each character in a given input string, based on how many times the character has previously appeared in the string, with the frequency capped at 2.

### Example
```
Input string: yaraku is a japanese
Labels:       00010000012202020011
```
The model is implemented from scratching exclusively using basic PyTorch units such as ``nn.Linear`` and ``nn.Embedding``. The transformer encoder architecture includes custom implementations of the embedding layer, a multi-headed attention mechanism, and layer normalisation.
The model was trained to achieve a **99.6% accuracy** on validation data, and is provided in ``model/model.pth``.

## Project Structure

The project is divided into the following sections:
1. **Training**: Dataset tokenisation, model creation, and execution of the training process.
2. **Hyperparameter Tuning**: Optional bayesian optimisation process to find the optimal hyperparameter settings to minimise validation loss.
3. **Model Serving**: Deployment of the trained model using FastAPI, exposing a RESTful API for inference.

## Data

For simplicity, the model is constrained to work with training data examples which are exactly 20 characters long, containing only lowercase characters and spaces. Data is provided in ``data/train.txt`` and ``data/test.txt``, comprising of 10,000 and 1,000 examples respectively.

## Setup

### Prerequisites

- Python (created and tested with version `3.10.11`)
- Poetry (created and tested with version `1.8.3`)
- (Optional) Docker

### Setup

1. Clone the repository into your local environment.
2. Install poetry in your local environment by running: `pip install poetry`
3. Create the virtual environment for the project by running: `poetry install`
4. Initialise the virtual environment by running: `poetry shell`
5. Run the entrypoint script with: `python main.py`

This will train the model and deploy a FastAPI application hosted on ``0.0.0.0:8000``.

## Model Serving

The FastAPI application provides a RESTful endpoint for model inference. The API can be accessed and was tested via Swagger UI at ``http://localhost:8000/docs``.

### Example

- Endpoint: ``/prediction`` (POST)
- Request Body:
```
{
  "text": "yaraku is a japanaese"
}
```
- Response Body:
```
{
  "prediction": "00010000012202020011"
}

```

## Configuration

The ``config.yaml`` file allows for the customisation of the program execution. Various parameters are provided, including:
- Model hyperparameters.
- Whether to perform training or load a pre-trained model from a specified path.
- Whether to perform hyperparameter tuning.

## CUDA Support

Although the program is written to support CUDA, additional steps are required to enable it during execution due to the limitation of the Poetry dependency management system. By default, a CPU-only version of PyTorch is specified in ``pyproject.toml``. To enable CUDA, replace the default PyTorch dependency with the path to your local CUDA-enabled PyTorch.

## Testing

Unit tests are provided in the ``tests`` directory to ensure the correct functionality of various project components.
