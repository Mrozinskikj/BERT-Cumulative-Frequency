import torch

from bert_cumulative_frequency.utils import score, print_line
from bert_cumulative_frequency import BERT


def load_model(
    model: 'BERT',
    model_path: str,
    device : torch.device
) -> BERT:
    """
    Loads a model saved in a local directory.

    Parameters
    ----------
    model : BERT
        An instance of the model class, not containing the saved parameters.
    model_path : str
        The path of the saved model parameters, to be loaded into 'model'.
    device : torch.device
        The device on which to load the model (CPU or GPU).
    """
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    print(f"Model successfully loaded from {model_path}")
    print_line()
    return model


def save_model(
    model: 'BERT',
    model_path: str
) -> BERT:
    """
    Saves a trained model to a local directory.

    Parameters
    ----------
    model : BERT
        An instance of the trained model to be saved.
    model_path : str
        The to save the trained model parameters into.
    """
    print(f"Model successfully saved to {model_path}")
    print_line()
    torch.save(model.state_dict(), model_path)


def predict(
    input_ids: torch.Tensor,
    model: BERT
) -> torch.Tensor:
    """
    Compute the predictions of a batch of examples by feeding the input through the model and finding the max logits.

    Parameters
    ----------
    model : BERT
        The BERT model for computing predictions with.
    input_ids : torch.Tensor (shape [batch_size, length])
        The tensor containing token indices for the input sequences of a given batch.
    
    Returns
    -------
    torch.Tensor  (shape [batch_size, length])
        The class labels corresponding to each input.
    """
    logits = model(input_ids) # derive the logits of a batch of inputs
    prediction = torch.argmax(logits, dim=-1) # prediction is the highest value logit for each item in sequence
    return prediction


def test_accuracy(
    model: 'BERT',
    dataset_test: torch.Tensor
):
    """
    Compute the model predictions of every example in 'dataset_test' and calculate score comparing to ground truth.

    Parameters
    ----------
    model : BERT
        The BERT model for computing predictions with.
    labels : dict
        - 'input_ids' : torch.Tensor (shape [num_batches, batch_size, tensor_length])
            The batched tensor of tokenised input strings.
        - 'labels' : torch.Tensor (shape [num_batches, batch_size, tensor_length])
            The batched tensor of labels corresponding to input IDs.
    """
    print("Beginning evaluation...")
    predictions_list = [] # list to store every batch of predictions
    for batch in dataset_test['input_ids']:
        predictions_list.append(predict(batch, model))
    
    predictions = torch.stack(predictions_list).view(1000, 20) # convert list to tensor and flatten batch dimension
    labels = dataset_test['labels'].view(1000, 20) # flatten batch dimension of labels
    
    print(f"Test Accuracy: {(100.0 * score(predictions, labels)):.1f}%") # calculate score
    print_line()