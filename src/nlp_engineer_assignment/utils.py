import numpy as np
import torch


def count_letters(text: str) -> torch.Tensor:
    """
    Count the number of times each letter appears in the text up to that point

    Parameters
    ----------
    text : str
        The text to count the letters in

    Returns
    -------
    torch.Tensor
        A tensor of counts, one for each letter in the text
    """
    output = np.zeros(len(text), dtype=np.int32)
    for i in range(0, len(text)):
        output[i] = min(2, len([c for c in text[0:i] if c == text[i]]))

    return torch.tensor(output, dtype=torch.long) # array into tensor


def print_line():
    """
    Print a line of dashes
    """
    print("-" * 80)


def read_inputs(path: str) -> list:
    """
    Read the inputs from a file

    Parameters
    ----------
    path : str
        Path to the file

    Returns
    -------
    list
        A list of strings, one for each line in the file
    """
    lines = [line[:-1] for line in open(path, mode="r")]
    print(f"{len(lines)} lines read")
    print_line()
    return lines


def score(
    predictions: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """
    Compute the accuracy of the predictions

    Parameters
    ----------
    predictions : torch.Tensor (shape [examples, length])
        Predicted labels.
    labels : torch.Tensor (shape [examples, length])
        Ground truth labels.

    Returns
    -------
    float
        Accuracy of the predictions
    """
    return torch.sum(labels == predictions).float() / labels.numel()


def test_accuracy(model: 'BERT', dataset_test: torch.Tensor):
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
    predictions_list = [] # list to store every batch of predictions
    for batch in dataset_test['input_ids']:
        logits = model(batch) # derive the logits of one batch of inputs
        prediction = torch.argmax(logits, dim=-1) # prediction is the highest value logit for each item in sequence
        predictions_list.append(prediction)
    
    predictions = torch.stack(predictions_list).view(1000, 20) # convert list to tensor and flatten batch dimension
    labels = dataset_test['labels'].view(1000, 20) # flatten batch dimension of labels
    
    print(f"Test Accuracy: {100.0 * score(predictions, labels):.2f}%") # calculate score
    print_line()