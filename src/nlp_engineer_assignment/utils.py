import numpy as np
import torch
import matplotlib.pyplot as plt


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
        logits = model(batch) # derive the logits of one batch of inputs
        prediction = torch.argmax(logits, dim=-1) # prediction is the highest value logit for each item in sequence
        predictions_list.append(prediction)
    
    predictions = torch.stack(predictions_list).view(1000, 20) # convert list to tensor and flatten batch dimension
    labels = dataset_test['labels'].view(1000, 20) # flatten batch dimension of labels
    
    print(f"Test Accuracy: {(100.0 * score(predictions, labels)):.1f}%") # calculate score
    print_line()
    

def plot_train(plot_data: dict):
    """
    Displays a plot of the training timeline for various variables.

    Parameters
    ----------
    plot_data : dict
        A dictionary of x and y timeline data of training progress.
        - 'train' : dict
            Timeline data for the training loss.
            - 'x': list
                A list of x-coordinate values, representing the given training step.
            - 'y': list
                A list of y-coordinate values, representing the value at the given training step.
        - 'test' : dict
            Timeline data for the validation loss.
            Refer to 'train'.
        - 'lr' : dict
            Timeline data for the learning rate.
            Refer to 'train'.
    """
    fig, axs = plt.subplots(len(plot_data.keys()), 1, figsize=(8, 6), sharex=True) # create subplots

    for p,plot in enumerate(plot_data.keys()): # plot x,y of each subplot in plot_data
        axs[p].plot(plot_data[plot]['x'],plot_data[plot]['y'])
        axs[p].set_title(plot)
    
    plt.tight_layout()
    plt.show()
    

def load_model(
    model: 'BERT',
    model_path: str,
    device : torch.device
    ) -> 'BERT':
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
    ) -> 'BERT':
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