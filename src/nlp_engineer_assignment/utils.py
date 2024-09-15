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
    _, axs = plt.subplots(len(plot_data.keys()), 1, figsize=(8, 6), sharex=True) # create subplots

    for p,plot in enumerate(plot_data.keys()): # plot x,y of each subplot in plot_data
        axs[p].plot(plot_data[plot]['x'],plot_data[plot]['y'])
        axs[p].set_title(plot)
    
    plt.tight_layout()
    plt.show()