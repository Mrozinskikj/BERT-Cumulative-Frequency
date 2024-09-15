import torch
import torch.nn as nn
from nlp_engineer_assignment.utils import print_line, plot_train
from nlp_engineer_assignment.model import BERT
import time


def lr_scheduler(
    warmup_ratio: float,
    step_current: int,
    step_total: int
) -> float:
    """
    Defines a custom learning rate scheduler (warmup and decay) to adjust learning rate based on current training step.

    Parameters
    ----------
    warmup_ratio : float
        The ratio of total training steps that learning rate warmup occurs for. 0 = no warmup, 1 = all warmup.
    step_current : int
        The current training step during evaluation.
    step_total : int
        The total number of training steps.

    Returns
    -------
    float
        The ratio that the learning rate will be multiplied by for the given training step.
    """
    warmup_steps = int(step_total*warmup_ratio)
    if step_current < warmup_steps: # LR warmup for initial steps
        return step_current/max(1,warmup_steps)
    else: # linear LR decay for remaining steps
        return (step_total-step_current) / max(1,step_total-warmup_steps)


def evaluate(
    model: BERT,
    dataset_test: dict,
    loss_fn: nn.CrossEntropyLoss,
    plot_data: dict,
    step_current: int,
    step_total: int,
    allow_print: bool
) -> float:
    """
    Peforms model evaluation by computing the average loss of the entire test dataset. The average loss is printed and 'plot_data' is updated.

    Parameters
    ----------
    model : BERT
        An instance of the BERT model to be evaluated.https://www.bing.com/search?&q=install+cuda
    dataset_test : dict
        A dictionary containing the inputs and labels of the test data.
        - 'input_ids' : torch.Tensor (shape [num_batches, batch_size, tensor_length])
            The batched tensor of tokenised input strings.
        - 'labels' : torch.Tensor (shape [num_batches, batch_size, tensor_length])
            The batched tensor of labels corresponding to input IDs.
    loss_fn : nn.CrossEntropyLoss
        The loss function used to compute the loss between the predictions and labels.
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
    step_current : int
        The current training step during evaluation.
    step_total : int
        The total number of training steps.
    allow_print : bool
        Whether to print the training state at every validation step.

    Returns
    -------
    dict
        The updated plot data dictionary with the validation loss added.
    float
        The final validation loss.
    """
    model.eval()  # set model to evaluation mode
    batches = len(dataset_test['input_ids']) # number of batches in the test dataset
    loss_total = 0

    with torch.no_grad():  # disable gradient calculation
        for batch in range(batches):
            
            logits = model(dataset_test['input_ids'][batch]) # forward pass to compute logits
            logits = logits.view(-1, logits.size(-1)) # flatten batch dimension: [batch_size * length, classes]
            labels = dataset_test['labels'][batch].view(-1) # flatten batch dimension: [batch_size * length]

            loss_batch = loss_fn(logits, labels) # calculate loss between output logits and labels
            loss_total += loss_batch.item()

    loss_average = loss_total / batches # loss is the average of all batches
    model.train() # revert model to training mode

    plot_data['test']['x'].append(step_current)
    plot_data['test']['y'].append(loss_average)
    if allow_print:
        print(f'step: {step_current}/{step_total} eval loss: {round(loss_average,2)}')
    return plot_data, loss_average


def train_classifier(
    model: BERT,
    dataset_train: dict,
    dataset_test: dict,
    learning_rate: float,
    epochs: int,
    warmup_ratio: float,
    eval_every: int,
    eval_first: bool = True,
    allow_print: bool = True,
    plot: bool = True
) -> BERT:
    """
    Creates and trains a BERT model for cumulative frequency classification given a training dataset.

    Parameters
    ----------
    model : BERT
        An instance of the BERT model to perform training on.
    dataset_train : dict
        A dictionary containing the inputs and labels of the training data.
        - 'input_ids' : torch.Tensor (shape [num_batches, batch_size, tensor_length])
            The batched tensor of tokenised input strings.
        - 'labels' : torch.Tensor (shape [num_batches, batch_size, tensor_length])
            The batched tensor of labels corresponding to input IDs.
    dataset_train : dict
        A dictionary containing the inputs and labels of the test data.
        Refer to 'dataset_train'.
    learning_rate : float
        The learning rate for the optimiser (magnitiude of weight updates per step).
    epochs : int
        The number of epochs for training. Each epoch corresponds to one full iteration through training data.
    warmup_ratio : float
        The ratio of total training steps that learning rate warmup occurs for. 0 = no warmup, 1 = all warmup.
    eval_every : int
        The step interval between evaluations on test dataset during training. If eval_every>=steps, only eval at the end.
    eval_first : bool, optional
        Whether to evaluate before the first training step. Defaults to False.
    
    allow_print : bool, optional
        Whether to print the training state at every validation step. Defaults to True.
    plot : bool, optional
        Whether to display a plot of the training timeline once training is finished. Defaults to True.

    Returns
    -------
    BERT
        The trained BERT model.
    float
        The final validation loss.
    """
    plot_data = {key: {'x':[], 'y':[]} for key in ['train','test','lr']} # dict storing x,y plot data for training progress
    
    model.train() # set model to training mode

    batches = len(dataset_train['input_ids']) # number of batches in the training dataset
    step_total = batches*epochs
    
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate) # initialise AdamW optimiser
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lambda step: lr_scheduler(warmup_ratio, step, step_total)) # create custom learning rate scheduler
    loss_fn = nn.CrossEntropyLoss() # initialise cross-entropy loss function for classification

    if allow_print:
        print("Beginning Training.")
        print_line()
    start_time = time.time()

    for epoch in range(epochs): # iterate through epochs
        for batch in range(batches): # iterate through batches in epoch
            step_current = batch*(epoch+1)
            
            if batch%eval_every == 0 and not (not eval_first and step_current==0): # perform evaluation on test split at set intervals
                plot_data, val_loss = evaluate(model, dataset_test, loss_fn, plot_data, step_current, step_total, allow_print)

            logits = model(dataset_train['input_ids'][batch]) # forward pass to compute logits
            logits = logits.view(-1, logits.size(-1)) # flatten batch dimension: [batch_size * length, classes]
            labels = dataset_train['labels'][batch].view(-1) # flatten batch dimension: [batch_size * length]
            
            loss = loss_fn(logits, labels) # calculate loss between output logits and labels
            
            optimiser.zero_grad() # zero the gradients from previous step (no gradient accumulation)
            loss.backward() # backpropagate to compute gradients
            optimiser.step() # update model weights
            scheduler.step() # update learning rate

            plot_data['train']['x'].append(step_current)
            plot_data['train']['y'].append(loss.item())
            plot_data['lr']['x'].append(step_current)
            plot_data['lr']['y'].append(scheduler.get_last_lr()[0])
    
    if batch%eval_every != 0: # perform final evaluation (as long as not already performed on this step)
        plot_data, val_loss = evaluate(model, dataset_test, loss_fn, plot_data, step_total, step_total, allow_print)
    if allow_print:
        print(f"Finishing Training. Time taken: {(time.time()-start_time):.2f} seconds.")
        print_line()
    if plot:
        plot_train(plot_data)
    return model, val_loss