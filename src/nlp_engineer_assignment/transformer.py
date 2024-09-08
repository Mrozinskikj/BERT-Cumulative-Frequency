import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from nlp_engineer_assignment.utils import print_line

class BERTEmbedding(nn.Module):
    """
    A class for a BERT Embedding layer which creates and combines token and position embeddings.

    Attributes
    ----------
    length : int
        Expected length of input strings. Defaults to 20.
    token_embedding : nn.Embedding
        Embedding layer which maps each token to a dense vector of size 'embed_dim'.
    position_embedding : nn.Embedding
        Embedding layer which maps each position index to a dense vector of size 'embed_dim'.
    dropout : nn.Dropout
        Dropout layer for regularisation.

    Methods
    -------
    forward(input_ids: torch.Tensor) -> torch.Tensor
        Performs a forward pass, computing the BERT embeddings used as model input for a given 'input_ids'.
    """
    def __init__(self, vocab_size: int, length: int, embed_dim: int, dropout: int):
        """
        Initialises the BERT Embedding.

        Parameters
        ----------
        vocab_size : int
            Total number of unique tokens.
        length : int
            Expected length of input strings.
        embed_dim : int
            Dimensionality of the token and position embeddings.
        dropout : int
            Dropout probability, used for regularisation.
        """
        super().__init__() # initialise the nn.Module parent class
        self.length = length # store the sequence length

        self.token_embedding = nn.Embedding(vocab_size, embed_dim) # map each token to a dense vector of size embed_dim
        self.position_embedding = nn.Embedding(length, embed_dim) # map each position index to a dense vector of size embed_dim
        self.dropout = nn.Dropout(dropout) # dropout layer for regularisation


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass, computing the BERT embeddings used as model input for a given 'input_ids'.

        Parameters
        ----------
        input_ids : torch.Tensor (shape [batch_size, length])
            The tensor containing token indices for the input sequences of a given batch.

        Returns
        -------
        torch.Tensor  (shape [batch_size, length, embed_dim])
            The tensor containing the BERT embeddings for the input sequences of a given batch.
        """
        device = input_ids.device # used to ensure all tensors are on same device

        token_embedding = self.token_embedding(input_ids) # look up token embeddings for each token in input_ids

        position_input = torch.arange(self.length, device=device).unsqueeze(0) # create position indices for each token
        position_embedding = self.position_embedding(position_input) # look up position embeddings for each position index in input_ids
        
        embedding = token_embedding + position_embedding # BERT embedding is element-wise sum of token embeddings and position embeddings
        embedding = self.dropout(embedding) # apply dropout for regularisation
        return embedding


class BERT(nn.Module):
    """
    A class for a BERT model, used to classify the cumulative frequencies of the respective character of every 'input_ids' item.

    Attributes
    ----------
    embedding : BERTEmbedding
        Embedding layer which combines token and position embeddings.
    encoder_block : nn.TransformerEncoder
        Transformer Encoder.
    classifier : nn.Linear
        Output layer, predicting classes 0, 1, 2 for cumulative character frequency for each position in sequence

    Methods
    -------
    forward(input_ids: torch.Tensor) -> torch.Tensor
        Performs a forward pass, computing the logits for each class of each item of 'input_ids'.
    """
    def __init__(self, vocab_size: int = 27, length: int = 20, embed_dim: int = 768, dropout: int = 0.1, attention_heads: int = 12, layers: int = 2):
        """
        Initialises the BERT Model.

        Parameters
        ----------
        vocab_size : int, optional
            Total number of unique tokens. Defaults to 27.
        length : int, optional
            Expected length of input strings. Defaults to 20.
        embed_dim : int, optional
            Dimensionality of the token and position embeddings. Defaults to 768.
        dropout : int, optional
            Dropout probability, used for regularisation. Defaults to 0.1.
        attention_heads : int, optional
            The number of attention heads in the Transformer encoder layer. Defaults to 12.
        layers : int, optional
            The number of Transformer encoder layers. Defaults to 2.
        """
        super().__init__()  # initialise the nn.Module parent class
        
        self.embedding = BERTEmbedding(vocab_size, length, embed_dim, dropout) # embedding layer which combines token and position embeddings
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, attention_heads, dim_feedforward=embed_dim * 4, dropout=dropout, activation="gelu") # instance of transformer encoder layer
        self.encoder_block = nn.TransformerEncoder(encoder_layer, layers) # full transformer encoder consisting of multiple layers

        self.classifier = nn.Linear(embed_dim, 3) # output layer, predicting classes 0, 1, 2 for each position in sequence


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass, computing the logits for each class of each item of 'input_ids'.

        Parameters
        ----------
        input_ids : torch.Tensor (shape [batch_size, length])
            The tensor containing token indices for the input sequences of a given batch.

        Returns
        -------
        torch.Tensor  (shape [batch_size, length, 3 (classes)])
            The tensor containing the class logits for each item of the input sequences of a given batch.
        """
        embeddings = self.embedding(input_ids) # get embeddings for each token in input_ids
        embeddings = embeddings.transpose(0, 1) # rearrange embeddings from [batch_size, length, embed_dim] to [length, batch_size, embed_dim] for encoder block

        encoder_output = self.encoder_block(embeddings) # pass embeddings through transformer encoder block

        logits = self.classifier(encoder_output.transpose(0, 1)) # apply classifier to each position to get logits for each class
        return logits


def evaluate(model : BERT, dataset_test: dict, loss_fn: nn.CrossEntropyLoss, plot_data: dict, step_current: int, step_total: int) -> float:
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
    print(f'step: {step_current}/{step_total} eval loss: {round(loss_average,2)}')
    return plot_data


def plot_train(plot_data: dict):
    fig, axs = plt.subplots(len(plot_data.keys()), 1, figsize=(8, 6), sharex=True) # create subplots

    for p,plot in enumerate(plot_data.keys()): # plot x,y of each subplot in plot_data
        axs[p].plot(plot_data[plot]['x'],plot_data[plot]['y'])
        axs[p].set_title(plot)
    
    plt.tight_layout()
    plt.show()


def train_classifier(dataset_train: dict, dataset_test: dict, learning_rate: float = 1e-6, epochs: int = 1, eval_every: int = 250, print_train: bool = False, plot: bool = True) -> BERT:
    """
    Creates and trains a BERT model for cumulative frequency classification given a training dataset.

    Parameters
    ----------
    dataset_train : dict
        A dictionary containing the inputs and labels of the training data.
        - 'input_ids' : torch.Tensor (shape [num_batches, batch_size, tensor_length])
            The batched tensor of tokenised input strings.
        - 'labels' : torch.Tensor (shape [num_batches, batch_size, tensor_length])
            The batched tensor of labels corresponding to input IDs.
    dataset_train : dict
        A dictionary containing the inputs and labels of the test data.
        Refer to 'dataset_train'.
    learning_rate : float, optional
        The learning rate for the optimiser (magnitiude of weight updates per step). Defaults to 1e-6.
    epochs : int, optional
        The number of epochs for training. Each epoch corresponds to one full iteration through training data. Defaults to 1.
    print_train : bool, optional
        Whether to print the training state at every training step. Defaults to False.
    plot : bool, optional
        Whether to display a plot of the training timeline once training is finished. Defaults to True.

    Returns
    -------
    BERT
        The trained BERT model.
    """
    plot_data = {'train':{'x':[],'y':[]}, 'test':{'x':[],'y':[]}} # dict storing x,y plot data for training progress
    
    model = BERT() # initialise model
    model.train() # set model to training mode

    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate) # initialise AdamW optimiser
    loss_fn = nn.CrossEntropyLoss() # initialise cross-entropy loss function for classification

    print("Beginning Training.")
    print_line()

    batches = len(dataset_train['input_ids']) # number of batches in the training dataset
    for epoch in range(epochs): # iterate through epochs
        for batch in range(batches): # iterate through batches in epoch
            step_current = batch*(epoch+1)
            step_total = batches*epochs
            
            if batch%eval_every == 0: # perform evaluation on test split at set intervals
                plot_data = evaluate(model, dataset_test, loss_fn, plot_data, step_current, step_total)

            logits = model(dataset_train['input_ids'][batch]) # forward pass to compute logits
            logits = logits.view(-1, logits.size(-1)) # flatten batch dimension: [batch_size * length, classes]
            labels = dataset_train['labels'][batch].view(-1) # flatten batch dimension: [batch_size * length]
            
            loss = loss_fn(logits, labels) # calculate loss between output logits and labels
            
            optimiser.zero_grad() # zero the gradients from previous step (no gradient accumulation)
            loss.backward() # backpropagate to compute gradients
            optimiser.step() # update model weights

            plot_data['train']['x'].append(step_current)
            plot_data['train']['y'].append(loss.item())
            if print_train:
                print(f'step: {step_current}/{step_total} train loss: {round(loss.item(),2)}')
    
    if batch%eval_every != 0: # perform final evaluation (as long as not already performed on this step)
        plot_data = evaluate(model, dataset_test, loss_fn, plot_data, step_current, step_total)
    print("Finishing Training.")
    print_line()
    if plot:
        plot_train(plot_data)
    return model