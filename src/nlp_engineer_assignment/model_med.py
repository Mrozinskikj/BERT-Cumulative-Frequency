import torch
import torch.nn as nn
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
    def __init__(
        self,
        embed_dim: int,
        dropout: float,
        vocab_size: int,
        length: int,
    ):
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
        dropout : float
            Dropout probability, used for regularisation.
        """
        super().__init__() # initialise the nn.Module parent class
        self.length = length # store the sequence length

        self.token_embedding = nn.Embedding(vocab_size, embed_dim) # map each token to a dense vector of size embed_dim
        self.position_embedding = nn.Embedding(length, embed_dim) # map each position index to a dense vector of size embed_dim
        self.dropout = nn.Dropout(dropout) # dropout layer for regularisation


    def forward(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
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


class TransformerLayer(nn.Module):
    """
    A class for a single Transformer layer composed of multi-head attention, normalisation, and feed-forward layers.
    
    Attributes
    ----------
    attention : nn.MultiheadAttention
        Attention mechanism capturing the relationships between each item in the input sequence.
    layer_norm1 : nn.LayerNorm
        Normalisation of the attention sub-layer, for stability.
    feedforward : nn.Sequential
        Two layer deep feed-forward network to process the attention sub-layer.
    layer_norm2 : nn.LayerNorm
        Normalisation of the feed-forward sub-layer, for stability.
    dropout : nn.Dropout
        Dropout layer for regularisation.

    Methods
    -------
    forward(input_ids: torch.Tensor) -> torch.Tensor
        Performs a forward pass, computing the intermediate transformer output representation.
    """
    def __init__(self, embed_dim: int, attention_heads: int, dropout: float):
        """
        Initialises the BERT Model.

        Parameters
        ----------
        embed_dim : int
            Dimensionality of the embeddings.
        dropout : float
            Dropout probability, used for regularisation.
        attention_heads : int
            The number of attention heads in the Transformer encoder layer.
        """
        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim, attention_heads, dropout=dropout, batch_first=True) # attention mechanism capturing relationships between each item in input
        self.layer_norm1 = nn.LayerNorm(embed_dim) # normalisation for stability
        
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        ) # 2 layer deep feed-forward network
        self.layer_norm2 = nn.LayerNorm(embed_dim) # normalisation for stability
        self.dropout = nn.Dropout(dropout) # dropout for regularisation
    

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass, computing the intermediate transformer output representation.

        Parameters
        ----------
        input_tensor : torch.Tensor (shape [batch_size, length, embed_dim])
            The transformer input tensor.

        Returns
        -------
        torch.Tensor (shape [batch_size, length, embed_dim])
            The transformer output tensor.
        """
        attn_output, _ = self.attention(input_tensor, input_tensor, input_tensor) # compute the attention scores
        attn_output = input_tensor + self.dropout(attn_output) # residual connection and dropout
        attn_output = self.layer_norm1(attn_output) # layer normalisation

        ffwd_output = self.feedforward(attn_output) # process through feed-forward network
        ffwd_output = attn_output + self.dropout(ffwd_output) # residual connection and dropout
        output_tensor = self.layer_norm2(ffwd_output) # layer normalisation
        
        return output_tensor


class BERT(nn.Module):
    """
    A class for a BERT model, used to classify the cumulative frequencies of the respective character of every 'input_ids' item.

    Attributes
    ----------
    embedding : BERTEmbedding
        Embedding layer which combines token and position embeddings.
    transformer_layers : nn.ModuleList
        A list of TransformerLayer modules. Input is fed through each layer in sequence.
    classifier : nn.Linear
        Output layer, predicting classes 0, 1, 2 for cumulative character frequency for each position in sequence

    Methods
    -------
    forward(input_ids: torch.Tensor) -> torch.Tensor
        Performs a forward pass, computing the logits for each class of each item of 'input_ids'.
    """
    def __init__(
        self,
        embed_dim: int,
        dropout: float,
        attention_heads: int,
        layers: int,
        vocab_size: int = 27,
        length: int = 20,
    ):
        """
        Initialises the BERT Model.

        Parameters
        ----------
        embed_dim : int
            Dimensionality of the token and position embeddings.
        dropout : float
            Dropout probability, used for regularisation.
        attention_heads : int
            The number of attention heads in the Transformer encoder layer.
        layers : int
            The number of Transformer encoder layers.
        vocab_size : int, optional
            Total number of unique tokens. Defaults to 27.
        length : int, optional
            Expected length of input strings. Defaults to 20.
        """
        super().__init__() # initialise the nn.Module parent class
        
        self.embedding = BERTEmbedding(embed_dim, dropout, vocab_size, length) # embedding layer which combines token and position embeddings
        
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim, attention_heads, dropout) for _ in range(layers)
        ]) # sequence of transformer layers

        self.classifier = nn.Linear(embed_dim, 3) # output layer, predicting classes 0, 1, 2 for each position in sequence
        
        print(f"Model created. Architecture:\n{self}")
        print_line()


    def forward(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
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

        for layer in self.transformer_layers: # feed input through each transformer layer in sequence
            embeddings = layer(embeddings)

        logits = self.classifier(embeddings) # apply classifier to each position to get logits for each class
        return logits