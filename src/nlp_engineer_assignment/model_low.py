import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    A class for Layer Normalisation, used to normalise input tensors such that the embedding dimension (-1) has zero mean and unit variance.
    Learnable gain and bias parameters for each embedding element allow for increased flexibility for downstream tasks.
    Helps to stabilise learning by keeping weights within a controlled range.

    Attributes
    ----------
    epsilon : float
        Small constant preventing division by zero.
    gain : nn.Parameter
        Learnable gain (multiplier) parameters for each element in embedding dimension. Applied after normalisation.
    bias : nn.Parameter
        Learnable bias (addition) parameters for each element in embedding dimension. Applied after normalisation.

    Methods
    -------
    forward(input_ids: torch.Tensor) -> torch.Tensor
        Normalises and scales the embedding dimension of the input tensor.
    """
    def __init__(
        self,
        embed_dim: int,
        epsilon: float = 1e-5
    ):
        """
        Initialises the LayerNorm module.

        Parameters
        ----------
        embed_dim : int
            The size of the embedding dimension. Used to correctly initialise gain and bias parameters.
        epsilon : float, optional
            Small constant preventing division by zero. Defaults to 1e-5.
        """
        super().__init__() # initialise the nn.Module parent class
        self.epsilon = epsilon # small constant prevents division by zero
        self.gain = nn.Parameter(torch.ones(embed_dim)) # learnable gain (multiplier) parameters for each element in embed_dim
        self.bias = nn.Parameter(torch.zeros(embed_dim)) # learnable bias (addition) parameters for each element in embed_dim


    def forward(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalises and scales the embedding dimension of the input tensor.

        Parameters
        ----------
        inputs : torch.Tensor (shape [batch_size, length, embed_dim])
            The input tensor to be normalised across 'embed_dim'.

        Returns
        -------
        torch.Tensor (shape [batch_size, length, embed_dim])
            The normalised and scaled input tensor. Prior to scaling, 'embed_dim' has zero mean and unit variance.
        """
        mean = inputs.mean(dim=-1, keepdim=True) # compute the mean across the embedding dimension (-1)
        variance = inputs.var(dim=-1, keepdim=True, unbiased=True) # compute the unbiased variance (average of squared deviations from mean) across the embedding dimension (-1)
        std = torch.sqrt(variance + self.epsilon) # calculate standard deviation

        normalised = (inputs - mean) / std # normalise inputs to mean 0 and standard deviation 1 (unbiased variance means std=1)
        scaled = normalised * self.gain + self.bias # normalised tensor is shifted and scaled by learnable parameters. increased flexibility
        
        return scaled


class AttentionHead(nn.Module):
    """
    A class for an individual attention head within a MultiHeadAttention module.
    Projects input embeddings into keys, values, and queries, then computes attention scores.

    Attributes
    ----------
    head_size : int
        Dimension of each of key, value, and query. Calculated by MultiHeadAttention module based on embed_dim and num_heads.
    key : nn.Linear
        Linear transformation for projecting input tensor into key space.
    query : nn.Linear
        Linear transformation for projecting input tensor into query space.
    value : nn.Linear
        Linear transformation for projecting input tensor into value space.
    dropout : nn.Dropout
        Dropout layer for regularisation.

    Methods
    -------
    forward(input_tensor: torch.Tensor) -> torch.Tensor
        Performs a forward pass, computing the attention scores.
    """
    def __init__(
        self,
        embed_dim: int,
        head_size: int,
        dropout: float
    ):
        """
        Initialises the AttentionHead module.

        Parameters
        ----------
        embed_dim : int
            The size of the embedding dimension.
        head_size : int
            Dimension of each of key, value, and query. Calculated by MultiHeadAttention module based on embed_dim and num_heads.
        dropout : float
            Dropout probability, used for regularisation.
        """
        super().__init__() # initialise the nn.Module parent class

        self.head_size = head_size

        # keys queries and values are projected from embedding dimension to 'head_size'
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        self.dropout = nn.Dropout(dropout) # dropout for regularisation
    
    
    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs a forward pass, computing the attention scores.

        Parameters
        ----------
        input_tensor : torch.Tensor (shape [batch_size, length, embed_dim])
            The transformer input tensor for which attention needs to be calculated.

        Returns
        -------
        torch.Tensor (shape [batch_size, length, head_size])
            The attention scores.
        """
        # project input tensor to 'head_size' for keys, queries, and values ([batch_size, length, embed_dim] -> [batch_size, length, head_size])
        key = self.key(input_tensor)
        query = self.query(input_tensor)
        value = self.value(input_tensor)

        scores = torch.matmul(query, key.transpose(-2,-1)) # attention scores are dot product between query and key ([batch_size, length, head_size] x [batch_size, head_size, length ] -> [batch_size, length, length])
        scores = scores / (self.head_size**0.5) # divide by sqrt of head size to normalise to unit variance. increases stability- high variance would make softmax sharp

        attn_weights = nn.functional.softmax(scores, dim=-1) # convert scores into probability distribution

        output = torch.matmul(attn_weights, value) # output is the weighted sum of values
        output = self.dropout(output) # apply dropout for regularisation
        return output


class MultiHeadAttention(nn.Module):
    """
    A class for processing an input tensor with multiple attention heads in parallel.
    Attention head output is recombined with a linear transformation.

    Attributes
    ----------
    num_heads : int
        Number of attention heads. Must be a factor of 'embed_dim'.
    heads : nn.ModuleList
        List containing all 'num_heads' intances of 'AttentionHead'.
    linear : nn.Linear
        Linear transformation to re-integrate attenion head outputs into a unified representation.

    Methods
    -------
    forward(input_tensor: torch.Tensor) -> torch.Tensor
        Performs a forward pass, processing the input_tensor through multiple attention heads.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float
    ):
        """
        Initialises the MultiHeadAttention module.

        Parameters
        ----------
        embed_dim : int
            The size of the embedding dimension.
        num_heads : int
            Number of attention heads. Must be a factor of 'embed_dim'.
        dropout : float
            Dropout probability, used for regularisation.
        """
        super().__init__() # initialise the nn.Module parent class

        if embed_dim % num_heads != 0: # ensure each attention head gets an equal distribution of the input tensor
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.num_heads = num_heads
        head_size = embed_dim // num_heads # size of each attention head is such that the concatenation of all attention heads is embed_dim

        self.heads = nn.ModuleList([
            AttentionHead(embed_dim, head_size, dropout) for _ in range(num_heads)
        ]) # list of all attention heads

        self.linear = nn.Linear(embed_dim, embed_dim) # linear transformation
    
    
    def forward(
        self,
        input_tensor
    ) -> torch.Tensor:
        """
        Performs a forward pass, processing the input_tensor through multiple attention heads.

        Parameters
        ----------
        input_tensor : torch.Tensor (shape [batch_size, length, embed_dim])
            The input tensor to be processed by the attention mechanism.

        Returns
        -------
        torch.Tensor (shape [batch_size, length, embed_dim])
            The final output tensor after attention computation and reintegration.
        """
        head_outputs = [head(input_tensor) for head in self.heads] # compute the attention scores of each head in parallel
        concatenated = torch.cat(head_outputs, dim=-1) # concatenate all attention head outputs back into single tensor
        output = self.linear(concatenated) # re-integrate attention heads into unified representation with final linear transformation
        return output


class TransformerLayer(nn.Module):
    """
    A class for a single Transformer layer composed of multi-head attention, normalisation, and feed-forward layers.
    
    Attributes
    ----------
    attention : MultiHeadAttention
        Attention mechanism capturing the relationships between each item in the input sequence.
    layer_norm1 : LayerNorm
        Normalisation of the attention sub-layer, for stability.
    feedforward : nn.Sequential
        Two layer deep feed-forward network to process the attention sub-layer. Uses GELU activation as per BERT paper.
    layer_norm2 : LayerNorm
        Normalisation of the feed-forward sub-layer, for stability.
    dropout : nn.Dropout
        Dropout layer for regularisation.

    Methods
    -------
    forward(input_ids: torch.Tensor) -> torch.Tensor
        Performs a forward pass, computing the intermediate transformer output representation.
    """
    def __init__(
        self,
        embed_dim: int,
        attention_heads: int,
        dropout: float
    ):
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

        self.attention = MultiHeadAttention(embed_dim, attention_heads, dropout) # attention mechanism capturing relationships between each item in input
        self.layer_norm1 = LayerNorm(embed_dim) # normalisation for stability
        
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        ) # 2 layer deep feed-forward network
        self.layer_norm2 = LayerNorm(embed_dim) # normalisation for stability
        self.dropout = nn.Dropout(dropout) # dropout for regularisation
    

    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
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
        attn_output = self.attention(input_tensor) # compute the attention scores
        attn_output = input_tensor + self.dropout(attn_output) # residual connection and dropout
        attn_output = self.layer_norm1(attn_output) # layer normalisation

        ffwd_output = self.feedforward(attn_output) # process through feed-forward network
        ffwd_output = attn_output + self.dropout(ffwd_output) # residual connection and dropout
        output_tensor = self.layer_norm2(ffwd_output) # layer normalisation
        
        return output_tensor


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