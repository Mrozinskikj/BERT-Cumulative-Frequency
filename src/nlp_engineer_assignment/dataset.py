import torch
import random
from nlp_engineer_assignment import count_letters, print_line


class Tokeniser:
    """
    A class for encoding and decoding strings into tokens for model input.

    Attributes
    ----------
    length : int
        Expected length of input strings. Defaults to 20.
    char_to_id : dict
        Dictionary mapping characters to their corresponding token IDs.
    id_to_char : dict
        Dictionary mapping token IDs to their corresponding characters.

    Methods
    -------
    encode(string: str) -> torch.Tensor
        Encodes a string into a tensor of token IDs.
    
    decode(tokens: torch.Tensor) -> str
        Decodes a tensor of token IDs into a string.
    """
    def __init__(
        self,
        length: int = 20,
    ):
        """
        Initialises the tokeniser, defining the vocabulary.

        Parameters
        ----------
        length : int, optional
            Expected length of input strings. Defaults to 20.
        """
        self.length = length
        
        vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' '] # vocab of lowerchase chars and space

        self.char_to_id = {ch: i for i, ch in enumerate(vocab)} # dictionary of character to token id
        self.id_to_char = {i: ch for i, ch in enumerate(vocab)} # dictionary of token id to character
    
    
    def encode(
        self,
        string: str,
    ) -> torch.Tensor:
        """
        Encodes a string into a tensor of token IDs.

        Parameters
        ----------
        string : str
            The input string to encode.
        
        Returns
        -------
        torch.Tensor (shape [length])
            A tensor containing the token IDs corresponding to input string.
            
        Raises
        ------
        ValueError
            If 'string' is not 'self.length' characters long.
            If 'string' contains out-of-vocabulary characters.
        """
        if len(string) != self.length: # ensure input string is correct length
            raise ValueError(f"Input string must be exactly {self.length} characters long, but got {len(string)} characters.")
        
        try:
            tokens_list = [self.char_to_id[c] for c in string] # convert string to tokens list
        except KeyError as e:
            raise ValueError(f"Out of vocabulary character encountered: '{e.args[0]}'")
        
        tokens_tensor = torch.tensor(tokens_list, dtype=torch.long) # convert token list into tensor
        return tokens_tensor
    
    
    def decode(
        self,
        tokens: torch.Tensor,
    ) -> str:
        """
        Decodes a tensor of token IDs into a string.

        Parameters
        ----------
        tokens : torch.Tensor (shape [length])
            A tensor containing token IDs to decode.
        
        Returns
        -------
        str
            A decoded string corresponding to input tokens.
        """
        return "".join([self.id_to_char[i.item()] for i in tokens])


def batch_tensor(
    tensor_list: list,
    batch_size: int,
) -> torch.Tensor:
    """
    Converts a list of 1D tensors into a batched 3D tensor. Used with 'process_dataset'.

    Parameters
    ----------
    tensor_list : list of torch.Tensor
        A list of 1D tensors to be batched together.
    batch_size : int
        The number of tensors to include in each batch.
    
    Returns
    -------
    torch.Tensor (shape [num_batches, batch_size, tensor_length])
        A 3D batched tensor, grouping each input tensor into groups of size 'batch_size'.
    """
    tensor_stacked = torch.stack(tensor_list) # convert list of 1D tensors to stacked 2D tensor
    
    num_batches = len(tensor_stacked) // batch_size # find whole number of batches (may trim last items)
    excess_items = len(tensor_stacked) % batch_size # calculate number of extra items which don't fit into batches
    if excess_items != 0:
        print(f"Trimming last {excess_items} items to ensure equal batch sizes.")
        tensor_stacked = tensor_stacked[:-excess_items] # trim tensor
    
    batched_tensor = tensor_stacked.view(num_batches, batch_size, -1) # reshape 2D tensor into batched 3D tensor
    return batched_tensor
    

def process_dataset(
    inputs: list, 
    tokeniser: Tokeniser,
    batch_size: int,
    device: torch.device,
) -> dict:
    """
    Processes raw data into input tokens and labels, creating a dataset dictionary of batched tensors.

    Parameters
    ----------
    inputs : list of str
        Train or test data examples split into a list.
    tokeniser : Tokeniser
        An instance of the Tokeniser class used to encode the input.
    batch_size : int
        The number of items to include in each batch.
    device : torch.device
        The device on which to place the tensors (CPU or GPU).

    Returns
    -------
    dict
        - 'input_ids' : torch.Tensor (shape [num_batches, batch_size, tensor_length])
            The batched tensor of tokenised input strings.
        - 'labels' : torch.Tensor (shape [num_batches, batch_size, tensor_length])
            The batched tensor of labels corresponding to input IDs.
    
    Raises
    ------
    ValueError
        If length of 'inputs' is less than 'batch_size'.
    """
    
    if len(inputs) < batch_size:
        raise ValueError("Input list is too short for a single batch.")

    random.shuffle(inputs) # shuffle incase inputs are ordered
    input_ids_list = [tokeniser.encode(text) for text in inputs] # list of token tensors for each input
    labels_list = [count_letters(text) for text in inputs] # list of label tensors for each input

    # create dictionary of batched 3D input and label tensors
    dataset = {
        'input_ids': batch_tensor(input_ids_list, batch_size).to(device),
        'labels': batch_tensor(labels_list, batch_size).to(device)
    }
    print("Dataset created.", ", ".join([f"{key}: {tensor.size()}" for key, tensor in dataset.items()]))
    print_line()
    return dataset