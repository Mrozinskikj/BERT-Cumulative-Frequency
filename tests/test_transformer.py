import torch
import torch.nn as nn
from nlp_engineer_assignment.transformer import BERT, BERTEmbedding
from nlp_engineer_assignment import Tokeniser, process_dataset

tokeniser = Tokeniser(length=20)
model = BERT()


def test_bert_embedding():
    """Tests that BERTEmbedding returns a correctly shaped tensor"""
    embedding_layer = BERTEmbedding(vocab_size=27, length=20, embed_dim=768, dropout=0.1)
    input_ids = torch.randint(0, 27, (4, 20))
    try:
        embeddings = embedding_layer(input_ids)
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (4, 20, 768), f"Expected shape (4, 20, 768), but got {embeddings.shape}"
    except Exception as e:
        assert False, f"BERTEmbedding raised an exception: {e}"


def test_bert_forward():
    """Tests that BERT model forward pass returns a correctly shaped tensor"""
    input_ids = torch.randint(0, 27, (4, 20))
    try:
        logits = model(input_ids)
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (4, 20, 3), f"Expected shape (4, 20, 3), but got {logits.shape}"
    except Exception as e:
        assert False, f"BERT model raised an exception: {e}"