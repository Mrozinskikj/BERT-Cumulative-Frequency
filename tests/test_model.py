import torch
import torch.nn as nn
from nlp_engineer_assignment.model_low import BERT, BERTEmbedding, LayerNorm, AttentionHead, MultiHeadAttention, TransformerLayer


def test_layer_norm():
    """Tests that LayerNorm forward pass works and returns a tensor of correct shape, and that output mean and standard deviation are close to 0 and 1 respectively."""
    torch.manual_seed(0)
    batch_size, length, embed_dim = 2, 4, 8
    input_tensor = torch.randn(batch_size, length, embed_dim)
    layer_norm = LayerNorm(embed_dim)
    output_tensor = layer_norm(input_tensor)

    assert output_tensor.size() == input_tensor.size(), f"Expected shape ({batch_size}, {length}, {embed_dim}), but got {output_tensor['input_ids'].shape}"
    assert torch.allclose(output_tensor.mean(dim=-1), torch.zeros(batch_size, length), atol=1e-5), "Mean should be close to 0"
    assert torch.allclose(output_tensor.std(dim=-1), torch.ones(batch_size, length), atol=1e-5), "Std should be close to 1"


def test_attention_head():
    """Tests that AttentionHead forward pass works and returns a tensor of correct shape."""
    torch.manual_seed(0)
    batch_size, length, embed_dim = 2, 4, 8
    head_size = 5
    input_tensor = torch.randn(batch_size, length, embed_dim)
    attention_head = AttentionHead(embed_dim, head_size, dropout=0.1)
    output_tensor = attention_head(input_tensor)

    assert output_tensor.size() == (batch_size, length, head_size), f"Expected shape ({batch_size}, {length}, {head_size}), but got {output_tensor['input_ids'].shape}"


def test_multi_head_attention():
    """Tests that MultiHeadAttention forward pass works and returns a tensor of correct shape."""
    torch.manual_seed(0)
    batch_size, length, embed_dim = 2, 4, 8
    num_heads = 2
    input_tensor = torch.randn(batch_size, length, embed_dim)
    attention = MultiHeadAttention(embed_dim, num_heads, dropout=0.1)
    output_tensor = attention(input_tensor)

    assert output_tensor.size() == (batch_size, length, embed_dim), f"Expected shape ({batch_size}, {length}, {num_heads}), but got {output_tensor['input_ids'].shape}"


def test_multi_head_attention_indivisible():
    """Tests that MultiHeadAttention catches embed_dim being indivisible by num_heads."""
    torch.manual_seed(0)
    embed_dim = 8
    num_heads = 3
    
    try:
        MultiHeadAttention(embed_dim, num_heads, dropout=0.1)
        assert False, "Expected ValueError, but no error was raised."
    except Exception as e:
        assert isinstance(e, ValueError), f"Expected ValueError, but got {type(e).__name__}."


def test_transformer_layer():
    """Tests that TransformerLayer forward pass works and returns a tensor of correct shape."""
    torch.manual_seed(0)
    batch_size, length, embed_dim = 2, 4, 8
    num_heads = 2
    input_tensor = torch.randn(batch_size, length, embed_dim)
    transformer_layer = TransformerLayer(embed_dim, num_heads, dropout=0.1)
    output_tensor = transformer_layer(input_tensor)

    assert output_tensor.size() == input_tensor.size(), f"Expected shape ({batch_size}, {length}, {embed_dim}), but got {output_tensor['input_ids'].shape}"


def test_bert_embedding():
    """Tests that BERTEmbedding forward pass works and returns a tensor of correct shape."""
    torch.manual_seed(0)
    batch_size, length, embed_dim = 2, 4, 8
    vocab_size = 27
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, length))
    bert_embedding = BERTEmbedding(embed_dim, dropout=0.1, vocab_size=vocab_size, length=length)
    output_tensor = bert_embedding(input_ids)

    assert output_tensor.size() == (batch_size, length, embed_dim), f"Expected shape ({batch_size}, {length}, {embed_dim}), but got {output_tensor['input_ids'].shape}"


def test_bert():
    """Tests that BERT forward pass works and returns a tensor of correct shape."""
    torch.manual_seed(0)
    batch_size, length, embed_dim = 2, 4, 8
    dropout = 0.1
    attention_heads = 2
    layers = 2
    vocab_size = 27
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, length))
    bert = BERT(embed_dim, dropout, attention_heads, layers, vocab_size, length)
    output_tensor = bert(input_ids)

    assert output_tensor.size() == (batch_size, length, 3), f"Expected shape ({batch_size}, {length}, {3}), but got {output_tensor['input_ids'].shape}"
