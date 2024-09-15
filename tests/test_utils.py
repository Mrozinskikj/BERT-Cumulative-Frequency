import numpy as np
import torch

from bert_cumulative_frequency import count_letters, score


def test_count_letters():
    assert torch.equal(count_letters("hello"), torch.tensor([0, 0, 0, 1, 0]))
    assert torch.equal(count_letters("world"), torch.tensor([0, 0, 0, 0, 0]))
    assert torch.equal(
        count_letters("hello hello"),
        torch.tensor([0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1])
    )


def test_score():
    assert score(torch.tensor([[0, 1, 1, 0, 1]]),
                 torch.tensor([[0, 1, 1, 0, 1]])) == 1.0
    assert score(torch.tensor([[0, 1, 1, 0, 1]]),
                 torch.tensor([[1, 1, 0, 0, 1]])) == 0.6
    assert score(torch.tensor([[0, 1, 1, 0, 1]]),
                 torch.tensor([[0, 0, 0, 0, 0]])) == 0.4
    assert score(torch.tensor([[0, 1, 1, 0, 1]]),
                 torch.tensor([[1, 0, 0, 1, 0]])) == 0.0