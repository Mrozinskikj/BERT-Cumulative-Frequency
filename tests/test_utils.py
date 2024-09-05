import numpy as np
import torch

from nlp_engineer_assignment import count_letters, score, Tokeniser, process_dataset


def test_count_letters():
    assert np.array_equal(count_letters("hello"), np.array([0, 0, 0, 1, 0]))
    assert np.array_equal(count_letters("world"), np.array([0, 0, 0, 0, 0]))
    assert np.array_equal(
        count_letters("hello hello"),
        np.array([0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1])
    )


def test_score():
    assert score(np.array([[0, 1, 1, 0, 1]]),
                 np.array([[0, 1, 1, 0, 1]])) == 1.0
    assert score(np.array([[0, 1, 1, 0, 1]]),
                 np.array([[1, 1, 0, 0, 1]])) == 0.6
    assert score(np.array([[0, 1, 1, 0, 1]]),
                 np.array([[0, 0, 0, 0, 0]])) == 0.4
    assert score(np.array([[0, 1, 1, 0, 1]]),
                 np.array([[1, 0, 0, 1, 0]])) == 0.0


def test_tokeniser_length_correct():
    """Tests that tokeniser encoding outputs tensor of correct shape"""
    tokeniser = Tokeniser(length=20)
    string = "yaraku is a japanese"
    tensor = tokeniser.encode(string)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (20,)


def test_tokeniser_length_incorrect():
    """Tests that tokeniser encoding catches input of incorrect length"""
    tokeniser = Tokeniser(length=20)
    string = "hello world"
    try:
        tokeniser.encode(string)
        assert False, "Expected ValueError, but no error was raised."
    except ValueError:
        assert True


def test_tokeniser_out_of_vocab():
    """Tests that tokeniser encoding catches out-of-vocabulary in input"""
    tokeniser = Tokeniser(length=20)
    string = "Yaraku is a Japanese"
    try:
        tokeniser.encode(string)
        assert False, "Expected ValueError, but no error was raised."
    except Exception as e:
        assert isinstance(e, ValueError), f"Expected ValueError, but got {type(e).__name__}."


def test_tokeniser_decode():
    """Tests that tokeniser decoding outputs correct string"""
    tokeniser = Tokeniser(length=20)
    string = tokeniser.decode(torch.tensor([24, 0, 17, 0, 10, 20, 26, 8, 18, 26, 0, 26, 9, 0, 15, 0, 13, 4, 18, 4]))
    assert string == "yaraku is a japanese"


def test_process_dataset_correct_batches():
    """Tests that dataset gets processed into batched input and label tensors of correct shape"""
    inputs = [
        "heir average albedo ",
        "ed by rank and file ",
        "s can also extend in",
        "erages between nine ",
    ]
    tokeniser = Tokeniser(length=20)
    dataset = process_dataset(inputs, tokeniser, batch_size=2)
    assert dataset['input_ids'].shape == (2, 2, 20), f"Expected shape (2, 2, 20), but got {dataset['input_ids'].shape}"
    assert dataset['labels'].shape == (2, 2, 20), f"Expected shape (2, 2, 20), but got {dataset['labels'].shape}"


def test_process_dataset_trim_items():
    """Tests that inputs of uneven length get appropriately trimmed"""
    inputs = [
        "heir average albedo ",
        "ed by rank and file ",
        "s can also extend in",
    ]
    tokeniser = Tokeniser(length=20)
    dataset = process_dataset(inputs, tokeniser, batch_size=2)
    assert dataset['input_ids'].shape == (1, 2, 20), f"Expected shape (1, 2, 20), but got {dataset['input_ids'].shape}"
    assert dataset['labels'].shape == (1, 2, 20), f"Expected shape (1, 2, 20), but got {dataset['labels'].shape}"


def test_process_dataset_empty():
    """Tests that empty or fully trimmed inputs are caught"""
    inputs = [
        "heir average albedo ",
    ]
    tokeniser = Tokeniser(length=20)
    try:
        process_dataset(inputs, tokeniser, batch_size=2)
        assert False, "Expected ValueError, but no error was raised."
    except Exception as e:
        assert isinstance(e, ValueError), f"Expected ValueError, but got {type(e).__name__}."