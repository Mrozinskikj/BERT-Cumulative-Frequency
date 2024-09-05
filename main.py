import numpy as np
import os
import uvicorn

from nlp_engineer_assignment import count_letters, print_line, read_inputs, \
    score, train_classifier, Tokeniser, process_dataset


def train_model():
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    ###
    # Setup
    ###

    tokeniser = Tokeniser()

    inputs_train = read_inputs(
        os.path.join(cur_dir, "data", "train.txt")
    )
    dataset_train = process_dataset(inputs_train, tokeniser)

    inputs_test = read_inputs(
        os.path.join(cur_dir, "data", "test.txt")
    )
    dataset_test = process_dataset(inputs_test, tokeniser)

    model = train_classifier(inputs_train)

    # TODO: Extract predictions from the model and save it to a
    # variable called `predictions`. Observe the shape of the
    # example random predictions.
    golds = np.stack([count_letters(text) for text in test_inputs])
    predictions = np.random.randint(0, 3, size=golds.shape)

    # Print the first five inputs, golds, and predictions for analysis
    for i in range(5):
        print(f"Input {i+1}: {test_inputs[i]}")
        print(
            f"Gold {i+1}: {count_letters(test_inputs[i]).tolist()}"
        )
        print(f"Pred {i+1}: {predictions[i].tolist()}")
        print_line()

    print(f"Test Accuracy: {100.0 * score(golds, predictions):.2f}%")
    print_line()


if __name__ == "__main__":
    train_model()
    uvicorn.run(
        "nlp_engineer_assignment.api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1
    )
