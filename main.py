import os
import uvicorn

from nlp_engineer_assignment import read_inputs, test_accuracy,\
train_classifier, Tokeniser, process_dataset
    

def train_model():
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    ###
    # Setup
    ###

    tokeniser = Tokeniser()

    inputs_train = read_inputs(os.path.join(cur_dir, "data", "train.txt"))
    dataset_train = process_dataset(inputs_train, tokeniser)
    
    inputs_test = read_inputs(os.path.join(cur_dir, "data", "test.txt"))
    dataset_test = process_dataset(inputs_test, tokeniser)

    model = train_classifier(dataset_train, dataset_test)

    test_accuracy(model, dataset_test)


if __name__ == "__main__":
    train_model()
    uvicorn.run(
        "nlp_engineer_assignment.api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1
    )