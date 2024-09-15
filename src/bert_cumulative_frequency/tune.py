from skopt import gp_minimize
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence
from functools import partial
from skopt.space import Real, Integer, Categorical
import time
import math

from bert_cumulative_frequency import BERT, train_classifier, print_line


def build_sample_space(sample_space_config) -> tuple[list,list]:
    """
    Converts the sample space defined in the config file into skopt.space objects used by the optimiser.
    
    Parameters
    ----------
    sample_space_config : dict
        The dictionary defining the sample space from the the config file.
        Each key is the hyperparamater name, and the value defines the sample space for the parameter.
        - 'type' : str
            Specifies the type of sample space. May be either 'Real', 'Integer', or 'Categorical'.
            If 'Real' or 'Integer', expects 'low' and 'high' values. If 'Categorical', expects a 'categories' list.
    
    Returns
    -------
    tuple[list,list]
        - List of hyperparameter names.
        - skopt.space objects defining the sample space corresponding to the hyperparameter.
        Names and values separated due to skopt limitations.
    
    Raises
    ------
    ValueError
        If a parameter type defined in the config file is unrecognised.
    """
    param_names = []
    dimensions = []
    
    for param in sample_space_config:
        param_type = sample_space_config[param]['type']
        
        if param_type == 'Real': # create object for a real sample space
            dimension = Real(
                low=sample_space_config[param]['low'],
                high=sample_space_config[param]['high'],
                prior=sample_space_config[param].get('prior', 'uniform'), # optional parameter, defaults to 'uniform'
                base=sample_space_config[param].get('base', 10), # optional parameter, defaukts to 10
            )
        elif param_type == 'Integer': # create object for an integer sample space
            dimension = Integer(
                low=sample_space_config[param]['low'],
                high=sample_space_config[param]['high'],
            )
        elif param_type == 'Categorical': # create object for a categorical sample space
            dimension = Categorical(
                categories=sample_space_config[param]['categories'],
            )
        else: # catch unknown parameter types
            raise ValueError(f"Unknown parameter type: {param_type}")
        
        param_names.append(param)
        dimensions.append(dimension)
    return param_names, dimensions


def objective(
    params_list: list,
    param_names: list,
    params: dict,
    dataset_train: dict,
    dataset_test: dict,
    device,
    counter: dict
) -> float:
    """
    Objective function for the bayesian optimisation process.
    Given a list of hyperparameter values, the model gets trained and the final loss is returned.
    
    Parameters
    ----------
    params_list : list
        A list of hyperparameter values suggested by the optimiser.
    param_names : list
        A list of the associated parameter names for every value of 'params_list'. Names and values separated due to skopt limitations.
    params : dict
        A dictionary containing the default hyperparameters. Updated with 'params_list' during execution.
    dataset_train : dict
        A dictionary containing the inputs and labels of the training data.
        - 'input_ids' : torch.Tensor (shape [num_batches, batch_size, tensor_length])
            The batched tensor of tokenised input strings.
        - 'labels' : torch.Tensor (shape [num_batches, batch_size, tensor_length])
            The batched tensor of labels corresponding to input IDs.
    dataset_train : dict
        A dictionary containing the inputs and labels of the test data.
        Refer to 'dataset_train'.
    device : str
        The device that the model should be loaded on. 'cpu' or 'cuda'.
    counter : dict
        Used to keep train of current and total iterations. Mutable dictionary is used as it cannot be returned due to skopt limitations.
    
    Returns
    -------
    float
        The final objective value (loss) obtained by training the model with the given hyperparameters. Infinity if invalid parameter combination.
    
    Raises
    ------
    ValueError
        If an invalid hyperparameter combination provided (e.g. attention_heads not a factor of embed_dim).
    """
    params_dict = dict(zip(param_names, params_list)) # combine the paramater values suggested by the optimiser back into a dictionary with paramater names
    params_dict_formatted = {k:f"{v:.2e}" if isinstance(v, float) else v for k,v in params_dict.items()} # round all floats in the dictionary for print
    params.update(params_dict) # update all default hyperparameters with parameters suggested by optimiser

    counter['iteration'] += 1 # track current iteration and time
    print(f"{counter['iteration']}/{20}: {params_dict_formatted}")
    start_time = time.time()

    try:
        model = BERT(
            embed_dim=params['embed_dim'],
            dropout=params['dropout'],
            attention_heads=params['attention_heads'],
            layers=params['layers']
        ).to(device) # create model
        
        _, loss = train_classifier(
            model=model,
            dataset_train=dataset_train,
            dataset_test=dataset_test,
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            warmup_ratio=params['warmup_ratio'],
            eval_every=1e6, # do not perform eval
            eval_first=False,
            allow_print=False,
            plot=False
        ) # get validation loss after training
    except ValueError as e: # catch invalid parameter combination (e.g. attention_heads not a factor of embed_dim) and return a high objective value
        print(f"Invalid parameter combination: {e}")
        return float('inf')
    
    print(f"{counter['iteration']}/{counter['total']} loss: {round(loss,2)}, Time taken: {(time.time()-start_time):.2f} seconds.")
    return loss


def tune_hyperparameters(
    sample_space_config: dict,
    params: dict,
    dataset_train: dict,
    dataset_test: dict,
    iterations: int,
    device: str,
    plot: bool = True,
) -> dict:
    """
    Performs bayesian optimisation for hyperparameter tuning to find the optimal hyperparameter combintation for the BERT model.
    
    Parameters
    ----------
    sample_space_config : dict
        The dictionary defining the sample space from the the config file.
        Each key is the hyperparamater name, and the value defines the sample space for the parameter.
        - 'type' : str
            Specifies the type of sample space. May be either 'Real', 'Integer', or 'Categorical'.
            If 'Real' or 'Integer', expects 'low' and 'high' values. If 'Categorical', expects a 'categories' list.
    params : dict
        A dictionary containing the default hyperparameters. Updated with 'params_list' during execution.
    dataset_train : dict
        A dictionary containing the inputs and labels of the training data.
        - 'input_ids' : torch.Tensor (shape [num_batches, batch_size, tensor_length])
            The batched tensor of tokenised input strings.
        - 'labels' : torch.Tensor (shape [num_batches, batch_size, tensor_length])
            The batched tensor of labels corresponding to input IDs.
    dataset_train : dict
        A dictionary containing the inputs and labels of the test data.
        Refer to 'dataset_train'.
    iterations : int
        Total number of optimisation iterations (calls to the objective function).
    device : str
        The device that the model should be loaded on. 'cpu' or 'cuda'.
    
    plot : bool, optional
        Whether to display a plot of the objective convergence once tuning is finished. Defaults to True.
    
    Returns
    -------
    dict
        The updated 'params' dictionary containing the optimal hyperparameters found through optimisation.
    
    Raises
    ------
    ValueError
        If invalid hyperparameter combination during tuning (e.g. attention_heads not a factor of embed_dim).
    """
    counter = {'iteration': 0, 'total': iterations} # store iterations counter (must be mutable dictionary as cannot be returned from objective function)
    
    param_names, dimensions = build_sample_space(sample_space_config) # convert dictionary into skopt space objects
    
    objective_data = partial(
        objective,
        param_names=param_names,
        params=params,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        device=device,
        counter=counter
    ) # add extra arguments to the objective function
    
    start_time = time.time() # time the tuning process
    print("Beginning hyperparameter tuning.")
    print_line()
    
    result = gp_minimize(
        func=objective_data,
        dimensions=dimensions,
        n_calls=iterations,
        n_initial_points=max(10, math.ceil(iterations/10)), # number of random initial explorations proportional to iterations
        random_state=params['seed']
    ) # perform the bayesian optimisation to find optimal hyperparameter values
    
    print(f"Finishing hyperparameter tuning. Total time taken: {(time.time()-start_time):.2f} seconds.")
    print_line()
    
    params_dict = dict(zip(param_names, result.x)) # combine the paramater values back into a dictionary with paramater names
    params_dict_formatted = {k:f"{v:.2e}" if isinstance(v, float) else v for k,v in params_dict.items()} # round all floats in the dictionary
    
    print(f"Optimal hyperparameters: {params_dict_formatted}\nLoss {round(result.fun,2)}")
    print_line()

    if plot: # plot the objective convergence against iterations
        plot_convergence(result)
        plt.show()

    params.update(params_dict) # update params with found optimal values
    return params