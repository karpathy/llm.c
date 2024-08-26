import numpy as np
from smac.model.gaussian_process.kernels import MaternKernel, ConstantKernel, RBFKernel
# from smac.model.gaussian_process.kernels import RBFKernel
from smac.model.gaussian_process.gaussian_process import GaussianProcess
from smac.acquisition.function.expected_improvement import EI
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace import Configuration, ConfigurationSpace, Float
from matplotlib import pyplot as plt
from smac.intensifier.hyperband import Hyperband
from smac.model.random_forest import RandomForest
import argparse
from functools import partial
from train_gpt2_hpo import train_eval_hpo
from smac import MultiFidelityFacade, RunHistory, Scenario
# from smac.intensifier.hyperband_utils import get_n_trials_for_hyperband_multifidelity
from smac.multi_objective.parego import ParEGO
import logging
import datetime
from smac.facade.abstract_facade import AbstractFacade
import submitit

def set_queue(q_, log_folder, maximum_runtime=None):
    global ex
    global q
    if q_ == 'all':
        q = 'alldlc_gpu-rtx2080'
    if q_ == 'ml':
        q = 'mldlc_gpu-rtx2080'
    if q_ == 'mlhiwi':
        q = "mlhiwidlc_gpu-rtx2080"

    if maximum_runtime is None:
        if q == 'alldlc_gpu-rtx2080' or q == 'mlhiwidlc_gpu-rtx2080':
            maximum_runtime = 24*60*1-1
        else:
            maximum_runtime = 24*60*4-1

    ex = submitit.AutoExecutor(folder=log_folder)
    ex.update_parameters(timeout_min=maximum_runtime,
                        slurm_partition=q, #  mldlc_gpu-rtx2080
                        slurm_signal_delay_s=180, # time to pass the USR2 signal to slurm before the job times out so that it can finish the run
                        tasks_per_node=1,
                        nodes=1,
                        cpus_per_task=30, #24
                        mem_per_cpu=4096,
                        job_name='smac_hpo',
                        slurm_gres=f'gpu:{1}'
       )

    return maximum_runtime

def plot_pareto(smac: AbstractFacade, incumbents: list[Configuration]) -> None:
    """Plots configurations from SMAC and highlights the best configurations in a Pareto front."""
    average_costs = []
    average_pareto_costs = []
    for config in smac.runhistory.get_configs():
        # Since we use multiple seeds, we have to average them to get only one cost value pair for each configuration
        average_cost = smac.runhistory.average_cost(config)

        if config in incumbents:
            average_pareto_costs += [average_cost]
        else:
            average_costs += [average_cost]

    # Let's work with a numpy array
    costs = np.vstack(average_costs)
    pareto_costs = np.vstack(average_pareto_costs)
    pareto_costs = pareto_costs[pareto_costs[:, 0].argsort()]  # Sort them

    costs_x, costs_y = costs[:, 0], costs[:, 1]
    pareto_costs_x, pareto_costs_y = pareto_costs[:, 0], pareto_costs[:, 1]

    plt.scatter(costs_x, costs_y, marker="x", label="Configuration")
    plt.scatter(pareto_costs_x, pareto_costs_y, marker="x", c="r", label="Incumbent")
    plt.step(
        [pareto_costs_x[0]] + pareto_costs_x.tolist() + [np.max(costs_x)],  # We add bounds
        [np.max(costs_y)] + pareto_costs_y.tolist() + [np.min(pareto_costs_y)],  # We add bounds
        where="post",
        linestyle=":",
    )

    plt.title("Pareto-Front")
    plt.xlabel(smac.scenario.objectives[0])
    plt.ylabel(smac.scenario.objectives[1])
    plt.legend()
    
    # Save the plot to a file
    plt.savefig("pareto_front.png", format='png', dpi=300)  # Save as PNG with high resolution
    plt.show()
    

def setup_logger():

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    current_time = datetime.datetime.now()
    formatted_date = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # File handler
    file_handler = logging.FileHandler(f'logs/smac_{formatted_date}.log')

    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s',  datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)
    logger.propagate = False
    logging.captureWarnings(True)

def print_with_tabs(obj, num_tabs=1):
    # Convert the object to a string representation
    obj_str = str(obj)

    # Split the string representation into lines
    lines = obj_str.split('\n')

    # first line should not be tabbed
    tabbed_lines = ["\t"*num_tabs +lines[0] + "\n"]

    # Add a tab at the beginning of each line
    tabbed_lines = tabbed_lines + ['\t'*(num_tabs+1) + line + "\n" for line in lines[1:]]

    # Join the tabbed lines back into a single string and print
    return "".join(tabbed_lines).rstrip("\n")


def main_smac(args):
    
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info(f'============Starting============\n')

    logger.info("\t== Creating the Partial/Target/Evaluation function ==")
    partial_function = partial(train_eval_hpo, 
                               input_bin=args.input_bin, 
                               input_val_bin=args.input_val_bin, 
                               model=args.model, 
                               batch_size=args.batch_size, 
                               total_batch_size=args.total_batch_size, 
                               warmup_iters=args.warmup_iters, 
                               learning_rate_decay_frac=args.learning_rate_decay_frac, 
                               grad_clip=args.grad_clip, 
                               val_max_steps=args.val_max_steps, 
                               dtype=args.dtype, 
                               zero_stage=args.zero_stage,
                               logger=logger, 
                               multi_objective=args.multiobjective,
                               )

    logger.info(f"\t{print_with_tabs(partial_function,1)}")
    
    
    # Define the configuration space
    cs = ConfigurationSpace()
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-6, 1e-3, default_value=1e-4, log=True)
    weight_decay = UniformFloatHyperparameter("weight_decay", 1e-6, 0.1, default_value=0.01, log=True)
    sequence_length = UniformIntegerHyperparameter("sequence_length", 256, 1024, default_value=1024)
    cs.add_hyperparameters([learning_rate, weight_decay, sequence_length])

    logger.info(f"\t== SMAC ConfigSpace ==\n\t{print_with_tabs(cs,1)}")


    if args.multiobjective:
        scenario = Scenario(
                    cs,
                    # name="SMAC",
                    objectives=["val_loss", "train_time"],
                    walltime_limit=60*60*23,  
                    n_trials=args.n_trials, #500,  # Evaluate max 500 different trials
                    min_budget=10**3,  # Train the MLP using a hyperparameter configuration for at least 5 epochs
                    max_budget=2*10**4,  # Train the MLP using a hyperparameter configuration for at most 25 epochs
                    n_workers=1,
                    deterministic=True
        )
    else:
        scenario = Scenario(
            cs,
            # objectives=["val_loss", "train_time"],
            walltime_limit=60*60*23,  
            n_trials=args.n_trials, #500,  # Evaluate max 500 different trials
            min_budget=10**3,  # Train the MLP using a hyperparameter configuration for at least 5 epochs
            max_budget=2*10**4,  # Train the MLP using a hyperparameter configuration for at most 25 epochs
            n_workers=1,
            deterministic=True
        )

    logger.info(f"\t{print_with_tabs(scenario,1)}")

    # # Set up the Gaussian Process model with Matern Kernel
    # kernel = MaternKernel(nu=2.5)  # Matern kernel with nu=2.5 (commonly used)
    if args.surrogate == "gp":
        kernel = MaternKernel(nu=2.5) * ConstantKernel(1.0, constant_value_bounds="fixed")  # Radial Basis Function (RBF) kernel
        model = GaussianProcess(configspace=cs, kernel=kernel)
    else:
        model = RandomForest(configspace=cs)
        
    multi_objective_algorithm = ParEGO(scenario)

    # rf_model = RandomForest(configspace=cs)
    # Use Expected Improvement (EI) as the acquisition function
    # acquisition_function = EI(model=gp_model)
    initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=args.n_initial)
    intensifier = Hyperband(scenario, eta=args.eta, incumbent_selection="highest_budget")
    
    logger.info("\t== Creating SMAC MultiFidelityFacade ==")

    if args.multiobjective:
        smac = MultiFidelityFacade(
            scenario=scenario,
            target_function=partial_function,
            initial_design=initial_design,
            intensifier=intensifier,
            multi_objective_algorithm=multi_objective_algorithm,
            overwrite=True,            
            model=model,
            # acquisition_function=acquisition_function,
        )
    else:
        smac = MultiFidelityFacade(
            scenario=scenario,
            target_function=partial_function,
            initial_design=initial_design,
            intensifier=intensifier,
            overwrite=True,            
            model=model,
            # acquisition_function=acquisition_function,
        )

    logger.info(f"\t{print_with_tabs(smac, 1)}")
    logger.info(f"\t== SMAC Configuration ==\n\t{print_with_tabs(smac._scenario,1)}")
    print(f"== SMAC Configuration ==\n\t{print_with_tabs(smac._scenario,1)}")
    logger.info(f"\t== SMAC Configuration ==\n\t{print_with_tabs(smac._intensifier,1)}")
    print(f"== SMAC Configuration ==\n\t{print_with_tabs(smac._intensifier,1)}")
    logger.info(f"\t== SMAC Configuration ==\n\t{print_with_tabs(smac._multi_objective_algorithm,1)}")
    print(f"== SMAC Configuration ==\n\t{print_with_tabs(smac._multi_objective_algorithm,1)}")
    logger.info(f"\t== SMAC Configuration ==\n\t{print_with_tabs(smac._model,1)}")
    print(f"== SMAC Configuration ==\n\t{print_with_tabs(smac._model,1)}")

    logger.info(f"\t Multi-objective optimization: {args.multiobjective}")
    print(f"Multi-objective optimization: {args.multiobjective}")
    logger.info(f"\t Number of initial configurations: {args.n_initial}")
    print(f"Number of initial configurations: {args.n_initial}")
    logger.info(f"\t Number of trials: {args.n_trials}")
    print(f"Number of trials: {args.n_trials}")
    logger.info(f"\t eta: {args.eta}")
    print(f"eta: {args.eta}")
    logger.info("\t== Starting the optimization ==")
    
    # Start the optimization
    incumbents = smac.optimize()

    # Print the best configuration
    print(f"Best found configuration: {incumbents}")
    
    plot_pareto(smac, incumbents)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("-s", "--slurm", type=bool, default=False, help="flag to run training on slurm") # if not provided you can just run it from terminal (for debugging)
    parser.add_argument('-i', "--input_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_train.bin", help="input .bin to train on")
    parser.add_argument('-j', "--input_val_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to eval validation loss on")
    parser.add_argument('-o', "--output_dir", type=str, default="", help="output directory to which to write logs and checkpoints")
    parser.add_argument('-e', "--model", type=str, default="d6", help="gpt2-tiny|gpt2|gpt2-medium|gpt2-large|gpt2-xl|d6|d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.add_argument('-b', "--batch_size", type=int, default=4, help="batch size, in units of #batch dimensions")
    # parser.add_argument('-t', "--sequence_length", type=int, default=1024, help="sequence length")
    parser.add_argument('-d', "--total_batch_size", type=int, default=-1, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    # parser.add_argument('-x', "--num_iterations", type=int, default=-1, help="number of iterations to run")
    # optimization
    # parser.add_argument('-l', "--learning_rate", type=float, default=1e-4, help="learning rate warmup iterations")
    parser.add_argument('-u', "--warmup_iters", type=int, default=0, help="learning rate warmup iterations")
    parser.add_argument('-q', "--learning_rate_decay_frac", type=float, default=1.0, help="learning rate warmup iterations")
    # parser.add_argument('-c', "--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument('-m', "--val_max_steps", type=int, default=20, help="how many batches of val to average?")
    parser.add_argument("--dtype", type=str, default="float32", help="float32|float16|bfloat16")
    parser.add_argument('-z', "--zero_stage", type=int, default=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.add_argument("--multiobjective", type=bool, default=False, help="multiobjective optimization")
    parser.add_argument("--n_initial", type=int, default=5, help="number of initial configurations to evaluate")
    parser.add_argument("--n_trials", type=int, default=500, help="number of trials to evaluate")
    parser.add_argument("--eta", type=int, default=2, help="eta parameter for Hyperband")
    parser.add_argument("--surrogate", type=str, default="gp", help="surrogate model to use")
    args = parser.parse_args()
    
    if args.slurm == True:
        print("Running on slurm")
        global ex
        global q
        maximum_runtime = 0
        log_folder = './logs_cluster/'
        maximum_runtime = set_queue('mlhiwi', log_folder)
        submit_func = ex.submit
        job = submit_func(main_smac, args)

        print(job)
    else:
        print("Running on local machine")
        print(args.slurm)
        main_smac(args)
