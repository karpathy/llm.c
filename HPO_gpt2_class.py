import numpy as np
from smac.model.gaussian_process.kernels import MaternKernel, ConstantKernel, RBFKernel
# from smac.model.gaussian_process.kernels import RBFKernel
from smac.model.gaussian_process.gaussian_process import GaussianProcess
from smac.acquisition.function.expected_improvement import EI
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace import Configuration, ConfigurationSpace, Float
from matplotlib import pyplot as plt
from smac.intensifier.hyperband import Hyperband, SuccessiveHalving
from smac.model.random_forest import RandomForest
import argparse
from functools import partial
from train_gpt2_hpo import train_eval_hpo, setup_logger, print_with_tabs
from smac import MultiFidelityFacade, RunHistory, Scenario
# from smac.intensifier.hyperband_utils import get_n_trials_for_hyperband_multifidelity
from smac.multi_objective.parego import ParEGO
import logging
from smac.facade.abstract_facade import AbstractFacade
import submitit
import pickle
import torch
from smac.runhistory.dataclasses import TrialValue

global log_file_name

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
    global log_file_name
    # Save the plot to a file
    plt.savefig(f"pareto_front_{log_file_name}.png", format='png', dpi=300)  # Save as PNG with high resolution
    plt.show()
    
class AskTellSMACOptimizer:
    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = self.setup_logger('HPO_gpt2')
        self.logger.info(f'============Starting============\n')
        
        self.args = args
        self.hp_details = []
        
        self.partial_function = train_eval_hpo
        
        self.logger.info("\t== Creating the Partial/Target/Evaluation function ==")
        self.logger.info(f"\t{self.partial_function}")
        
        # Define the configuration space
        self.cs = ConfigurationSpace()
        learning_rate = UniformFloatHyperparameter("learning_rate", 1e-6, 1e-3, default_value=1e-4, log=True)
        weight_decay = UniformFloatHyperparameter("weight_decay", 1e-6, 0.1, default_value=0.01, log=True)
        sequence_length = UniformIntegerHyperparameter("sequence_length", 256, 1024, default_value=1024)
        self.cs.add_hyperparameters([learning_rate, weight_decay, sequence_length])
        
        self.logger.info(f"\t== SMAC ConfigSpace ==\n\t{self.cs}")
        
        # Create the scenario
        self.scenario = self.create_scenario(args.multiobjective)
        
        # Choose the model
        self.model = self.select_model(args.surrogate)

        # Select multi-objective algorithm if needed
        self.multi_objective_algorithm = ParEGO(self.scenario) if args.multiobjective else None
        
        # Initialize intensifier
        self.initial_design = MultiFidelityFacade.get_initial_design(self.scenario, n_configs=args.n_initial)
        self.intensifier = SuccessiveHalving(self.scenario, eta=args.eta, incumbent_selection="highest_budget" if args.multiobjective else None)

        # Initialize SMAC optimizer
        self.smac = self.create_smac()
        
        self.logger.info(f"\t Multi-objective optimization: {args.multiobjective}")
        self.logger.info(f"\t Number of initial configurations: {args.n_initial}")
        self.logger.info(f"\t Number of trials: {args.n_trials}")
        self.logger.info(f"\t eta: {args.eta}")
    
    def setup_logger(self, name):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(name)
        return logger
    
    def create_scenario(self, multiobjective):
        if multiobjective:
            return Scenario(
                self.cs,
                name="SMAC_trial",
                objectives=["val_loss", "train_time"],
                walltime_limit=23*60*60,
                n_trials=self.args.n_trials,
                min_budget=20*60,
                max_budget=23*60*60,
                n_workers=1,
                seed=0,
                deterministic=True
            )
        else:
            return Scenario(
                self.cs,
                name="SMAC_trial",
                walltime_limit=23*60*60,
                n_trials=self.args.n_trials,
                min_budget=20*60,
                max_budget=23*60*60,
                n_workers=1,
                seed=0,
                deterministic=True
            )

    def select_model(self, surrogate_type):
        if surrogate_type == "gp":
            kernel = MaternKernel(nu=2.5) * ConstantKernel(1.0, constant_value_bounds="fixed")
            return GaussianProcess(configspace=self.cs, kernel=kernel)
        else:
            return RandomForest(configspace=self.cs)

    def create_smac(self):
        if self.args.multiobjective:
            return MultiFidelityFacade(
                scenario=self.scenario,
                target_function=self.partial_function,
                initial_design=self.initial_design,
                intensifier=self.intensifier,
                multi_objective_algorithm=self.multi_objective_algorithm,
                overwrite=False,            
                model=self.model,
            )
        else:
            return MultiFidelityFacade(
                scenario=self.scenario,
                target_function=self.partial_function,
                initial_design=self.initial_design,
                intensifier=self.intensifier,
                overwrite=False,            
                model=self.model,
            )
    
    def ask(self):
        """
        Ask for a new candidate configuration from the optimizer.
        """
        return self.smac.ask()

    def tell(self, config, result):
        """
        Tell the optimizer the result of evaluating the candidate.
        Args:
        - config: The configuration evaluated.
        - result: The outcome or evaluation metric for the configuration.
        """
        self.smac.tell(config, result)
    
    def optimize(self, n_trials=None):
        """
        Run the optimization loop for a given number of trials.
        """
        n_trials = n_trials if n_trials is not None else self.args.n_trials
        
        for i in range(n_trials):
            info = self.ask()
            assert info.seed is not None
            cost = self.partial_function(info.config, budget=info.budget, seed=info.seed)
            value = TrialValue(cost=cost, time=0.5)  # Assuming a sample result, this can be customized.
            self.tell(info, value)
        
        incumbents = self.smac.intensifier.get_incumbents()
        return incumbents
    
    def save_results(self, incumbents):
        """
        Save the optimization results to disk.
        """
        global log_file_name
        pickle.dump(incumbents, open(f"{log_file_name}_incumbents.pkl", "wb"))
        pickle.dump(self.smac, open(f"{log_file_name}_smac.pkl", "wb"))




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("-s", "--slurm", type=bool, default=False, help="flag to run training on slurm") # if not provided you can just run it from terminal (for debugging)
    parser.add_argument('-i', "--input_bin", type=str, default="dev/data/fineweb10B/fineweb_train_*.bin", help="input .bin to train on")
    parser.add_argument('-j', "--input_val_bin", type=str, default="dev/data/fineweb10B/fineweb_val_*.bin", help="input .bin to eval validation loss on")
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
    parser.add_argument('-u', "--warmup_iters", type=int, default=700, help="learning rate warmup iterations")
    parser.add_argument('-q', "--learning_rate_decay_frac", type=float, default=0.0, help="learning rate warmup iterations")
    # parser.add_argument('-c', "--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument('-m', "--val_max_steps", type=int, default=20, help="how many batches of val to average?")
    parser.add_argument("--dtype", type=str, default="float32", help="float32|float16|bfloat16")
    parser.add_argument('-z', "--zero_stage", type=int, default=1, help="zero redundancy optimizer stage (0/1/2/3)")
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
        ask_tell_interface = AskTellSMACOptimizer(args)
        main_smac = ask_tell_interface.optimize
        job = submit_func(main_smac)

        print(job)
    else:
        print("Running on local machine")
        print(args.slurm)
        ask_tell_interface = AskTellSMACOptimizer(args)
        ask_tell_interface.optimize()
