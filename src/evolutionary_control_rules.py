"""
Script that implements the evolutionary loop to create control rules using Genetic Programming. I think we need to steal the individuals from gplearn, but I would like to keep the rest from inspyred, because it's better.
"""
import inspyred
import numpy as np
import random
import sys

# steal parts from gplearn: _Program is the basically an individual class
from gplearn._program import _Program
from gplearn.functions import _Function, _function_map

# local modules
from logging_utils import initialize_logging, close_logging
from multithread_utils import ThreadPool, Worker
from viability_theory import ViabilityTheoryProblem
from threading import Thread, Lock

def equation_string_representation(individual) :

    terminals = [0]
    output = ''
    for i, node in enumerate(individual.program):
        if isinstance(node, _Function):
            terminals.append(node.arity)
            output += node.name + '('
        else:
            if isinstance(node, int):
                if individual.feature_names is None:
                    output += 'X%s' % node
                else:
                    output += program.feature_names[node]
            else:
                output += '%.3f' % node
            terminals[-1] -= 1
            while terminals[-1] == 0:
                terminals.pop()
                terminals[-1] -= 1
                output += ')'
            if i != len(individual.program) - 1:
                output += ', '
        return output

def generator(random, args) :

    logger = args["logger"]

    # we can take into account the idea of having individuals
    # as a list/dictionary of equations, one per control function, depending
    # on the viability problem; so, first let's check the structure of the controls
    individual = dict()

    logger.debug("Generating new individual...")
    for control_variable in args["vp_control_structure"] :
        individual[control_variable] = _Program(**args["gplearn_settings"])
        logger.debug("For control variable \"%s\", equation \"%s\"" % (control_variable, individual[control_variable].program))

    return individual

def fitness_function(candidates, args) :

    fitness_list = np.zeros(len(candidates))
    
    # draw a given number of random tuples as starting conditions

    # for each given candidate control rule (individual)

        # for each random starting condition
            # solve the viability problem for the given control, ideally on a different thread
            # get the result, and compute the fitness function given its result

    return fitness_list

@inspyred.ec.variators.crossover
def variator(random, individual1, individual2, args) :
    """
    The variator for this particular problem has to be decorated as crossover, as it can perform multiple operations on one or two individuals.
    """

    return


def observer(population, num_generations, num_evaluations, args) :
    """
    The observer is a classic function for inspyred, that prints out information and/or saves individuals. However, it can be easily re-used by other
    evolutionary approaches.
    """

    # print the equation(s) corresponding to the best individual

    return

def multi_thread_evaluator(candidates, args) :
    """
    Wrapper function for multi-thread evaluation of the fitness.
    """

    # get logger from the args
    logger = args.get("logger", None)
    n_threads = args["n_threads"]

    # create list of fitness values, for each individual to be evaluated
    # initially set to 0.0 (setting it to None is also possible)
    fitness_list = [0.0] * len(candidates)

    # create Lock object and initialize thread pool
    thread_lock = Lock()
    thread_pool = ThreadPool(n_threads) 

    # create list of arguments for threads
    arguments = [ (candidates[i], args, i, fitness_list, thread_lock) for i in range(0, len(candidates)) ]
    # queue function and arguments for the thread pool
    thread_pool.map(evaluate_individual, arguments)

    # wait the completion of all threads
    if logger : logger.debug("Starting multi-threaded evaluation...")
    thread_pool.wait_completion()

    return fitness_list

def evaluate_individual(individual, args, index, fitness_list, thread_lock, thread_id) :
    """
    Wrapper function for individual evaluation, to be run inside a thread.
    """

    logger = args["logger"]

    logger.debug("[Thread %d] Starting evaluation..." % thread_id)

    # thread_lock is a threading.Lock object used for synchronization and avoiding
    # writing on the same resource from multiple threads at the same time
    thread_lock.acquire()
    fitness_list[index] = fitness_function(individual, args) # TODO put your evaluation function here, also maybe add logger and thread_id 
    thread_lock.release()

    logger.debug("[Thread %d] Evaluation finished." % thread_id)

    return

def evolve_rules(viability_problem, random_seed) :

    # start logging
    logger = initialize_logging(path=".", log_name="ea_vt.log")

    # hard-coded values, probably to be replaced with function arguments
    n_threads = 1

    # initialize the pseudo-random number generators
    prng = random.Random(random_seed)
    nprs = np.random.RandomState(random_seed) 

    # first, we interrogate the viability problem to check how many variables and control rules we're talking about
    # viability_problem.control is a dictionary 
    vp_control_structure = viability_problem.control
    vp_variables = [ variable for variable in viability_problem.equations ]

    # then, we setup all necessary values for GPlearn individuals, _Program instances
    gplearn_settings = {
            "function_set" : [ _function_map[f] for f in ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'sin', 'cos'] ],
            "arities" : (2, 2, 2, 2, 1, 1, 1, 1),
            "init_depth" : (2, 4),
            "init_method" : "half and half",
            "n_features" : len(vp_variables),
            "feature_names" : vp_variables,
            "const_range" : (-1.0, 1.0),
            "metric" : None,
            "p_point_replace" : 0.5,
            "random_state" : nprs,
            "parsimony_coefficient" : 0.01,
            "program" : None,
            }

    # we use inspyred's base stuff to manage the evolution
    ea = inspyred.ec.EvolutionaryComputation(prng)
    # these functions are pre-programmed in inspyred
    ea.selector = inspyred.ec.selectors.tournament_selection 
    ea.replacer = inspyred.ec.replacers.plus_replacement
    ea.terminator = inspyred.ec.terminators.evaluation_termination
    # these functions are custom and inside this script
    ea.variator = variator
    ea.observer = observer

    final_population = ea.evolve(
                            generator=generator,
                            evaluator=multi_thread_evaluator,
                            pop_size=100,
                            num_selected=150,
                            maximize=False,
                            max_evaluations=10000,

                            # all items below this line go into the 'args' dictionary passed to each function
                            n_threads = n_threads,
                            random_seed = random_seed,
                            logger = logger,
                            vp_control_structure = vp_control_structure,
                            gplearn_settings = gplearn_settings,
                            )

    return

if __name__ == "__main__" :
    
    equations = {
            "L" : "u",
            "P" : "-b * P + L + r * P**q/(m**q + P**q)"
            }
    control = {"u" : ""}
    constraints = {
            "u" : ["u >= umin", "u <= umax"],
            "L" : ["L >= Lmin", "L <= Lmax"],
            "P" : ["P >= 0", "P <= Pmax"],
            }
    parameters = {
            "b" : 0.8,
            "r" : 1.0,
            "q" : 8.0,
            "m" : 1.0,
            "umin" : -0.09,
            "umax" : 0.09,
            "Lmin" : 0.01,
            "Lmax" : 1.0,
            "Pmax" : 1.4,
            }

    vp = ViabilityTheoryProblem(equations=equations, control=control, constraints=constraints, parameters=parameters)
    print("Evolving control rules for the following viability problem:", vp)

    evolve_rules(viability_problem=vp, random_seed=42)

    sys.exit(0)

