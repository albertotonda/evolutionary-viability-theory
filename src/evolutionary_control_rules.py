"""
Script that implements the evolutionary loop to create control rules using Genetic Programming. I think we need to steal the individuals from gplearn, but I would like to keep the rest from inspyred, because it's better.
"""
import copy
import datetime
import inspyred
import numpy as np
import os
import pandas as pd
import random
import sympy
import sys

# steal parts from gplearn: _Program is the basically an individual class, the others are functions
from gplearn._program import _Program
from gplearn.functions import _Function, _function_map, add2, sub2, mul2, div2, sqrt1, log1, sin1, cos1 

# local modules
from logging_utils import initialize_logging, close_logging
from multithread_utils import ThreadPool, Worker
from viability_theory import ViabilityTheoryProblem
from threading import Thread, Lock

# this is a map used to convert from gplearn representation to string
function2string = {
        add2 : "+",
        sub2 : "-",
        mul2 : "*",
        div2 : "/",
        sqrt1 : "sqrt",
        log1 : "log",
        sin1 : "sin",
        cos1 : "cos",
        }

def equation_string_representation(individual) :

    # recursive function that is used to visit the tree (flattened as a list)
    def recursive_visit(individual, current_position) :
        
        current_node = individual.program[current_position]
        individual_string = ""

        # check the node: if it's a function, check its arity to find the next position
        if isinstance(current_node, _Function) :

            # if the function only has one argument
            if current_node.arity == 1 :
                individual_string += function2string[current_node] + "("
                individual_string += recursive_visit(individual, current_position+1)
                individual_string += ")"

            elif current_node.arity == 2 :
                individual_string += "(" + recursive_visit(individual, current_position+1)
                individual_string += function2string[current_node]
                individual_string += recursive_visit(individual, current_position+2) + ")"

        # in this case, the current node is a terminal, and more specifically a variable
        elif isinstance(current_node, int) :

                if individual.feature_names is None:
                    individual_string += 'X%s' % current_node
                else:
                    individual_string += individual.feature_names[current_node]

        # in this last case, the current node is a terminal, and more specifically a floating point value
        else :
            individual_string += '%.4f' % current_node

        return individual_string

    # string that will contain the representation of the individual
    individual_string = recursive_visit(individual, 0)

    return individual_string

def individual2string(candidate) :
    """
    Utility function to convert an individual's genotype to a string. Individuals here are dictionaries of programs.
    """
    individual_string = ""

    for variable, program in candidate.items() :
        individual_string += str(variable) + " -> {"
        individual_string += equation_string_representation(program) + "},"

    return individual_string[:-1] # remove the last ','

def generator(random, args) :
    """
    Generates the initial population for the evolutionary algorithm, using gplearn.
    """

    logger = args["logger"]

    # we can take into account the idea of having individuals
    # as a list/dictionary of equations, one per control function, depending
    # on the viability problem; so, first let's check the structure of the controls
    individual = dict()

    logger.debug("Generating new individual...")
    for control_variable in args["vp_control_structure"] :
        individual[control_variable] = _Program(**args["gplearn_settings"])
        logger.debug("For control variable \"%s\", equation \"%s\"" % (control_variable, equation_string_representation(individual[control_variable])))

    return individual

def fitness_function(individual, args) :

    logger = args["logger"]

    fitness = 0.0
    initial_conditions = args["current_initial_conditions"]
    vp = args["viability_problem"]
    time_step = args["time_step"]
    max_time = args["max_time"]

    # evaluating one solution takes a considerable amount of time, so we will first make a check;
    # if all control rules defined in the individual reduce to a constant, we can discard it 
    control_rules = { variable : equation_string_representation(control_rule) for variable, control_rule in individual.items() }
    logger.debug("Now evaluating individual corresponding to \"%s\"..." % control_rules)

    is_rule_constant = []
    for variable, rule in control_rules.items() :
        expression = sympy.sympify(rule)
        is_rule_constant.append(expression.is_constant())

    if all(is_rule_constant) :
        logger.debug("All control rules of the individual are constant, discarding...")
        return fitness

    # otherwise, create a local copy of the viability problem, that will be modified only here
    vp = copy.deepcopy(vp)
    # modify it, so that the control rules are now the same as the individual
    vp.set_control(control_rules)
    state_variables = [ v for v in vp.equations ]

    # and now, we run a simulation for each initial condition
    for ic in initial_conditions :
        # there might be some crash here, so 
        try :
            output_values, constraint_violations = vp.run_simulation(ic, time_step, max_time) 

            # compute the fitness, based on how long the simulation ran before a constraint violation
            fitness += len(output_values[state_variables[1]]) / (max_time/time_step)

        except Exception :
            # if executing the control rules raises an exception, fitness becomes zero and we immediately
            # terminate the loop
            logger.debug("Individual \"%s\" created an exception, it will have fitness zero" % control_rules)
            fitness = 0.0
            break 

    logger.debug("Fitness for individual \"%s\" is %.4f" % (control_rules, fitness))

    return fitness

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

    logger = args["logger"]
    best_individual = max(population, key=lambda x : x.fitness)
    
    # get some information
    best_string = individual2string(best_individual.candidate)
    best_fitness = best_individual.fitness

    # print the equation(s) corresponding to the best individual
    logger.info("Generation #%d (%d evaluations) Best individual: \"%s\"; Best fitness: %.4f" % (num_generations, num_evaluations, best_string, best_fitness))
    
    # save the current generation
    file_generation = os.path.join(args["directory_output"], "generation-%d.csv" % num_generations)
    logger.debug("Saving current generation to file \"%s\"..." % file_generation)

    dictionary_generation = {
            "generation" : [],
            "individual" : [],
            "fitness" : []
            }

    dictionary_generation["generation"] = [num_generations] * len(population)

    for i, ci in enumerate(args["current_initial_conditions"]) :
        dictionary_generation["initial_conditions_%d" % i] = [str(ci)] * len(population)

    for individual in population :
        dictionary_generation["individual"].append(individual2string(individual.candidate))
        dictionary_generation["fitness"].append(individual.fitness)

    df = pd.DataFrame.from_dict(dictionary_generation)
    df.to_csv(file_generation, index=False)

    return

def multi_thread_evaluator(candidates, args) :
    """
    Wrapper function for multi-thread evaluation of the fitness.
    """

    # get logger and other useful information from the args dictionary
    logger = args.get("logger", None)
    logger.info("Now starting the evaluation of %d individuals in the population..." % len(candidates))

    n_threads = args["n_threads"]
    n_initial_conditions = args["n_initial_conditions"]

    # NOTE  we have to draw some random initial conditions that will be shared by all individuals in this generation
    #       we save them inside the "args" dictionary, to make them accessible to other functions
    initial_conditions = [ args["viability_problem"].get_random_viable_point(args["random"]) for i in range(0, n_initial_conditions) ]
    logger.debug("Initial conditions for this generation: %s" % str(initial_conditions))
    args["current_initial_conditions"] = initial_conditions

    # create list of fitness values, for each individual to be evaluated
    # initially set to 0.0 (setting it to None is also possible)
    fitness_list = [0.0] * len(candidates)

    time_start = datetime.datetime.now()

    # create Lock object and initialize thread pool
    thread_lock = Lock()
    thread_pool = ThreadPool(n_threads) 

    # create list of arguments for threads
    arguments = [ (candidates[i], args, i, fitness_list, thread_lock) for i in range(0, len(candidates)) ]
    # queue function and arguments for the thread pool
    thread_pool.map(evaluate_individual, arguments)

    # wait the completion of all threads
    logger.debug("Starting multi-threaded evaluation...")

    thread_pool.wait_completion()

    time_end = datetime.datetime.now()
    time_difference = time_end - time_start
    logger.debug("The evaluation lasted %.2f minutes" % (time_difference.total_seconds() / float(60.0)))

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

    # create directory with name in the date
    directory_output = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-viability-theory" 
    if not os.path.exists(directory_output) :
        os.makedirs(directory_output)

    # start logging
    logger = initialize_logging(path=directory_output, log_name="log", date=True)
    logger.info("Starting, all results will be saved in folder \"%s\"..." % directory_output)
    logger.info("Setting up evolutionary algorithm...")

    # hard-coded values, probably to be replaced with function arguments
    n_threads = 1
    n_initial_conditions = 10
    time_step = 0.1
    max_time = 100

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

    logger.info("Starting evolutionary optimization...")
    final_population = ea.evolve(
                            generator=generator,
                            evaluator=multi_thread_evaluator,
                            pop_size=100,
                            num_selected=150,
                            maximize=True,
                            max_evaluations=10000,

                            # all items below this line go into the 'args' dictionary passed to each function
                            n_threads = n_threads,
                            time_step = time_step,
                            max_time = max_time,
                            random_seed = random_seed,
                            logger = logger,
                            vp_control_structure = vp_control_structure,
                            viability_problem = viability_problem,
                            gplearn_settings = gplearn_settings,
                            n_initial_conditions = n_initial_conditions,
                            random = prng,
                            directory_output = directory_output,
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

