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

function2string = {
        "add" : "+",
        "sub" : "-",
        "mul" : "*",
        "div" : "/",
        "sqrt" : "sqrt",
        "log" : "log",
        "sin" : "sin",
        "cos" : "cos",
        }

def equation_string_representation(individual) :
    """
    This is the correct version of the function that transforms a gplearn _Program instance to a sympy-processable string. It is important to notice
    that gplearn internally stores a GP-tree as a flattened list representation, using a stack-based encoding that employs the -arity of the functions
    to exactly understand where the function arguments are placed in the list. This is why the use of a stack is necessary in decoding the list.
    """

    # the general idea is that we go over the nodes (flattened as elements of a list) one by one, and we create a 'stack' (another list)
    # that contains, for each element: a list with first element the node of a function, and as following elements all the COMPUTED ARGUMENTS
    # of the function. If not all the argument of the function are computed yet (because they are other sub-trees), that part is not added to
    # the string representation of the equation until later, when all arguments are processed. This creates a stack where progressively all function
    # nodes are added, one after the other, along with their arguments, and AS SOON AS A FUNCTION HAS ALL ITS COMPUTED ARGUMENTS, that part of the
    # stack is processed, and the intermediate result is appended to the arguments of the previous element of the stack (e.g. as a computed argument
    # of the previous function. Eventually, the stack becomes empty.
    #
    # Example: sin(log(div(P, L))) ; nodes in the program: [sin, log, div, P, L]
    # we read 'sin', who has a 'arity' of 1: stack [[sin]]
    # we read 'log', which is not a terminal, and has a 'arity' of 1: stack [[sin], [log]]
    # we read 'div', which has a 'arity' of 2: stack [[sin], [log], [div]]
    # we read 'P', which is a terminal, so it gets added to the list of the previous element in the stack: stack [[sin], [log], [div, P]]
    # the stack is not processed yet, because the last element does not have enough arguments to be processed (it requires 2, it only has 'P')
    # we read 'L', which is a terminal: stack [[sin], [log], [div, P, L]]
    # now the last element in the stack has enough arguments to be processed, so it gets converted to a string and appended as an argument of the previous element
    # stack: [[sin], [log, "(P/L)"]]
    # but now the 'new' last element has the 1 argument it needs, so it can be processed again and appended as an argument to the previous
    # stack: [[sin, "log(P/L"]]
    # and again, since 'sin' only needs 1 argument; the stack will be empty and the final result will be:
    # "sin(log(P/L))"

    # this is the case for one-node degenerate individuals
    node = individual.program[0]
    if isinstance(node, int) :
        return individual.feature_names[node]
    elif isinstance(node, float) :
        return "%.4f" % node

    # if the individual is more than one node, let's go over it!
    final_string = ""
    apply_stack = []

    # this is for regular individuals
    for node in individual.program :

        if isinstance(node, _Function) :
            apply_stack.append([node])
        else :
            apply_stack[-1].append(node)

        while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1 :

            function = function2string[apply_stack[-1][0].name]
            terminals = []
            for t in apply_stack[-1][1:] :
                if isinstance(t, int) :
                    terminals.append(individual.feature_names[t])
                elif isinstance(t, float):
                    terminals.append("(%.4f)" % t)
                elif isinstance(t, str) :
                    terminals.append(t)
                else :
                    print("This is an error!")

            # create partial string
            intermediate_result = ""
            if apply_stack[-1][0].arity == 2 :
                intermediate_result = "(" + terminals[0] + function + terminals[1] + ")"
            else :
                intermediate_result = function + "(" + terminals[0] + ")"

            if len(apply_stack) != 1 :
                apply_stack.pop()
                apply_stack[-1].append(intermediate_result)
            else :
                return intermediate_result

    # we should never get here
    return None


def individual2string(candidate) :
    """
    Utility function to convert an individual's genotype to a string. Individuals here are dictionaries of programs.
    """
    individual_string = ""

    for variable, program in candidate.items() :
        individual_string += "\'" + str(variable) + "\' : \'"
        individual_string += equation_string_representation(program) + "\', "

    return individual_string[:-2] # remove the last ', '

def individual_to_gplearn_string(candidate) :

    """
    Utility function to convert an individual's genotype to a string, using gplearn's formalism. Individuals here are dictionaries of gplearn programs.
    """
    individual_string = ""

    for variable, program in candidate.items() :
        individual_string += "\'" + str(variable) + "\' : \'"
        individual_string += str(program) + "\', "

    return individual_string[:-2] # remove the last ', '

def are_individuals_equal(individual1, individual2) :
    """
    Utility function, to check whether individuals are identical.
    """

    for variable in individual1 :
        expr1 = sympy.sympify(equation_string_representation(individual1[variable]))
        expr2 = sympy.sympify(equation_string_representation(individual2[variable]))

        # this would be the best way of doing this comparison, comparing directly the symbolic
        # representations; however, for complex individuals, it can take HOURS of computation
        #if not expr2.equals(expr1) :
        #    return False

        # so, we settle for the second-best option: compare the string representations
        # sympy apparently always orders symbolic elements in the same way, even if they
        # are presented in a different order (e.g. "L + P" and "P + L", converted to symbolic
        # and then back to string, would both be "L + P")
        string1 = str(expr1)
        string2 = str(expr2)

        if string2 != string1 :
            return False

    return True

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
    """
    Fitness function for an individual.
    """

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
    logger = args["logger"]
    random_state = args["random_state"] # we are going to use a numpy pseudo-random number generator, because gplearn likes that
    p_crossover = args["p_crossover"]
    p_subtree = args["p_subtree"]
    p_hoist = args["p_hoist"]
    p_point = args["p_point"]

    offspring = []

    # we are going to loop until the new individual is at least a bit different from individual1
    is_offspring_equal_to_parent = True

    while is_offspring_equal_to_parent :
        # the genome of an individual is a dictionary, with one gplearn program for each control variable
        # this variator should take that into account, and consider that sometimes it's better to just change
        # one or few of the control rules, to preserve locality; at the moment, there is a probabilty that
        # nothing happens, but then we have to check that the new individual is actually different from individual1
        logger.debug("Generating new individual...")
        new_individual = { variable : None for variable in individual1 }

        program = None
        for variable in new_individual :
            # let's select a probability to apply one of the mutation operators
            p = random_state.uniform()
            logger.debug("Creating gplearn program for state variable \"%s\"..." % variable)

            new_program = None
            if p < p_crossover :
                logger.debug("Performing a crossover...")
                program, removed, remains = individual1[variable].crossover(individual2[variable].program, random_state)

            elif p < (p_crossover + p_subtree) :
                logger.debug("Performing a subtree mutation...")
                program, removed, _ = individual1[variable].subtree_mutation(random_state)

            elif p < (p_crossover + p_subtree + p_hoist) :
                logger.debug("Performing a hoist mutation...")
                program, removed = individual1[variable].hoist_mutation(random_state)

            elif p < (p_crossover + p_subtree + p_hoist + p_point) :
                logger.debug("Performing a point mutation...")
                program, mutated = individual1[variable].point_mutation(random_state)

            else :
                logger.debug("Copying the original genome from individual1...")
                program = individual1[variable].reproduce() 

            # create new instance of _Program
            new_program = _Program(function_set=individual1[variable].function_set,
                       arities=individual1[variable].arities,
                       init_depth=individual1[variable].init_depth,
                       init_method=individual1[variable].init_method,
                       n_features=individual1[variable].n_features,
                       metric=individual1[variable].metric,
                       transformer=individual1[variable].transformer,
                       const_range=individual1[variable].const_range,
                       p_point_replace=individual1[variable].p_point_replace,
                       parsimony_coefficient=individual1[variable].parsimony_coefficient,
                       feature_names=individual1[variable].feature_names,
                       random_state=random_state,
                       program=program)

            logger.debug("New program generated: %s" % str(new_program))
            new_individual[variable] = new_program

        # check if the new individual is identical to individual 1
        logger.debug("New candidate individual: \"%s\"" % individual2string(new_individual))
        logger.debug("Parent individual to be compared against: \"%s\"" % individual2string(individual1))
        
        is_offspring_equal_to_parent = are_individuals_equal(individual1, new_individual)

        if not is_offspring_equal_to_parent :
            offspring.append(new_individual)
        else :
            logger.debug("The two individuals are exactly identical! I should re-loop and create a new one")

        # end of while: if the new individual is equal to parent individual1, we loop and try again 

    return offspring


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
            "individual_gplearn" : [],
            "fitness" : []
            }

    dictionary_generation["generation"] = [num_generations] * len(population)

    for i, ci in enumerate(args["current_initial_conditions"]) :
        dictionary_generation["initial_conditions_%d" % i] = [str(ci)] * len(population)

    for individual in population :
        dictionary_generation["individual"].append(individual2string(individual.candidate))
        dictionary_generation["individual_gplearn"].append(individual_to_gplearn_string(individual.candidate))
        dictionary_generation["fitness"].append(individual.fitness)

    # conver the dictionary to a pandas DataFrame, and sort it by descending fitness (easier to read later)
    df = pd.DataFrame.from_dict(dictionary_generation)
    df.sort_values(by=["fitness"], ascending=False, inplace=True)
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
    n_threads = 8
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
    function_set = [ _function_map[f] for f in ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'sin', 'cos'] ]
    arities = {} # 'arities' is a dictionary that contains information on the number of arguments for each function
    for function in function_set :
        arity = function.arity
        arities[arity] = arities.get(arity, [])
        arities[arity].append(function)

    gplearn_settings = {
            "function_set" : function_set,
            "arities" : arities,
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
                            max_evaluations=1000,

                            # all items below this line go into the 'args' dictionary passed to each function
                            directory_output = directory_output,
                            logger = logger,
                            n_threads = n_threads,
                            random_seed = random_seed, # this is the numpy random number generator, used by gplearn
                            random = prng, # this is the random.Random instance used by inspyred
                            random_state = nprs,
                            vp_control_structure = vp_control_structure,
                            viability_problem = viability_problem,
                            # these parameters below are used for the evaluation of candidate solutions
                            time_step = time_step,
                            max_time = max_time,
                            n_initial_conditions = n_initial_conditions,
                            # settings for gplearn
                            gplearn_settings = gplearn_settings,
                            p_crossover = 0.4,
                            p_hoist = 0.1,
                            p_subtree = 0.1,
                            p_point = 0.1,
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

