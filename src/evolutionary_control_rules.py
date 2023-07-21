"""
Script that implements the evolutionary loop to create control rules using Genetic Programming. I think we need to steal the individuals from gplearn, but I would like to keep the rest from inspyred, because it's better.
"""
# TODOs
# - add a lexicographic fitness evaluation that includes size of the trees
import copy
import datetime
import inspyred
import multiprocessing
import numpy as np
import os
import pandas as pd
import platform # used to understand whether we are under Linux or Windows, for timeouts
import random
import signal # used to manage timeouts
import sympy
import sys
import time # just used in one point to sleep for a few seconds...

# all this stuff below is to prevent scipy.ode.odeint from writing to stderr in a thread-unsafe way
# now, this part seems weird, but it's actually done to prevent one of the functions that scipy.ode uses
# (that is written and compiled in C) from writing to stderr. The issue is that writing to stderr is not thread-safe, so if we
# use a multi-threaded approach, multiple threads writing to stderr at the same time will create a segmentation fault
import contextlib
import io
import sys

@contextlib.contextmanager
def no_stderr_stdout() :
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = io.BytesIO()
    sys.stderr = io.BytesIO()
    yield
    sys.stdout = save_stdout
    sys.stderr = save_stderr
# end of the stuff to prevent writing to stderr; if we find another way, all this stuff could be removed
# UPDATE: it did not work, NOTHING can prevent the @#%!ing function from writing to stderr...so, I just
# suppressed all printouts during individual evaluation, which is a pity because logging is cool

# steal parts from gplearn: _Program is the basically an individual class, the others are functions
from gplearn._program import _Program
from gplearn.functions import _Function, _function_map, make_function, add2, sub2, mul2, div2, sqrt1, log1, sin1, cos1 

# parts from multiprocessing, to try and have a personalized version of the multiprocessing evaluation
from multiprocessing import Manager, Process, Pool, TimeoutError

# local modules
from logging_utils import initialize_logging, close_logging
from multithread_utils import ThreadPool, Worker
from viability_theory import ViabilityTheoryProblem
from threading import Thread, Lock

# utility dictionary, to map from gplearn's internal representation to a string that sympy can interpret
function2string = {
        "add" : "+",
        "sub" : "-",
        "mul" : "*",
        "div" : "/",
        "sqrt" : "sqrt",
        "log" : "log",
        "sin" : "sin",
        "cos" : "cos",
        "min" : "min",
        "max" : "max"
        }

# these two are wrapper functions, needed by gplearn to add 'min' and 'max' to the function set
def _min(x1, x2) :
    return np.minimum(x1, x2) # we need to use this, because gplearn's functions are designed to work on arrays

def _max(x1, x2) :
    return np.maximum(x1, x2)

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
            if apply_stack[-1][0].arity == 2 and function != "min" and function != "max" :
                intermediate_result = "(" + terminals[0] + function + terminals[1] + ")"
            elif function == "max" or function == "min" :
                intermediate_result = function + "(" + terminals[0] + "," + terminals[1] + ")"
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
    
    # there is one extra key in the individual's dictionary called "id_", that is
    # just used to keep track of the individuals' lineage; we must ignore it while
    # we are creating the string corresponding to the individual's genotype
    control_law_variables = [v for v in candidate if v != "id_"]
    
    for variable in control_law_variables :
        program = candidate[variable]
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
    genotype2id = args["genotype2id"]
    id2genotype = args["id2genotype"]

    # we can take into account the idea of having individuals
    # as a dictionary of equations, one per control function, depending
    # on the viability problem; so, first let's check the structure of the controls
    # I added another element to the individual's dictionary, "id_", to keep track
    # of individuals' lineage (and solve some bugs)
    individual = dict()

    logger.debug("Generating new individual...")
    for control_variable in args["vp_control_structure"] :
        individual[control_variable] = _Program(**args["gplearn_settings"])
        logger.debug("For control variable \"%s\", equation \"%s\"" % (control_variable, equation_string_representation(individual[control_variable])))
        
    # this part is to keep track of individuals' lineage
    individual_id = args["individual_id"]
    individual["id_"] = individual_id
    
    # these dictionaries are also used to keep track of individuals' genotype
    genotype = individual2string(individual)
    if genotype not in genotype2id :
        genotype2id[genotype] = []
    genotype2id[genotype].append(individual_id)
    
    id2genotype[individual_id] = {"genotype" : genotype, "parent" : "None", 
                                  "created_by" : "generator", "generation" : 0} 
    
    args["individual_id"] += 1
        
    return individual


def fitness_function(individual, args) :
    """
    Fitness function for an individual.
    """

    logger = args["logger"]
    saturate_control_function_on_boundaries = args["saturate_control_function_on_boundaries"]

    fitness = 0.0
    initial_conditions = args["current_initial_conditions"]
    vp = args["viability_problem"]
    time_step = args["time_step"]
    max_time = args["max_time"]

    # evaluating one solution takes a considerable amount of time, so we will first make a check;
    # if all control rules defined in the individual reduce to a constant, we can discard it 
    control_rules = { variable : equation_string_representation(control_rule) 
                     for variable, control_rule in individual.items() if variable != "id_" } # 'id_' is a special tag
    #logger.debug("Now evaluating individual corresponding to \"%s\"..." % control_rules)

    # the try / except statement has been moved up here, so that we also catch exceptions 
    # due to sympy not being able to analyze the candidate individual (a problem that came out with the sphere)
    try :

        is_rule_constant = []
        for variable, rule in control_rules.items() :
            expression = sympy.sympify(rule)
            is_rule_constant.append(expression.is_constant())

        if all(is_rule_constant) :
            #logger.debug("All control rules of the individual are constant, discarding...")
            return fitness

        # otherwise, create a local copy of the viability problem, that will be modified only here
        vp = copy.deepcopy(vp)
        # modify it, so that the control rules are now the same as the individual
        vp.set_control(control_rules)
        state_variables = [ v for v in vp.equations ]

        # and now, we run a simulation for each initial condition
        for ic in initial_conditions :
            with no_stderr_stdout() : # try to mute the standard output and the standard error # NOTE: it does not work
                #logger.debug("Now running simulation for initial conditions %s..." % str(ic))
                # there might be some crash here, so we perform exception handling; we also set a timeout of 10 minutes PER CONDITION that will raise an exception
                #signal.signal(signal.SIGALRM, timeout_handler)
                #signal.alarm(360)
                output_values, constraint_violations = vp.run_simulation(ic, time_step, max_time, saturate_control_function_on_boundaries=saturate_control_function_on_boundaries) 

                # compute the fitness, based on how long the simulation ran before a constraint violation
                fitness += len(output_values[state_variables[1]]) / (max_time/time_step)

                # reset the alarm
                #signal.alarm(0)

    except Exception :
        # if executing the control rules raises an exception, fitness becomes zero
        #logger.debug("Individual \"%s\" created an exception, it will have fitness zero" % control_rules)
        fitness = 0.0

        # also reset the alarm, just to be safe
        #signal.alarm(0)

    #logger.debug("Fitness for individual \"%s\" is %.4f" % (control_rules, fitness))
    # normalize fitness with respect to the number of initial conditions, so that it is always in (0.0,1.0)
    fitness /= len(initial_conditions)

    return fitness


def timeout_handler(num, stack) :
    """
    Utility function that is called when a timeout expires; raises an exception, that should be caught by the exception handler inside fitness_function
    """
    raise TimeoutError


def evaluator_multiprocess(candidates, args) :
    """
    Wrapper of the fitness function, used for experiments with multiple processes (and not threads).
    """
    
    fitness_list = []
    for c in candidates :
        fitness_list.append( fitness_function(c, args) )

    return fitness_list


@inspyred.ec.variators.crossover
def variator(random, parent1, parent2, args) :
    """
    The variator for this particular problem has to be decorated as crossover, as it can perform multiple operations on one or two individuals.
    """
    logger = args["logger"]
    random_state = args["random_state"] # we are going to use a numpy pseudo-random number generator, because gplearn likes that
    p_crossover = args["p_crossover"]
    p_subtree = args["p_subtree"]
    p_hoist = args["p_hoist"]
    p_point = args["p_point"]
    
    # this is used to keep track of individuals' lineage
    id2genotype = args["id2genotype"]
    genotype2id = args["genotype2id"]

    offspring = []

    # we are going to loop until the new individual is at least a bit different from individual1
    # these two variables below will be used to loop until a different individual is created
    n_attempts = 1
    is_offspring_equal_to_parent = True
    
    # these variables below are used to keep track of the lineage of an individual
    parent_id = parent1["id_"]
    created_by = ""
    generation = args["_ec"].num_generations

    while is_offspring_equal_to_parent == True :
        # the genome of an individual is a dictionary, with one gplearn program for each control variable
        # this variator should take that into account, and consider that sometimes it's better to just change
        # one or few of the control rules, to preserve locality; at the moment, there is a probabilty that
        # nothing happens, but then we have to check that the new individual is actually different from individual1
        logger.debug("Generating new individual, attempt %d..." % n_attempts)
        new_individual = { variable : None for variable in parent1 }

        # we have to reset the "created_by" information, in case we are looping
        created_by = ""
        
        # get the control laws in the individual (ignoring the key "id_" in the
        # dictionary, as it is only used to keep track of the individuals' lineage)
        control_law_variables = [v for v in new_individual if v != "id_"]

        program = None
        for variable in control_law_variables :
            # let's select a probability to apply one of the mutation operators
            p = random_state.uniform()
            logger.debug("Creating gplearn program for state variable \"%s\"..." % variable)

            new_program = None
            if p < p_crossover :
                logger.debug("Performing a crossover...")
                program, removed, remains = parent1[variable].crossover(parent2[variable].program, random_state)
                created_by = "crossover"
                
            elif p < (p_crossover + p_subtree) :
                logger.debug("Performing a subtree mutation...")
                program, removed, _ = parent1[variable].subtree_mutation(random_state)
                created_by = "subtree_mutation"

            elif p < (p_crossover + p_subtree + p_hoist) :
                logger.debug("Performing a hoist mutation...")
                program, removed = parent1[variable].hoist_mutation(random_state)
                created_by = "hoist_mutation"

            elif p < (p_crossover + p_subtree + p_hoist + p_point) :
                logger.debug("Performing a point mutation...")
                program, mutated = parent1[variable].point_mutation(random_state)
                created_by = "point_mutation"

            else :
                logger.debug("Copying the original genome from parent1...")
                program = parent1[variable].reproduce()
                created_by = "copy"

            # also keep track of the state variable/control law on which we operated
            created_by += " (" + str(variable) + ");"            

            # create new instance of _Program
            new_program = _Program(function_set=parent1[variable].function_set,
                       arities=parent1[variable].arities,
                       init_depth=parent1[variable].init_depth,
                       init_method=parent1[variable].init_method,
                       n_features=parent1[variable].n_features,
                       metric=parent1[variable].metric,
                       transformer=parent1[variable].transformer,
                       const_range=parent1[variable].const_range,
                       p_point_replace=parent1[variable].p_point_replace,
                       parsimony_coefficient=parent1[variable].parsimony_coefficient,
                       feature_names=parent1[variable].feature_names,
                       random_state=random_state,
                       program=program)

            logger.debug("New program generated: %s" % str(new_program))
            new_individual[variable] = new_program
            
        # check if the new individual is identical to individual 1
        logger.debug("New candidate individual: \"%s\"" % individual2string(new_individual))
        logger.debug("Parent individual to be compared against: %d \"%s\"" % (parent1["id_"], individual2string(parent1)))
        
        try :
            is_offspring_equal_to_parent = are_individuals_equal(parent1, new_individual)
        except Exception :
            is_offspring_equal_to_parent = True

        if not is_offspring_equal_to_parent :
            offspring.append(new_individual)
            
            # add information on the lineage to the dictionaries
            individual_id = args["individual_id"]
            genotype = individual2string(new_individual)
            id2genotype[individual_id] = {"genotype" : genotype, "parent" : parent_id, 
                                     "generation" : generation, "created_by" : created_by}
            if genotype not in genotype2id :
                genotype2id[genotype] = []
            genotype2id[genotype].append(individual_id)
            
            # increase individual id
            logger.debug("Id of the freshly created new individual: %d" % individual_id)
            new_individual["id_"] = args["individual_id"]
            args["individual_id"] += 1
        else :
            logger.debug("The two individuals are exactly identical! I should re-loop and create a new one")
            n_attempts += 1

        # end of while: if the new individual is equal to parent individual1, we loop and try again 

    return offspring


def replacer(random, population, parents, offspring, args) :
    """
    This is a custom replacer, that basically rescales the fitness of all individuals in the population, based on the fitness of the best
    individuals among the parents. This is necessary because individuals are evaluated on a different set of initial conditions at each
    generations, and it would be interesting to know whether the apparent improvement seen over the generations is actually due to real
    improvements in the best individual, or just random luck in the choice of the initial conditions (more viable vs non-viable initial
    conditions).
    """
    # TODO the difficult part here is to store the non-scaled and scaled fitness values separatedly 

    logger = args["logger"]
    logger.debug("Starting the replacement procedure...")
    
    # some debugging here
    
    logger.debug("Initial state of the parents:")
    for p in parents :
        genotype = individual2string(p.candidate)
        individual_id = p.candidate["id_"]
        logger.debug(str(individual_id) + ":" + genotype)
        
    logger.debug("Initial state of the population:")
    for i in population :
        genotype = individual2string(i.candidate)
        individual_id = i.candidate["id_"]
        logger.debug(str(individual_id) + ":" + genotype)    
        
    logger.debug("Initial state of the offspring:")
    for o in offspring :
        genotype = individual2string(o.candidate)
        individual_id = p.candidate["id_"]
        logger.debug(str(individual_id) + ":" + genotype)
        
    survivors = sorted(population + offspring, key=lambda x : x.fitness, reverse=True)
    
    logger.debug("Final sorting of population+offspring, before culling:")
    for s in survivors :
        genotype = individual2string(s.candidate)
        individual_id = p.candidate["id_"]
        logger.debug(str(individual_id) + ":" + genotype)
    
    # this part below is not used at the moment, it implements the fitness rescaling
    
    # find the best individual among the parents
    # best_parent = parents.sort(reverse=True) # highest value first
    # logger.debug("The best individual is: \"%s\"; re-evaluating on initial conditions %s" % 
    #         (individual2string(best_parent.candidate), str(args["current_initial_conditions"])))

    # # evaluate the best parent on the current initial conditions
    # old_best_fitness = best_parent.fitness
    # new_best_fitness = fitness_function(best_parent.candidate, args)
    # logger.debug("Previous fitness for best individual: %4.f; new fitness: %.4f" % (old_best_fitness, new_best_fitness))

    # # rescale old fitness values according to the new fitness value of the best individual:
    # # - parents' fitness is rescaled by fitness_value * new_best_fitness / old_best_fitness
    # for parent in parents :
    #     parent.fitness = parent.fitness * new_best_fitness / old_best_fitness

    # # now, sort by fitness and save the best using a mu+lambda scheme
    # survivors = sorted(parents + offspring, lambda x : x.fitness, reverse=True)

    return survivors[:len(population)]


def observer(population, num_generations, num_evaluations, args) :
    """
    The observer is a classic function for inspyred, that prints out information and/or saves individuals. However, it can be easily re-used by other
    evolutionary approaches.
    """

    logger = args["logger"]
    id2genotype = args["id2genotype"]
    genotype2id = args["genotype2id"]
    best_individual = max(population, key=lambda x : x.fitness)

    print(genotype2id)
    print(id2genotype)
    
    # get some information
    best_string = individual2string(best_individual.candidate)
    best_fitness = best_individual.fitness

    # print the equation(s) corresponding to the best individual
    logger.info("Generation #%d (%d evaluations) Best individual: \"%s\"; Best fitness: %.8f" % (num_generations, num_evaluations, best_string, best_fitness))
    
    # save the current generation
    file_generation = os.path.join(args["directory_output"], "generation-%d.csv" % num_generations)
    logger.debug("Saving current generation to file \"%s\"..." % file_generation)

    dictionary_generation = {
            "generation" : [],
            "id" : [],
            "id_in_genotype_dictionary" : [],
            "birthdate" : [],
            "individual" : [],
            "individual_gplearn" : [],
            "fitness" : [],
            "generation_created" : [],
            "created_by" : [],
            "parent" : [],
            }

    dictionary_generation["generation"] = [num_generations] * len(population)

    for i, ci in enumerate(args["current_initial_conditions"]) :
        dictionary_generation["initial_conditions_%d" % i] = [str(ci)] * len(population)

    for individual in population :
        string_representation = individual2string(individual.candidate)
        dictionary_generation["individual"].append(string_representation)
        dictionary_generation["individual_gplearn"].append(individual_to_gplearn_string(individual.candidate))
        dictionary_generation["fitness"].append(individual.fitness)
        dictionary_generation["birthdate"].append(individual.birthdate)
        
        # here we collect information from the data structures used to track
        # individuals' phyolgeny
        individual_id = genotype2id.get(string_representation, -1)
        dictionary_generation["id_in_genotype_dictionary"].append(individual_id)
        
        # but now we can directly use the individual "id_" tag inside the individual
        individual_id = individual.candidate["id_"]
        dictionary_generation["id"].append(individual_id)
        
        generation_created = -1
        created_by = "Not found"
        parent = "Not found"
        
        if individual_id in id2genotype :
            logger.info("Individual id %d found in dictionary" % individual_id)
            generation_created = id2genotype[individual_id]["generation"]
            created_by = id2genotype[individual_id]["created_by"]
            parent = id2genotype[individual_id]["parent"]
        else :
            logger.info("Individual id %d not found!" % individual_id)
            
        dictionary_generation["generation_created"].append(generation_created)
        dictionary_generation["created_by"].append(created_by)
        dictionary_generation["parent"].append(parent)
        

    # conver the dictionary to a pandas DataFrame, and sort it by descending fitness (easier to read later)
    df = pd.DataFrame.from_dict(dictionary_generation)
    df.sort_values(by=["fitness"], ascending=False, inplace=True)
    df.to_csv(file_generation, index=False)

    # if the multi-process evaluation is active, we need to draw a new set of initial conditions here
    # TODO I need to understand where the multiprocess thing is defined, and how to check it; I could set a flag
    initial_conditions = [ args["viability_problem"].get_random_viable_point(args["random"]) for i in range(0, args["n_initial_conditions"]) ]
    logger.debug("Initial conditions for generation %d: %s" % (num_generations+1, str(initial_conditions)))
    args["current_initial_conditions"] = initial_conditions

    return

def queue_consumer_process(queue, logger) :
    """
    Process that consumes items from a queue where the other processes write
    """
    while True :
        item = queue.get()
        if item is not None :
            logger.debug(item)
        else :
            break

    return


def multi_process_evaluator(candidates, args) :
    """
    Alternative multi-process function, that tries to impose timeouts. It's more of an attempt to see if we can improve
    inspyred's multi-process evaluation. Uses the multiprocessing Python package.
    """
    n_processes = args["n_threads"]
    logger = args["logger"]

    # create shared fitness list, using a Manager to arbitrate concurrent access
    fitness_list = [0.0] * len(candidates)
    with Manager() as manager :

        # create shared fitness list and a lock
        shared_fitness_list = manager.list(fitness_list)
        #lock = manager.Lock() # TODO this was an early attempt to use Lock() to manage shared resources; it worked poorly
        queue = manager.Queue() # this is used to store and write logging messages TODO maybe setting a maxsize=1000 or something could help in case of issues

        # create a process pool
        logger.debug("Setting up process pool...")
        pool = Pool(n_processes) 

        # map arguments to processes 
        arguments_list = []
        for index, candidate in enumerate(candidates) :
            arguments_list.append([candidate, args, shared_fitness_list, index, queue])

        # start queue consumer process
        queue.put("Queue consumer process started!")
        queue_process = Process(target=queue_consumer_process, args=(queue, logger))
        queue_process.start()

        # start evaluation
        logger.debug("Starting multi-process evaluation of %d individuals..." % len(candidates))
        pool.map(process_evaluator, arguments_list)

        # here the shared fitness list should be copied on the actual fitness list
        for i in range(0, len(shared_fitness_list)) :
            fitness_list[i] = shared_fitness_list[i]

        # terminate queue consumer process
        queue.put("Queue consumer process finished.")
        time.sleep(5)
        queue_process.terminate()

    return fitness_list


def process_evaluator(arguments) :
    """
    Wrapper function for multi-process evaluation. It should set a timeout and manage exclusive access to the fitness list, to avoid issues of synchronization.
    """
    candidate, args, shared_fitness_list, index, queue = arguments

    # let's try to perform some debugging, and for that, we need to get the process PID
    logger = args["logger"]
    process = multiprocessing.current_process()
    pid = process.pid
    control_rules = { variable : equation_string_representation(control_rule) 
                     for variable, control_rule in candidate.items() if variable != "id_"}
    
    # we need to lock access to the logger, to avoid multiple processes from trying to use it at the same time
    # TODO there is a big mystery here: on a 64-core processor, everything goes well; on an 8-core, everything gets stuck; investigate use of 'nice'
    #lock.acquire()
    #logger.debug("[%s] Starting evaluation of candidate %d..." % (str(pid), index))
    #logger.debug("Starting evaluation of a candidate...")
    #lock.release()
    # instead of using a Lock() object, we are now attempting to employ a Queue()
    queue.put("[%s] Starting evaluation of candidate %d..." % (str(pid), index))

    # we start a timeout here, the exception raised by timeout_handler should be caught inside the function
    # unfortunately, the SIGALARM only works on Linux ^_^;
    if platform.system() == 'Linux' :
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(3600*2) # two hours

    try :
        # call the true evaluation function
        fitness_value = fitness_function(candidate, args)  

        # wrap up the evaluation
        #lock.acquire()
        #logger.debug("[%s] Evaluation of candidate %d finished." % (str(pid), index))
        #logger.debug("Evaluation of candidate finished.")
        #lock.release()
        queue.put("[%s] Evaluation of candidate %d finished." % (str(pid), index))
        
        # reset alarm; again, it only works on Linux systems
        if platform.system() == 'Linux' :
            signal.alarm(0)

    except TimeoutError as te :
        fitness_value = 0.0
        #lock.acquire()
        #logger.debug("[%s] Evaluation of candidate %d \"%s\" failed due to a timeout." % (str(pid), index, str(control_rules)))
        #lock.release()
        queue.put("[%s] Evaluation of candidate %d \"%s\" failed due to a timeout." % (str(pid), index, str(control_rules)))
        signal.alarm(0)

    shared_fitness_list[index] = fitness_value

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
    fitness_value = fitness_function(individual, args) 

    # thread_lock is a threading.Lock object used for synchronization and avoiding
    # writing on the same resource from multiple threads at the same time
    thread_lock.acquire()
    fitness_list[index] = fitness_value 
    thread_lock.release()

    logger.debug("[Thread %d] Evaluation finished." % thread_id)

    return


def evolve_rules(viability_problem, random_seed=0, n_initial_conditions=10, 
                 time_step=0.1, max_time=100, n_threads=8, pop_size=100, offspring_size=200, 
                 max_evaluations=1000, saturate_control_function_on_boundaries=False, 
                 directory_name="viability-theory") :
    """
    This function implements the main evolutionary loop.

    Parameters
    ----------
    viability_problem : TYPE
        DESCRIPTION.
    random_seed : TYPE, optional
        DESCRIPTION. The default is 0.
    n_initial_conditions : TYPE, optional
        DESCRIPTION. The default is 10.
    time_step : TYPE, optional
        DESCRIPTION. The default is 0.1.
    max_time : TYPE, optional
        DESCRIPTION. The default is 100.
    n_threads : TYPE, optional
        DESCRIPTION. The default is 8.
    pop_size : TYPE, optional
        DESCRIPTION. The default is 100.
    offspring_size : TYPE, optional
        DESCRIPTION. The default is 200.
    max_evaluations : TYPE, optional
        DESCRIPTION. The default is 1000.
    saturate_control_function_on_boundaries : TYPE, optional
        DESCRIPTION. The default is False.
    directory_name : TYPE, optional
        DESCRIPTION. The default is "viability-theory".

    Returns
    -------
    None.

    """

    # create directory with name in the date
    directory_output = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-" + directory_name 
    if not os.path.exists(directory_output) :
        os.makedirs(directory_output)

    # start logging
    logger = initialize_logging(path=directory_output, log_name="log", date=True)
    logger.info("Starting, all results will be saved in folder \"%s\"..." % directory_output)
    logger.info("Setting up evolutionary algorithm...")

    # hard-coded values, probably to be replaced with function arguments
    time_step = time_step #0.1
    max_time = max_time #100

    # initialize the pseudo-random number generators
    logger.info("All pseudo-random number generators are initialized with seed: %d" % random_seed)
    prng = random.Random(random_seed)
    nprs = np.random.RandomState(random_seed) 

    # first, we interrogate the viability problem to check how many variables and control rules we're talking about
    # viability_problem.control is a dictionary 
    vp_control_structure = viability_problem.control
    vp_variables = [ variable for variable in viability_problem.equations ]

    # then, we setup all necessary values for GPlearn individuals, _Program instances
    # the first thing we do is to add other functions to the function set
    function_set = [ _function_map[f] for f in ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'sin', 'cos'] ]

    # these are added separately, because they are not part of gplearn's base
    # function set, but have been added by us
    logger.debug("Adding extra functions to gplearn's function set...")
    f_min = make_function(function=_min, name="min", arity=2)
    f_max = make_function(function=_max, name="max", arity=2)
    function_set.extend([f_min, f_max])

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
    
    # it could be useful to keep track of the individuals' lineage, here is a
    # data structure (or two) that will be used for that
    genotype2id = {} # string conversion to [individual id_1, individual_id_2, ...]
    id2genotype = {} # individual id to: {string conversion, generation}
    individual_id = 0

    # we use inspyred's base stuff to manage the evolution
    ea = inspyred.ec.EvolutionaryComputation(prng)
    # these functions are pre-programmed in inspyred
    ea.selector = inspyred.ec.selectors.tournament_selection 
    ea.replacer = replacer #inspyred.ec.replacers.plus_replacement
    ea.terminator = inspyred.ec.terminators.evaluation_termination
    # these functions are custom and inside this script
    ea.variator = variator
    ea.observer = observer

    # if multi-process evaluation is active, we need to draw the random initial conditions for generation 0 here
    logger.debug("Generating initial conditions for generation 0...")
    current_initial_conditions = [ viability_problem.get_random_viable_point(prng) for i in range(0, n_initial_conditions) ]
    logger.debug("Initial conditions for generation 0: %s" % str(current_initial_conditions))

    logger.info("Starting evolutionary optimization...")
    final_population = ea.evolve(
                            generator=generator,
                            #evaluator=multi_thread_evaluator, # uncomment this for personalized multi-threaded evaluation of individuals
                            #evaluator=inspyred.ec.evaluators.parallel_evaluation_mp, # uncomment the following three lines for multi-process evaluation
                            #mp_evaluator=evaluator_multiprocess,
                            #mp_num_cpus=n_threads,
                            evaluator=multi_process_evaluator, # uncomment this to use a personalized version of multi-processing, with timeout
                            pop_size=pop_size,
                            num_selected=offspring_size,
                            maximize=True,
                            max_evaluations=max_evaluations,

                            # all items below this line go into the 'args' dictionary passed to each function
                            directory_output = directory_output,
                            logger = logger,
                            n_threads = n_threads,
                            random_seed = random_seed, # this is the numpy random number generator, used by gplearn
                            random = prng, # this is the random.Random instance used by inspyred
                            random_state = nprs, # and this is a numpy random number generator
                            vp_control_structure = vp_control_structure,
                            viability_problem = viability_problem,
                            # these parameters below are used for the evaluation of candidate solutions
                            time_step = time_step,
                            max_time = max_time,
                            n_initial_conditions = n_initial_conditions,
                            current_initial_conditions = current_initial_conditions,
                            # settings for gplearn
                            gplearn_settings = gplearn_settings,
                            p_crossover = 0.4,
                            p_hoist = 0.1,
                            p_subtree = 0.1,
                            p_point = 0.1,
                            # should we saturate the control rules on the boundaries?
                            saturate_control_function_on_boundaries = saturate_control_function_on_boundaries,
                            # data structures to keep track of the individuals' lineage
                            id2genotype = id2genotype,
                            genotype2id = genotype2id,
                            individual_id = individual_id,
                            # this is a flag that will be used for rescaling the fitness
                            # TODO: at the moment, it is probably not used anywhere
                            rescale_fitness = True,
                            
                            )

    logger.info("Evolution terminated.")

    return

if __name__ == "__main__" :
    
    # this "main" function here is just a simple test on the Lake Eutrophication
    # case study
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
            "Lmin" : 0.1,
            "Lmax" : 1.0,
            "Pmax" : 1.4,
            }

    vp = ViabilityTheoryProblem(equations=equations, control=control, constraints=constraints, parameters=parameters)
    print("Evolving control rules for the following viability problem:", vp)

    evolve_rules(viability_problem=vp, random_seed=42, pop_size=10, offspring_size=10, 
                 n_threads=16, saturate_control_function_on_boundaries=False, 
                 directory_name="test-lake-eutrophication")

    sys.exit(0)

