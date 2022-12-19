"""
Script that implements the evolutionary loop to create control rules using Genetic Programming. I think we need to steal the individuals from gplearn, but I would like to keep the rest from inspyred, because it's better.
"""
import numpy as np
import sys

# steal parts from gplearn: _Program is the basically an individual
from gplearn._program import _Program

# local module
from viabilitytheory import ViabilityTheoryProblem

def generator(random, args) :

    return

def fitness_function(candidates, args) :

    fitness_list = np.zeros(len(candidates))
    # draw a given number of random tuples as starting conditions

    # for each given candidate control rule (individual)

        # for each random starting condition
            # solve the viability problem for the given control, ideally on a different thread
            # get the result, and compute the fitness function given its result

    return fitness_list

def evolve_rules(viability_problem, random_seed) :

    # we use inspyred's base stuff to manage the evolution

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

