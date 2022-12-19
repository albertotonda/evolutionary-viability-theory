"""
Script that implements the evolutionary loop to create control rules using Genetic Programming.
"""
import sys

from viabilitytheory import ViabilityTheoryProblem

def fitness_function(individual, args) :

    return

def main(viability_problem, random_seed) :


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

    sys.exit( main(viability_problem=vp, random_seed=42) )
