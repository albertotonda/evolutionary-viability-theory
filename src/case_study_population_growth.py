"""
Simple script that evolves rules for the population viability problem.
"""

import evolutionary_control_rules
import viability_theory
import sys

if __name__ == "__main__" :

    # define the viability problem as a set of dictionaries
    equations = {
            "x" : "x * y",
            "y" : "u"
            }
    control = {"u" : ""}
    constraints = {
            "x" : ["x >= a", "x <= b"],
            "y" : ["y >= d", "y <= e"],
            "u" : ["u >= -c", "u <= c"]
            }
    parameters = {
            "a" : 0.2,
            "b" : 3.0,
            "c" : 0.5,
            "d" : -2.0,
            "e" : 2.0
            }
    # create instance of the viability problem class with the above dictionaries
    vp = viability_theory.ViabilityTheoryProblem(equations=equations, control=control, constraints=constraints, parameters=parameters)

    # start evolution of control rules
    print("Evolving control rules for the following viability problem:", vp)
    evolutionary_control_rules.evolve_rules(viability_problem=vp, random_seed=43)

    sys.exit(0)
