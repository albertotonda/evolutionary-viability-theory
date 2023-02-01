"""
Script for the case study 4D-Cylinder.
"""
import evolutionary_control_rules
import viability_theory
import sys

if __name__ == "__main__" :

    # define viability problem through dictionaries
    equations = {
            "x" : "a * u_x",
            "y" : "a * u_y",
            "z" : "a * u_z"
            }
    control = {
            "u_x" : "",
            "u_y" : "",
            "u_z" : ""
            }
    constraints = {
            "x" : "x*x + y*y + z*z < r",
            "u_x" : "u_x * u_x + u_y * u_y + u_z * u_z < 1"
            }
    parameters = {
            "a" : 1.0,
            "r" : 1.5
            }
    vp = viability_theory.ViabilityTheoryProblem(equations=equations, control=control, constraints=constraints, parameters=parameters)

    # start the evolution
    print("Evolving control rules for the following viability problem:", vp)
    evolutionary_control_rules.evolve_rules(viability_problem=vp, random_seed=44)

    sys.exit(0)
