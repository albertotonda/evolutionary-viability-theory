# -*- coding: utf-8 -*-
"""
Script for a sample case study
"""
import matplotlib.pyplot as plt
import viability_theory
import sympy
import sys

from viability_theory import ViabilityTheoryProblem
from plot_utils import draw_arrow, plot_vp_trajectories

if __name__ == "__main__" :
    
    equations = {
            "X_1" : "X_1 + u",
            "X_2" : "X_2 + X_1 * X_2",
            }
    control = {
            "u" : "X_1 - X_2",
            }
    constraints = {
            "X_1" : ["X_1 >= 0", "X_1 < 10"],
            "X_2" : ["X_2 >= 0", "X_2 < 4"],
            "u" : []
            }
    parameters = {}
    vp = ViabilityTheoryProblem(equations=equations, control=control, constraints=constraints, parameters=parameters)
    
    initial_conditions = [{"X_1": 0.5, "X_2": 0.5}, {"X_1": 3, "X_2": 3}, {"X_1": 5, "X_2": 3}]
    fig = plot_vp_trajectories(vp, initial_conditions)
    good_trajectory = [[5,1], [6,1], [7,1.5], [8,2], [8,2.5], [7,2.5], [6,2.5], [5,2.5]]
    ax = fig.get_axes()[0]
    ax.scatter(good_trajectory[0][0], good_trajectory[0][1], color='green', marker='x', label="Viable initial conditions")
    ax.plot([x[0] for x in good_trajectory], [x[1] for x in good_trajectory], color='green', linestyle='--', label="Viable trajectory")
    #ax.get_legend().remove()
    #ax.legend(loc='best')
    plt.savefig("case_study_presentation.png", dpi=300)
    