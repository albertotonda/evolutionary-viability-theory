# Evolutionary Viability Theory
The mathematical viability theory (VT) develops methods and tools to study the compatibility between dynamics and constraints. It is an effective approach to dealing with the management and control of complex systems. It is especially useful to model sustainability issues where the point is to preserve the ability to simultaneously satisfy economic, ecological or social constraints rather than optimizing an objective function weighting all concerns. Given dynamics and constraints, the VT theoretical tools provide the necessary and sufficient conditions to ensure that the constraints will be satisfied over time. The subset of the state space which gathers all viable states is called the viability kernel. These results are particularly valuable for public managers to define quotas for instance.

However, as the technique is being more and more applied to real-world scenarios, the community is now facing several obstacles in upscaling. In particular, (i) it is extremely difficult to devise control rules that make trajectories in time always respect the given boundaries, even when the viability kernel is known; (ii) computing or even just approximating the viability kernel for high-dimensional problems is currently unfeasible with the available algorithms.

Evolutionary algorithms have already proven their efficiency in solving high-dimensional control problems. We propose to study how these techniques can help to meet the following two objectives:

1. Automatically create control rules for trajectories in viability problems
2. Find satisfying approximations for viability kernels in high dimensionality (e.g. more than 6)

## Instructions
This repository contains:
* A class used to represent viability problems `ViabilityTheoryProblem` in `viability_theory.py`
* An evolutionary algorithm designed to evolve sets of control rules (equations represented as Genetic Programming trees) for a given `ViabilityTheoryProblem`, in `evolutionary_control_rules.py`
* Several case studies that rely upon the previous two: `case_study_lake.py`, `case_study_3d_sphere.py`, `case_study_pc_sphere.py`
* The other scripts are used for validation of the best solutions found, or to plot figures

## Papers
The code in this repository has been used for the following paper:

Alberto Tonda, Isabelle Alvarez, Sophie Martin, Giovanni Squillero, and Evelyne Lutton. 2023. Towards Evolutionary Control Laws for Viability Problems. In Genetic and Evolutionary Computation Conference (GECCO ’23), July 15–19, 2023, Lisbon, Portugal. ACM, New York, NY, USA. https://doi.org/10.1145/3583131.3590415
