"""
Simple script to validate the individuals from the lake eutrophication viability problem and plot figures.
"""
import os
import pandas as pd
import random
import sys

from multiprocessing import Pool, TimeoutError

# local libraries
from viability_theory import ViabilityTheoryProblem

def process(args) :

    vp, ic, index, directory_name, base_file_name = args

    file_name = os.path.join(directory_name, base_file_name % index)
    
    # there is a chance that the file already exists
    if not os.path.exists(file_name) :
        print("Running simulation for ic #%d..." % index)
        output_values, constraint_violations = vp.run_simulation(ic, 0.01, 100, saturate_control_function_on_boundaries=False) 
        print("Saving results to file \"%s\"..." % file_name)
        df = pd.DataFrame.from_dict(output_values)
        df.to_csv(file_name, index=False)
    else :
        print("File for ic #%d already exists!" % index)

    return

if __name__ == "__main__" :

    # hard-coded values
    random_seed = 4242
    prng = random.Random(random_seed)
    directory_name = "lake_trajectories_final"
    ic_file_name = "lake_ic.csv"
    trajectory_base_file_name = "trajectory-base-%d.csv"
    trajectory_best_file_name = "trajectory-best-%d.csv"
    n_ic = 100

    control_generation_0 = "sin((L+(-0.6992)))-(log(L)-((-0.7323)-(0.7294)))"
    control_generation_10 = "sin(sin((L+(-0.6992))))-(log(L)-((-0.7323)-(0.7294)))"
    control_generation_19 = "sin(log(L))-(log(L)-((-0.7323)-(0.7294)))"

    control_generation_0_previous_experiment_1 = "log(cos((sin(L)*L)))"
    control_generation_0_previous_experiment_2 = "sin((L+-0.6992))-(L+-0.6992)"
    control_generation_0_previous_experiment_3 = "sin(((P-L)*(L-P)))"
    
    # this is for the most recent experiment with min/max in the function set
    # and a large population
    control_generation_0 = "sin(min(sin(P),(P-P)))"
    control_generation_7 = "min((sin((-0.2277))*(L+(-0.2894))),sin(min(sin(P),(P-P))))"
    control_generation_10 = "min((sin((-0.2277))*(L+(-0.2894))),sin((L-L)))"
    control_generation_13 = "min((sin(sin((-0.2277)))*(L+(-0.2894))),sin((P-P)))"

    # set up the viability problem
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

    # two instances: one with the best, one with base individual
    #vp_base = ViabilityTheoryProblem(equations=equations, control=control, constraints=constraints, parameters=parameters)
    #vp_base.set_control({"u" : control_generation_0})

    #vp_best = ViabilityTheoryProblem(equations=equations, control=control, constraints=constraints, parameters=parameters)
    #vp_best.set_control({"u" : control_generation_0})
    
    vp_current = ViabilityTheoryProblem(equations=equations, control=control, constraints=constraints, parameters=parameters)
    vp_current.set_control({"u" : control_generation_13})
    trajectory_file_name = trajectory_best_file_name

    # create a directory to store trajectories 
    if not os.path.exists(directory_name) : os.mkdir(directory_name)

    # create file with initial trajectories, if it does not exist yet
    if not os.path.exists(os.path.join(directory_name, ic_file_name)) :
        df_dict = {"L": [], "P": []}
        for i in range(0, n_ic) :
            ic = vp_current.get_random_viable_point(prng)
            for k, v in ic.items() :
                df_dict[k].append(v)

        df = pd.DataFrame.from_dict(df_dict)
        df.to_csv(os.path.join(directory_name, ic_file_name), index=False)

    # (re-) read the file
    df = pd.read_csv(os.path.join(directory_name, ic_file_name))

    # let's try to run with a multi-processing pool
    print("Preparing a pool of workers...")
    pool = Pool(16)
    process_arguments = []

    for index, row in df.iterrows() :
        ic = {"L" : row["L"], "P" : row["P"]}
        
        # add arguments for a vp_base and a vp_best run
        if index < 1001 :
            #process_arguments.append( [vp_base, ic, index, directory_name, trajectory_base_file_name] )
            #process_arguments.append( [vp_best, ic, index, directory_name, trajectory_best_file_name] )
            process_arguments.append( [vp_current, ic, index, directory_name, trajectory_file_name] )

    print("Starting the pool!")
    pool.map(process, process_arguments)
    
    print("Computation terminated successfully.")

    sys.exit(0)

    # re-evaluate the fitness, but also keep track of the trajectories as save files
    trajectories = []
    for i in range(0, n_ic) :

        ic_file_name = trajectory_file_name % i
        if not os.path.exists(ic_file_name) :
            # draw random initial point
            print("Running simulation for initial conditions #%d..." % i)
            ic = vp.get_random_viable_point(prng)
            output_values, constraint_violations = vp.run_simulation(ic, 0.01, 100, saturate_control_function_on_boundaries=True) 

            print("Saving results to file \"%s\"..." % ic_file_name)
            df = pd.DataFrame.from_dict(output_values)
            df.to_csv(ic_file_name, index=False)

        else :
            print("File for trajectory %d found, skipping..." % i)

    # use the trajectories to create a plot 

    sys.exit(0)
