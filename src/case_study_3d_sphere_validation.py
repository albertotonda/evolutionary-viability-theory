
"""
Simple script to validate the individuals from the sphere problem and plot figures.
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
    directory_name = "sphere_trajectories"
    ic_file_name = "sphere_ic.csv"
    trajectory_base_file_name = "trajectory-base-%d.csv"
    trajectory_best_file_name = "trajectory-best-%d.csv"
    n_ic = 1000

    control_generation_0 = {'u_x' : '((((-0.5125)-(-0.4354))-x)+x)', 'u_y' : 'sin((x+x))', 'u_z' : '(cos(z)+x)'}

    control_generation_72 = {'u_x' : '(y*(((-0.9655)/y)*x))', 'u_y' : 'cos(cos(((y+log(cos((z*(-0.4505)))))*(0.3767))))', 'u_z' : '(z*sin((-0.6640)))'}

    control_generation_85 = {'u_x' : '(y*(((-0.9655)/y)*x))', 'u_y' : 'cos(cos(((y+log(cos((z*(-0.4505)))))*(0.3767))))', 'u_z' : '(z*sin((-0.6640)))'}

    # set up the viability problem
    equations = {
            "x" : "x + a * u_x",
            "y" : "y + a * u_y",
            "z" : "z + a * u_z"
            }
    control = {
            "u_x" : "",
            "u_y" : "",
            "u_z" : ""
            }
    constraints = {
            "x" : ["x * x + y * y + z * z < r"],
            "u_x" : ["u_x * u_x + u_y * u_y + u_z * u_z < 1"]
            }
    parameters = {
            "a" : 1.0,
            "r" : 1.5
            }

    # I also need to import the special inherited class for the sphere,
    # to get the proper random generation of initial conditions
    from case_study_3d_sphere import ViabilityTheoryProblemSphere

    # two instances: one with the best, one with base individual
    #vp_base = ViabilityTheoryProblemSphere(equations=equations, control=control, constraints=constraints, parameters=parameters)
    #vp_base.set_control(control_generation_0)

    vp_best = ViabilityTheoryProblemSphere(equations=equations, control=control, constraints=constraints, parameters=parameters)
    vp_best.set_control(control_generation_72)

    # create a directory to store trajectories 
    if not os.path.exists(directory_name) : os.mkdir(directory_name)

    # create file with initial trajectories, if it does not exist yet
    if not os.path.exists(os.path.join(directory_name, ic_file_name)) :
        df_dict = {"x" : [], "y" : [], "z" : []}
        for i in range(0, n_ic) :
            print("Generating random initial condition #%d" % i)
            ic = vp_base.get_random_viable_point(prng)
            print("Random initial condition:", ic)
            for k, v in ic.items() :
                df_dict[k].append(v)

        df = pd.DataFrame.from_dict(df_dict)
        df.to_csv(os.path.join(directory_name, ic_file_name), index=False)

    # (re-) read the file
    df = pd.read_csv(os.path.join(directory_name, ic_file_name))

    # let's try to run with a multi-processing pool
    print("Preparing a pool of workers...")
    pool = Pool(8)
    process_arguments = []

    for index, row in df.iterrows() :
        ic = {"x" : row["x"], "y" : row["y"], "z" : row["z"]}
        
        # add arguments for a vp_base and a vp_best run
        if index < 1001 :
            #process_arguments.append( [vp_base, ic, index, directory_name, trajectory_base_file_name] )
            process_arguments.append( [vp_best, ic, index, directory_name, trajectory_best_file_name] )

    print("Starting the pool!")
    pool.map(process, process_arguments)

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
