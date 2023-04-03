# this is a quick and dirty script to plot a figure for the paper

import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys

from matplotlib.patches import Circle, Polygon, Rectangle

# all details of the individual
individual_best = {
        #"folder" : "../sphere_pc_trajectories_saturate_gen90",
        "folder" : "../sphere_pc_trajectories_saturate",
        "control" : "",
        "generation" : 100,
        "trajectory_files" : []
        }
individual_base = {
        "folder" : "../sphere_pc_trajectories_saturate",
        "control" : "",
        "generation" : 0,
        "trajectory_files" : []
        }

# first, let's read all trajectory files
individual_best["trajectory_files"] = [ f for f in os.listdir(individual_best["folder"]) if f.startswith("trajectory-best") and f.endswith(".csv") ]
individual_base["trajectory_files"] = [ f for f in os.listdir(individual_base["folder"]) if f.startswith("trajectory-base") and f.endswith(".csv") ]

# let's see what happens if we only plot 20 trajectories, more readable
#individual_best["trajectory_files"] = individual_best["trajectory_files"][:20]
#individual_base["trajectory_files"] = individual_base["trajectory_files"][:20]

for individual in [individual_best, individual_base] :
    for plane in [['x', 'y'], ['z', 'y']] :

        # get the trajectory files to be used for this individual
        trajectory_files = individual["trajectory_files"]

        # let's start organizing the figure
        sns.set_theme()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        # create polygonal patch that describes the frontier
        handle_polygon = ax.add_patch(Circle((0,0), radius=math.sqrt(1.5), edgecolor='blue', facecolor='blue', alpha=0.2, fill=True))

        # compute "fitness" of the selected files, and at the same time plot the trajectories (?)
        print("Computing fitness for the control...")
        fitness = 0.0
        for f in trajectory_files :
            df = pd.read_csv(os.path.join(individual["folder"], f))
            fitness += df.shape[0] / 10000

            # here, we need to convert the rho, theta, phi values to x, y, z values
            xyz = {'x' : [], 'y' : [], 'z' : []}
            for index, row in df.iterrows() :
                x = row["s"] * math.sin(row["theta"]) * math.cos(row["phi"]) 
                y = row["s"] * math.sin(row["theta"]) * math.sin(row["phi"])
                z = row["s"] * math.cos(row["theta"])

                xyz["x"].append(x)
                xyz["y"].append(y)
                xyz["z"].append(z)

            color = "green"
            line_style = "-"
            trajectory_label = "Viable trajectory"
            ic_label = "Viable initial conditions"
            if df.shape[0] < 10000 :
                color = "red"
                line_style = "--"
                trajectory_label = "Non-viable trajectory"
                ic_label = "Non-viable initial conditions"

            #x = df[plane[0]].values
            #y = df[plane[1]].values
            x = xyz[plane[0]]
            y = xyz[plane[1]]
            if len(x) > 0 : # trying to avoid a weird error
                handle_trajectory = ax.plot(x, y, color=color, linestyle=line_style, label=trajectory_label) 
                handle_ic = ax.scatter(x[0], y[0], marker='x', color=color, label=ic_label)
            else :
                print("Trajectory for file %s seems empty..." % f)

        # convert fitness to a percentage
        fitness = 100 * (fitness / (max(1, len(trajectory_files))))
        print("Total fitness (percentage): %.4f%%" % fitness)

        # this is done by hand...
        #handle_viable_area = ax.add_patch(Rectangle((0.1, 0.0), 1.0 - 0.1, 1.4 - 0.0, edgecolor='green', linestyle='--', facecolor='none', fill=False, label="Viable area")) 
        handle_viable_area = ax.add_patch(Circle((0,0), radius=math.sqrt(1.5), edgecolor='green', linestyle='--', facecolor='none', fill=False, label="Viable area"))

        # labeling
        ax.set_title("Best control for generation #%d (fitness=%.2f%%)" % (individual["generation"], fitness))
        ax.set_xlabel(plane[0])
        ax.set_ylabel(plane[1])

        ax.set_xlim(left=-1.3, right=1.3)
        ax.set_ylim(bottom=-1.3, top=1.3)
        fig.set_tight_layout(True)

        plt.savefig("sphere-pc-best-control-generation-%d-%s-%s.png" % (individual["generation"], plane[0], plane[1]), dpi=300)
        plt.show()
        plt.close(fig)
