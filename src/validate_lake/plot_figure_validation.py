# this is a quick and dirty script to plot a figure for the paper

import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys

from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle

# all details of the individual
individual_best = {
        #"folder" : "../lake_trajectories",
        #"folder" : "generation_19",
        "folder" : "../lake_trajectories_final",
        "control" : "",
        "generation" : 100,
        "trajectory_files" : []
        }
individual_base = {
        #"folder" : "../lake_trajectories",
        #"folder" : "generation_0",
        "folder" : "../lake_trajectories_final",
        "control" : "",
        "generation" : 0,
        "trajectory_files" : []
        }

# first, let's read all (or a part of the) trajectory files
for i in range(0, 250) :
    individual_best["trajectory_files"].append( os.path.join(individual_best["folder"], "trajectory-best-%d.csv" % i) )
    individual_base["trajectory_files"].append( os.path.join(individual_base["folder"], "trajectory-base-%d.csv" % i) )

#individual_best["trajectory_files"] = [ f for f in os.listdir(individual_best["folder"]) if f.startswith("trajectory-best") ]
#individual_base["trajectory_files"] = [ f for f in os.listdir(individual_base["folder"]) if f.startswith("trajectory-base") ]
#individual_base["trajectory_files"] = [ f for f in os.listdir("generation_0") if f.endswith(".csv") ]
#individual_best["trajectory_files"] = [ f for f in os.listdir("generation_19") if f.endswith(".csv") ]

for individual in [individual_best, individual_base] :

    # let's start organizing the figure
    sns.set_theme()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # create polygonal patch starting from the frontier
    df = pd.read_csv("lake_frontier_viability_kernel.csv")
    polygon_points = df[["L", "P"]].values.tolist()
    polygon_points.insert(0, [0.1, 1.4]) 
    polygon_points.append([0.1, 0.0]) 
    handle_polygon = ax.add_patch(Polygon(polygon_points, edgecolor='blue', facecolor='blue', alpha=0.2, fill=True))

    # compute "fitness" of the selected files, and at the same time plot the trajectories (?)
    print("Computing fitness for the control...")
    fitness = 0.0
    for f in individual["trajectory_files"] :
        df = pd.read_csv(os.path.join(individual["folder"], f))
        fitness += df.shape[0] / 10000

        color = "green"
        line_style = "-"
        trajectory_label = "Viable trajectory"
        ic_label = "Viable initial conditions"
        if df.shape[0] < 10000 :
            color = "red"
            line_style = "--"
            trajectory_label = "Non-viable trajectory"
            ic_label = "Non-viable initial conditions"

        x = df["L"].values
        y = df["P"].values
        handle_trajectory = ax.plot(x, y, color=color, linestyle=line_style, label=trajectory_label) 
        handle_ic = ax.scatter(x[0], y[0], marker='x', color=color, label=ic_label)

    # convert fitness to a percentage
    fitness = 100 * (fitness / (len(individual["trajectory_files"])))
    print("Total fitness (percentage): %.4f%%" % fitness)

    # this is done by hand...
    handle_viable_area = ax.add_patch(Rectangle((0.1, 0.0), 1.0 - 0.1, 1.4 - 0.0, edgecolor='green', linestyle='--', facecolor='none', fill=False, label="Viable area")) 

    # labeling
    ax.set_title("Best control for generation #%d (fitness=%.2f%%)" % (individual["generation"], fitness))
    ax.set_xlabel("L")
    ax.set_ylabel("P")

    fig.set_tight_layout(True)

    plt.savefig("lake-best-control-generation-%d.png" % individual["generation"], dpi=300)
    plt.show()
    plt.close(fig)
