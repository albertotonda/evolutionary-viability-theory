"""
Series of scripts to plot an individual's behavior. For the moment, it might only work in a two-dimensional case.
"""
import itertools
import math
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

def draw_arrow(ax, arr_start, arr_end, color='red'):
    """
    Utility function to draw an arrow on a line
    """
    dx = arr_end[0] - arr_start[0]
    dy = arr_end[1] - arr_start[1]
    ax.arrow(arr_start[0], arr_start[1], dx, dy, head_width=0.001, head_length=0.001, length_includes_head=True, color=color)

    return

def plot_vp_trajectories(vp, initial_conditions, time_step=0.1, max_time=100, color=None, label=None, fig=None) :
    """
    Function that plots trajectories or just add trajectories to an existing figures, with a certain color or caption
    """

    # set up some variables that will be useful later
    fig = fig
    axs = None

    # from the viability problem, we get the state variables; depending on the number of
    # state variables, we might have to add extra subplots
    variables = [ v for v in vp.equations ]
    # this function give us all unique permutations of non-repeated values from a list (so ["1", "2"] and ["2", "1"] are considered identical)
    variable_pairs = list(itertools.combinations(variables, 2))
    print(variable_pairs)

    # this sets a cool theme
    sns.set_theme()

    # if no figure is specified, we need to create one
    if fig is None :

        # get the number of necessary subplots
        number_of_subplots = len(variable_pairs)
        # determine the number of columns and rows that I need, knowing that I want maximum three columns
        max_columns = 3.0
        number_of_rows = math.ceil(number_of_subplots / max_columns)
        number_of_columns = min(number_of_subplots, max_columns)

        fig, axs = plt.subplots(nrows=number_of_rows, ncols=number_of_columns)

        # unfortunately, matplotlib decided that's it's a great idea to return one single ax
        # if there is only one, and a list if there is more than one; so we need to check
        # and repair; these lines are repeated because I could not find another good way of doing this
        if not hasattr(axs, '__iter__') :
            axs = [axs]

        # preliminary stuff: we go variable pair by variable pair, and plot a square around the viable area in the 2D plot
        for index, variable_pair in enumerate(variable_pairs) :

            var_x = variable_pair[0]
            var_y = variable_pair[1]
            ax = axs[index]

            x_min, x_max = vp.get_variable_boundaries(var_x)
            y_min, y_max = vp.get_variable_boundaries(var_y)

            # TODO check that no value is infinite, here...
            ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                edgecolor='green', linestyle='--', facecolor='none', fill=False, label="Viable area")) 

    else :

        # get all subplots that are already inside the figure
        axs = fig.axes
        if not hasattr(axs, '__iter__') :
            axs = [axs]

    # we set this parameter to autoresize the canvas
    fig.set_tight_layout(True)

    # now, we solve the differential equations in the viability problem instance for each set of initial conditions
    print("Solving differential equations for the initial conditions...")
    trajectories = []
    constraint_violations = []
    for ic in initial_conditions :
        trajectory, constraint_violation = vp.run_simulation(ic, time_step, max_time)
        trajectories.append( trajectory )
        constraint_violations.append( constraint_violation )

    # then, we go subplot by subplot, depending on the variable pairs; the pairs will be plot as 'x' or 'y' in order of appearance 
    for index, variable_pair in enumerate(variable_pairs) :
        var_x = variable_pair[0]
        var_y = variable_pair[1]

        ax = axs[index]

        # TODO if no color was specified, create a cool colormap here
        # TODO let's try to make something different, when trajectories with a constraint violation are in red, the others are in green

        # transform the list of initial conditions in two arrays, one for variable x and one for variable y
        ic_plot_x = []
        ic_plot_y = []

        for i_ic, ic in enumerate(initial_conditions) :
            ic_plot_x.append(ic[var_x])
            ic_plot_y.append(ic[var_y])

        #ax.scatter(ic_plot_x, ic_plot_y, marker='x', color=color)

        # now, plot the trajectories for this specific set of two variables
        for index, trajectory in enumerate(trajectories) :
            x_ic = initial_conditions[index][var_x]
            y_ic = initial_conditions[index][var_y]

            x = trajectory[var_x]
            y = trajectory[var_y]

            trajectory_label = None
            ic_label = None
            if index == 0 :
                trajectory_label = label
                ic_label = "Initial conditions"

            # let's find the color
            trajectory_color = 'green'
            if len(constraint_violations[index]) > 0 :
                trajectory_color = 'red'
            
            # this marks the initial condition
            ax.scatter(x_ic, y_ic, marker='x', color=trajectory_color, label=ic_label)
            # this draws the trajectory
            ax.plot(x, y, color=trajectory_color, label=trajectory_label) 

            # now, if the trajectory at a certain point became non-viable, I want to annotate why
            if trajectory_color == 'red' :
                #print(constraint_violations[index])
                text = str(constraint_violations[index][0]['constraint_violated'])
                t = ax.text(x_ic + 0.02, y_ic + 0.03, text, fontsize=6)
                t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='black'))

            # this draws an arrow
            #ax.arrow(x[-1], y[-1], 0.01, 0.01, length_includes_head=True, head_width=.05, color=color)
            #draw_arrow(ax, (x[0], y[0]), (x[-1], y[-1]), color=color)

        ax.set_title("Plot for %s, %s" % (var_x, var_y))
        ax.set_xlabel(var_x)
        ax.set_ylabel(var_y)
        ax.legend(loc='best')

    return fig

if __name__ == "__main__" :

    import pandas as pd
    import sys

    from viability_theory import ViabilityTheoryProblem

    # set up classical viability problem of the lake
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

    # read a file with a few individuals, find the row with the highest value, get the best individual
    results_file = "2022-12-22-14-56-38-viability-theory/generation-29.csv"
    print("Reading file \"%s\" and finding best individual..." % results_file)

    df = pd.read_csv(results_file)
    index_max = df["fitness"].idxmax()
    print("Found row with max value of fitness, index %d" % index_max)
    best_row = df.iloc[index_max]

    # TODO for the future, this part will use ast.literal_eval() instead, as I am going to save individuals as dictionary strings
    best_individual = best_row["individual"].split(" ")[2] # this only works for strings like "u -> sin(P)", in case of multiple controls, it has to be rewritten
    print("Best control rule found: \"%s\"" % best_individual)
    vp.set_control({"u" : best_individual[1:-1]})

    # find the initial conditions that were used to test the individual
    initial_conditions_columns = [ c for c in df.columns if c.startswith("initial_conditions") ]
    print("Found %d columns with initial conditions" % len(initial_conditions_columns))
    
    # get the list of values
    import ast
    initial_conditions = []
    for c in initial_conditions_columns :
        string = best_row[c]
        initial_conditions.append(ast.literal_eval(string))

    print(initial_conditions)

    # set up all other details and solve the differential equations for each initial conditions
    time_step = 0.1
    max_time = 100

    fig = plot_vp_trajectories(vp, initial_conditions, time_step=time_step, max_time=max_time, color='red', label='Trajectories')
    plt.savefig("figure-initial-conditions-training.png", dpi=300)

    sys.exit(0)

