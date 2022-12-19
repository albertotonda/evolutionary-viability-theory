"""
Set of scripts for managing viability theory problems. It's going to use symbolic computation.
"""
import copy
import sympy

from scipy import integrate

class ViabilityTheoryProblem :
    """
    Class that describes a viability theory problem
    """

    def __init__(self, equations={}, constraints={}, parameters={}, control={}) :

        # we assume that all ODEs are dX/dt
        self.equations = {}
        for state_variable, equation in equations.items() :
            self.equations[state_variable] = sympy.sympify(equation)

        self.control = control

        # we expect constraints to be a dictionary of lists of strings
        self.constraints = {}
        for variable, constraint_list in constraints.items() :
            self.constraints[variable] = []
            for c in constraint_list :
                self.constraints[variable].append( sympy.sympify(c) )

        self.parameters = parameters

    def set_control(self, new_control) :
        """
        Sets a new control value and/or equation. Expects a dictionary of controls, in the form {key : equation}
        """

        for variable, control_equation in new_control.items() :

            if variable in self.control :
                self.control[variable] = sympy.sympify(control_equation)
            else :
                raise Exception("While attempting to set new control, variable \"%s\" not found among internal control. self.control = \"%s\"" % (variable, self.control))

        return

    def run_simulation(self, initial_conditions, time_step, max_time) :
        """
        This part will actually solve the ODE system for a given set of initial conditions. Returns the values for each variable at each instant of time, and also the number and values of constraint violations.
        """

        # preliminary steps: replace all parameters in the equations for which we have values
        local_equations = { sympy.sympify(variable) : equation for variable, equation in self.equations.items() }
        local_parameters = { sympy.sympify(parameter) : value for parameter, value in self.parameters.items() }
        # include the control law inside the "parameters"
        for variable, control_equation in self.control.items() :
            local_parameters[sympy.sympify(variable)] = sympy.sympify(control_equation)

        for state_variable in local_equations :
            local_equations[state_variable] = local_equations[state_variable].subs(local_parameters) 

        print(local_equations)

        local_constraints = copy.deepcopy(self.constraints)
        for variable, constraint_list in local_constraints.items() :
            for i in range(0, len(constraint_list)) :
                constraint_list[i] = constraint_list[i].subs(local_parameters)

        print(local_constraints)

        # we define an internal "dX/dt" function that will be used by scipy to solve the system
        # TODO it could be better to put this at the same level at the other class methods, but
        # it only needs to be used here, and it should not be accessible from the outside...
        def dX_dt(t_local, X_local, par) :
            """
            Function that is used to solve the differential equation 
            """
            equations, symbols = par

            # create dictionary symbol -> value (order of symbols is important!)
            symbols_to_values = {s : X_local[i] for i, s in enumerate(symbols)}
            symbols_to_values['t'] = t_local

            # compute values, evaluating the symbolic functions after replacement
            values = [ eq.evalf(subs=symbols_to_values) for var, eq in equations.items() ]

            return values

        # setup; Y is the current state of all state variables; they appear in the order specified by self.equations
        # in the latest version of Python, dictionaries are guaranteed to always return keys in the same order
        Y = []
        time = []

        Y.append( [initial_conditions[state_variable] for state_variable in self.equations] )
        time.append(0)

        # we set up the integrator; we also need to prepare the variables that will be used by dX_dt
        symbols = [ state_variable for state_variable in local_equations ]
        r = integrate.ode(dX_dt)
        r.set_f_params([local_equations, symbols])
        r.set_integrator('dopri5')
        r.set_initial_value(Y[0], time[0])

        # prepare the output variables
        output_values = { str(variable) : [] for variable in local_equations }
        output_values["time"] = []
        constraint_violations = [] # when, what, by how much

        # start the loop, integrating for each time step and checking if the constraints are respected
        index = 1
        all_constraints_satisfied = True
        while r.successful() and r.t < max_time and all_constraints_satisfied :

            # get the next point in the solution of the ODE system
            r.integrate(r.t + time_step, step=True)
            Y.append(r.y)
            time.append(r.t)

            print("Time=%2.f, Values=%s" % (r.t, str(r.y)))

            # go from the values to the symbols
            current_values = { variable : r.y[i] for i, variable in enumerate(symbols) }

            # add checks on constraints
            for variable, constraint_list in local_constraints.items() :

                for constraint_equation in constraint_list :
                    
                    print(constraint_equation)
                    print(constraint_equation.subs(current_values))

                    # evaluate the constraint, replacing the values of the symbols; however,
                    # we also have to take into account that the constraint could have already been
                    # reduced to a single value (True or False)
                    constraint_satisfied = True

                    print("Check if it is Boolean:", isinstance(constraint_equation, sympy.logic.boolalg.Boolean))
                    print("Check if it is BooleanFalse:", isinstance(constraint_equation, sympy.logic.boolalg.BooleanFalse))
                    print("Check if it is BooleanTrue:", isinstance(constraint_equation, sympy.logic.boolalg.BooleanTrue))

                    if isinstance(constraint_equation, sympy.logic.boolalg.BooleanFalse) : 
                        constraint_satisfied = False
                    elif not isinstance(constraint_equation, sympy.logic.boolalg.BooleanTrue) :
                        constraint_satsified = constraint_equation.subs(current_values) # there is no need for .evalf() here, it should be reduced to True/False

                    if not constraint_satisfied :
                        print("Constraint \"%s\" was not satisfied!" % str(constraint_equation))
                        constraint_violations.append({"time" : r.t, "state_variables_values" : current_values, "constraint_violated" : str(constraint_equation)})
                        all_constraints_satisfied = False

            index += 1

        # return a dictionary of lists for all state variables and time
        # also, return the dictionary of constraint violations
        return output_values, constraint_violations

    def __str__(self) :
        """
        Returns class description as a string.
        """
        class_string = "\n"

        class_string += "Equations:\n"
        for state_variable, equation in self.equations.items() :
            class_string += "d" + state_variable + "/dt = " + str(equation) + "\n"

        class_string += "\n"

        if len(self.constraints) > 0 :
            class_string += "Constraints:\n"
            for variable, constraint_list in self.constraints.items() :
                for c in constraint_list :
                    class_string += variable + " : " + str(c) + "\n"
            class_string += "\n"

        if len(self.control) > 0 :
            class_string += "Control:\n"
            for variable, control_equation in self.control.items() :
                class_string += variable + " = " + str(control_equation)
                if control_equation == "" :
                    class_string += "?"
                class_string += "\n"
            class_string += "\n"

        if len(self.parameters) > 0 :
            class_string += "Parameters:\n"
            for parameter, value in self.parameters.items() :
                class_string += str(parameter) + " = " + str(value) + "\n"

        return class_string
        


if __name__ == "__main__" :
    """
    This part here is just to test the functions.
    """
    print("Testing the viability theory problem class using the \"Lake eutrophication\" benchmark.")

    # this is the famous "Lake eutrophication" example of viability theory
    # https://demo.vino.openmole.org/viabilityproblem/1/#kernel/1/
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
    print(vp)

    print("Setting control variable to a specific value.")
    vp.set_control({"u": 0.75})
    print(vp)

    initial_conditions = {"L" : 0.5, "P" : 0.5}
    print("Running simulation with some initial conditions:", initial_conditions)
    output_values, constraint_violations = vp.run_simulation(initial_conditions, 0.01, 100)

    if len(constraint_violations) > 0 :
        cv = constraint_violations[0]
        print("The simulation stopped for a constraint violation, at time %.2f, for values \"%s\", on constraint \"%s\"" % (cv["time"], str(cv["state_variables_values"]), cv["constraint_violated"]))
