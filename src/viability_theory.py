"""
Set of scripts for managing viability theory problems. It's going to use symbolic computation.
"""
import copy
import logging # it could be useful to send messages to other processes
import numpy as np
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

    def saturate_control_rules(self, control_rules, state_variables_values, control_rules_constraints) :
        """
        Function that saturates the control rules, so that they can never escape the constraints. If a control
        rule exceeds its constraints, it is fixed at the constraint value. It returns a dictionary of values
        for the symbols corresponding to control variables.
        """
        control_variables_values = dict()

        # "state_variables_values" could be a dictionary like {string -> value}, let's change it to {symbol -> value}
        state_variables_values_symbols = { sympy.sympify(state_variable) : value for state_variable, value in state_variables_values.items() }

        for control_variable, control_rule in control_rules.items() :
            # compute a single value of the control rule, by just replacing the state variable symbols with their values
            control_variables_values[sympy.sympify(control_variable)] = control_rule.evalf(subs=state_variables_values)

        # we are making a wide hypothesis here, that constraints could include other control rules or state variables
        # we go over the constraints, and replace the value of everything until the constraint is either satisfied or
        # not satisfied; if it is not satisfied, we set the value of the control rule to the value of the constraint
        # (plus or minus an epsilon, if it is less than)
        for control_variable, constraint_list in control_rules_constraints.items() :
            for index_constraint in range(0, len(constraint_list)) :

                constraint_equation = control_rules_constraints[control_variable][index_constraint]

                # replace the symbols inside the constraint with the current values of the state variables and
                # the current values of the control variables, and check if the expression is False
                constraint_value = constraint_equation.subs(state_variables_values)
                constraint_value = constraint_value.subs(control_variables_values)

                if isinstance(constraint_value, sympy.logic.boolalg.BooleanFalse) :
                    print("Control rule constraint \"%s\" violated; (%s)" % (str(constraint_equation), str(control_variables_values)))

                    # the constraint has not been satisfied! replace the value of the control variable with the constraint value
                    # if the inequality includes "LessThan" (<=) or "GreaterThen" (>=), we can just set the control variable to
                    # the value in the right hand side; otherwise, we need to add or remove an epsilon, because we are dealing
                    # with a strict inequality (< or >)
                    control_variable_value = float(constraint_equation.rhs) # NOTE I am not sure this would work with complex constraints
                    control_variables_values[sympy.sympify(control_variable)] = control_variable_value 

        return control_variables_values


    def create_saturated_control_rules(self, control_rules, constraints) : 
        """
        Creates a "saturated" (min-maxed) version of each control rule.
        """
        saturated_control_rules = dict()

        for control_variable, control_rule in control_rules.items() :
            control_rule_string = str(control_rule)
            s_control_variable = sympy.sympify(control_variable)

            for constraint in constraints[s_control_variable] :
                
                # check the type of inequality; we turn the "strict" > and < into >= and <=, adding or subtracting a machine epsilon
                if isinstance(constraint, sympy.LessThan) :
                    control_rule_string = "Min(" + control_rule_string + ", " + str(constraint.rhs) + ")"
                elif isinstance(constraint, sympy.StrictLessThan) :
                    control_rule_string = "Min(" + control_rule_string + ", " + str(sympy.Add(constraint.rhs, sympy.sympify(np.finfo(float).eps))) + ")"
                elif isinstance(constraint, sympy.GreaterThan) :
                    control_rule_string = "Max(" + control_rule_string + ", " + str(constraint.rhs) + ")"
                elif isinstance(constraint, sympy.StrictGreaterThan) :
                    control_rule_string = "Max(" + control_rule_string + ", " + str(sympy.Add(constraint.rhs, sympy.sympify(-np.finfo(float).eps))) + ")"

            # after analyzing all constraints, here is the new (saturated) control rule
            saturated_control_rules[s_control_variable] = sympy.sympify(control_rule_string)
            #print("Saturated control rule:", control_rule_string)

        return saturated_control_rules


    def run_simulation(self, initial_conditions, time_step, max_time, saturate_control_function_on_boundaries=False) :
        """
        This part will actually solve the ODE system for a given set of initial conditions. Returns the values for each variable at each instant of time, and also the number and values of constraint violations.
        """
        # TODO to obtain some proper logging here, we should get the list of all loggers and use one
        # define some utility symbols
        inequality_symbols = [sympy.LessThan, sympy.GreaterThan, sympy.StrictLessThan, sympy.StrictGreaterThan]
        replace_inequalities_dictionary = { s : sympy.Add for s in inequality_symbols }

        # preliminary steps: replace all parameters in the equations for which we have values
        local_equations = { sympy.sympify(variable) : equation for variable, equation in self.equations.items() }
        local_parameters = { sympy.sympify(parameter) : value for parameter, value in self.parameters.items() }

        # also start replacing available parameters inside the constraints for the control rules 
        control_rules_constraints = dict()
        for variable, constraint_list in self.constraints.items() :
            if variable in self.control :
                s_variable = sympy.sympify(variable)
                control_rules_constraints[s_variable] = []
                for expression in constraint_list :
                    s_expression = sympy.sympify(expression)
                    s_expression = s_expression.subs(local_parameters)
                    control_rules_constraints[s_variable].append(s_expression)

        # if we ARE saturating the control rules, we first need to transform them into their saturated version
        local_control = { sympy.sympify(v) : sympy.sympify(eq) for v, eq in self.control.items() }
        if saturate_control_function_on_boundaries :
            local_control = self.create_saturated_control_rules(self.control, control_rules_constraints)

        for variable, control_equation in local_control.items() :
            local_parameters[sympy.sympify(variable)] = sympy.sympify(control_equation)

        for state_variable in local_equations :
            local_equations[state_variable] = local_equations[state_variable].subs(local_parameters) 

        # let's re-analyze all constraints, including the ones for the state variables 
        state_variables_constraints = dict()

        for variable, constraint_list in self.constraints.items() :
            # get a copy of the list of constraints, then replace the (possible) parameters with their values
            local_constraint_list = copy.deepcopy(constraint_list)
            for i in range(0, len(local_constraint_list)) :
                local_constraint_list[i] = constraint_list[i].subs(local_parameters)

            # depending on whether the variable is a control rule or a state variable, add list of constraints
            # to the corresponding dictionary
            if variable in self.equations :
                state_variables_constraints[variable] = local_constraint_list
            elif variable in self.control and not saturate_control_function_on_boundaries : # we update the control constraints only in the case we do not saturate
                control_rules_constraints[variable] = local_constraint_list

        print("Constraints on state variables:", state_variables_constraints)
        print("Constraints on control rules:", control_rules_constraints)

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
        # in the latest version of Python, dictionaries are guaranteed to always return keys in the same order they
        # have been added in, so there is no need to sort alphabetically or something like that
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
        for variable in self.control :
            output_values[str(variable)] = []
        output_values["time"] = []
        constraint_violations = [] # when, what, by how much

        # initialize output variables with known values
        for output_variable in output_values :
            if output_variable in initial_conditions :
                output_values[output_variable].append(initial_conditions[output_variable])
            elif sympy.sympify(output_variable) in local_control :
                output_values[output_variable].append(local_control[sympy.sympify(output_variable)].subs({sympy.sympify(k) : value for k, value in initial_conditions.items()}))

        # start the loop, integrating for each time step and checking if the constraints are respected
        logging.debug("Starting integration...")
        index = 1
        all_constraints_satisfied = True
        while r.successful() and r.t < max_time and all_constraints_satisfied :

            # debugging
            #string_debug = "Step %d:" % index
            #for variable in output_values :
            #    string_debug += " %s (%d values);" % (variable, len(output_values[variable]))
            #print(string_debug)

            # get the next point in the solution of the ODE system
            r.integrate(r.t + time_step, step=True)
            Y.append(r.y)
            time.append(r.t)

            logging.debug("\tTime=%2.f, Values=%s" % (r.t, str(r.y)))

            # go from the values to the symbols
            current_values = { variable : r.y[i] for i, variable in enumerate(symbols) }

            # add the current values to the history of the symbols' values
            for variable in current_values :
                output_values[str(variable)].append( current_values[variable] )

            # do the same for the control rules
            for variable, control_equation in local_control.items() :
                control_equation_value = control_equation.subs(local_parameters)
                control_equation_value = control_equation_value.subs(current_values)
                output_values[str(variable)].append( control_equation_value )

            # add checks on constraints; depending on whether we are saturating the control rules, we need
            # to define a special dictionary of constraints
            local_constraints = state_variables_constraints
            if not saturate_control_function_on_boundaries :
                local_constraints = {**state_variables_constraints, **control_rules_constraints}

            for variable, constraint_list in local_constraints.items() :
                for index_constraint in range(0, len(constraint_list)) :

                    constraint_equation = local_constraints[variable][index_constraint]

                    # evaluate the constraint, replacing the values of the symbols; however,
                    # we also have to take into account that the constraint could have already been
                    # reduced to a single value (True or False)
                    constraint_satisfied = True

                    if not isinstance(constraint_equation, sympy.logic.boolalg.BooleanTrue) :
                        constraint_equation = constraint_equation.subs(current_values) # there is no need for .evalf() here, it should be reduced to True/False
                        #print("Value of constraint_equation after .subs():", constraint_equation)

                    if isinstance(constraint_equation, sympy.logic.boolalg.BooleanFalse) : 
                        constraint_satisfied = False

                    if not constraint_satisfied :
                        # we also want to know by how much the constraint was violated; this might not be super-easy,
                        # but let's give it a try; we replace the inequality symbol with "-" and then evaluate the function
                        subtraction = local_constraints[variable][index_constraint]
                        # first, we need to replace the right hand side of the inequality with the same number but multiplied by -1
                        subtraction = subtraction.subs( { subtraction.rhs : sympy.Mul(sympy.sympify("-1"), subtraction.rhs) })
                        #print(subtraction)
                        for inequality_symbol in inequality_symbols :
                            subtraction = subtraction.replace(inequality_symbol, sympy.Add)
                            #print(subtraction)
                        amount_of_violation = subtraction.subs(current_values)
                        # and it's going to be in absolute value
                        amount_of_violation = abs(amount_of_violation)

                        logging.debug("Constraint \"%s\" was not satisfied!" % str(constraint_equation))
                        constraint_violations.append({  "time" : r.t, 
                                                        "state_variables_values" : current_values, 
                                                        "constraint_violated" : str(local_constraints[variable][index_constraint]),
                                                        "amount_of_violation" : amount_of_violation,
                                                        })
                        all_constraints_satisfied = False

            # end of the integration step
            index += 1

        # return a dictionary of lists for all state variables and time
        # also, return the dictionary of constraint violations
        output_values["time"] = time
        return output_values, constraint_violations

    def get_random_viable_point(self, random) :
        """
        Returns a point that is viable (e.g. does not violate constraints)
        """
        point = dict()

        # can we find min and max for each state variable from the constraints?
        # TODO replace this part with self.get_variable_boundaries()
        for state_variable in self.equations :

            minimum = np.inf
            maximum = -np.inf

            # go over the list of constraints for that variable
            for constraint in self.constraints[state_variable] :

                # create a new constraint, replacing the parameters
                constraint_with_value = constraint.subs(self.parameters)

                # check the type of inequality inside the constraint
                atoms = constraint_with_value.atoms()
                value = [ a for a in atoms if isinstance(a, sympy.Number) ][0].evalf() 

                if len([a.func for a in constraint.atoms(sympy.LessThan)]) > 0 :
                    maximum = value
                elif len([a.func for a in constraint.atoms(sympy.StrictLessThan)]) > 0 :
                    maximum = value - np.finfo(float).eps
                elif len([a.func for a in constraint.atoms(sympy.GreaterThan)]) > 0 :
                    minimum = value
                elif len([a.func for a in constraint.atoms(sympy.StrictGreaterThan)]) > 0 :
                    minimum = value + np.finfo(float).eps

            # here we now have reliable values for maximum and minimum
            # TODO maybe I could use a logger for these messages
            #print("For state variable \"%s\", minimum=%.4f maximum=%.4f" % (state_variable, minimum, maximum))

            # and finally, draw the random initial condition for that state variable
            point[state_variable] = random.uniform(minimum, maximum)

        return point

    def get_variable_boundaries(self, state_variable) :
        """
        Get (min, max) viable values for a variable.
        """
        minimum = np.inf
        maximum = -np.inf

        # go over the list of constraints for that variable
        for constraint in self.constraints[state_variable] :

            # create a new constraint, replacing the parameters
            constraint_with_value = constraint.subs(self.parameters)

            # check the type of inequality inside the constraint
            atoms = constraint_with_value.atoms()
            value = [ a for a in atoms if isinstance(a, sympy.Number) ][0].evalf() 

            if len([a.func for a in constraint.atoms(sympy.LessThan)]) > 0 :
                maximum = value
            elif len([a.func for a in constraint.atoms(sympy.StrictLessThan)]) > 0 :
                maximum = value - np.finfo(float).eps
            elif len([a.func for a in constraint.atoms(sympy.GreaterThan)]) > 0 :
                minimum = value
            elif len([a.func for a in constraint.atoms(sympy.StrictGreaterThan)]) > 0 :
                minimum = value + np.finfo(float).eps

        return minimum, maximum

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
    #vp.set_control({"u": 0.075})
    #vp.set_control({"u" : "L - P + 0.001"})
    vp.set_control({"u" : "sin(L) - cos(P)"})
    print(vp)

    initial_conditions = {"L" : 0.5, "P" : 0.5}
    print("Running simulation with some initial conditions:", initial_conditions)
    output_values, constraint_violations = vp.run_simulation(initial_conditions, 0.01, 100)

    if len(constraint_violations) > 0 :
        cv = constraint_violations[0]
        print("The simulation stopped for a constraint violation, at time %.2f, for values \"%s\", on constraint \"%s\" (by a value of %.4f)" % (cv["time"], str(cv["state_variables_values"]), cv["constraint_violated"], cv["amount_of_violation"]))
