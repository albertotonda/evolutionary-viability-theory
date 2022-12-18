"""
Set of scripts for managing viability theory problems. It's going to use symbolic computation.
"""
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

        return class_string
        


if __name__ == "__main__" :
    """
    This part here is just to test the functions.
    """

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

    vp = ViabilityTheoryProblem(equations=equations, control=control, constraints=constraints)
    print(vp)

