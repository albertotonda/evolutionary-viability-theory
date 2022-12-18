"""
Set of scripts for managing viability theory problems. It's going to use symbolic computation.
"""
import sympy

from scipy import integrate

class ViabilityTheoryProblem :
    """
    Class that describes a viability theory problem
    """

    def __init__(self, equations={}, constraints={}, control={}) :

        # we assume that all ODEs are dX/dt
        self.equations = {}
        for state_variable, equation in equations.items() :
            self.equations[state_variable] = sympy.sympify(equation)

    def __str__() :
        """
        Returns class description as a string.
        """
        class_string = "\n"

        for state_variable, equation in self.equations.items() :
            class_string += "d" + state_variable + "/dt = " + str(equation)

        return class_string
        


if __name__ == "__main__" :
    """
    This part here is just to test the functions.
    """

    # this is the famous "Lake" example of viability theory

