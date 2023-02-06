"""
Script for the case study 3D-Sphere.
"""
import evolutionary_control_rules
import viability_theory
import sympy
import sys

# for this case study, we need to create a special class, because the function that generates random viable points
# starts from the assumption that it's possible to obtain minimum and maximum value of each state variable INDEPENDENTLY
# which is clearly not the case for the sphere function (x**2 + y**2 + z**2 <= r)
class ViabilityTheoryProblemSphere(viability_theory.ViabilityTheoryProblem) :
    """
    This class is the same as ViabilityTheoryProblem, except for the function that generates random viable points
    """

    def get_random_viable_point(self, random) :
        """
        Return random viable point for the sphere.
        """
        # first, find the only constraint
        constraint = None
        for state_variable in self.equations :
            if state_variable in self.constraints :
                constraint = self.constraints[state_variable][0]

        # then, replace all parameters in the constraint with their values
        constraint = constraint.subs(self.parameters)
        #print("Constraint to be satisfied: \"%s\"" % str(constraint))

        # and now, we draw random points until we get one that satisfies the constraint; the minimum and maximum are hard-coded because it's too hard to do otherwise
        point = None
        constraint_satisifed = False

        while constraint_satisifed == False :
            point = { state_variable : random.uniform(-1.23, 1.23) for state_variable in self.equations }
            # if, by replacing the values of the state variables in the constraint, the constraint is reduced to 'True' in sympy terms
            constraint_value = constraint.subs(point)
            #print("Point:", point)
            #print("Constraint value:", constraint_value)
            if isinstance(constraint_value, sympy.logic.boolalg.BooleanTrue) or constraint_value == True :
                constraint_satisifed = True

        #print("I got one viable point!")
        return point

if __name__ == "__main__" :

    # define viability problem through dictionaries
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
    vp = ViabilityTheoryProblemSphere(equations=equations, control=control, constraints=constraints, parameters=parameters)

    # start the evolution
    print("Evolving control rules for the following viability problem:", vp)
    evolutionary_control_rules.evolve_rules(viability_problem=vp, random_seed=44, saturate_control_function_on_boundaries=False)

    sys.exit(0)
