# Evolutionary Viability Theory
Repository for the experiments on EAs applied to viability theory

## Ideas
1. Add the limits to the state variables as values to seed the initial population of gplearn.
2. Saturate the control functions, using the constraints as actual values, as in: if u >= umax, u = umax; if u <= umin, u = umin; else, u = (genetic programming individual)
3. Create hash for the string representation of individuals, to avoid useless (?) fitness evaluations. However, if the same individual is evaluated in different generations, it should have different fitness values.

## Lessons learned and design choices

### 2023-01-19
1. Checking for individual equality using sympy's .equals() can create HUGE issues with complex individuals (as in, hours of evaluation for a single expression). Example: try comparing "sin((((sin(cos(sqrt((-0.8600))))-(log(((0.3135)\*P))\*(sqrt((-0.9027))+cos(P))))-((0.1684)+P))\*(L-P)))" to anything. Possible solution: convert to symbolic expression, then convert to string and directly compare the strings. Maybe less effective, but better than getting stuck.

## TODO
Maybe, since gplearn's library is not very big, I could just use a local version, to avoid issues.
