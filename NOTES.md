# Evolutionary Viability Theory - NOTES
Repository for the experiments on EAs applied to viability theory

## Ideas
1. Add the limits to the state variables as values to seed the initial population of gplearn. Other values (e.g. pi, e) could be also meaningful, depending on the problem.
2. Create hash for the string representation of individuals, to avoid useless (?) fitness evaluations. However, if the same individual is evaluated in different generations, it should have different fitness values.

## Lessons learned and design choices

### Known issues
1. Sometimes, processes just get stuck. I added timeouts and the situation somehow improved, but it id not 100% correct. There are too many weird things going on, like sub-processes writing on stderr and the Lock() on the logger pointer basically not working properly.
    i. When function scipy.optimize.odeint fails, since it's wrapped compiled C++, it writes directly on stderr. Even looking up redirections on stderr performed inside Python code, nothing seems to block that. Multiple processes writing at the same time on stderr can create terrible errors (and probably do).
    ii. We finally settled for multi-process evaluation, as multi-thread is not really necessary (not that much memory is used by each process). However, even with a Manager() object and related Lock(), it seems there is something weird going on when the processes try to write on the logger pointer. I tried to create a minimal case study, and I cannot reproduce the error (!!).
2. Function are\_individuals\_equal() tries to compare two individuals by converting them to two dictionaries of sympy expressions, and then to two strings. Apparently, there is something that does not work: in many cases, the function just returns the answer that the two individuals are equal.
3. It would be really nice to have some debugging logs from each process, but this does not work (see point 1). One way around that would be to use a Queue(), with several processes adding stuff to the Queue and one extra process that periodically removes stuff from the Queue and writes to log.
4. Code needs to be refactored and cleaned from all previous attempts at multi-something evaluation.

## Unorganized notes

### 2023-07-19
The replacer seems surprisingly to be ok (!!). The issue is likely in the genetic operators, somehow they also change the parent (??).

### 2023-07-18
There are some issues with the propagation of individuals in the population. I think that the replacer has some issues, maybe it's not using deepcopy?

### 2023-02-13
- Actually, for the idea of 2023-02-10, we can just set a flag in the run\_simulation() function, to stop or not stop if a constraint is violated.  

### 2023-02-10
- Remove all constraints, evaluate how long it takes to trace ten trajectories or so, with some very complicated control laws. Use this information to set up the likely timeout to stop processes.

### 2023-02-01
- Validate individuals by sampling the state space along P=0, for the lake?
- Use equations for sphere instead of cylinder

### 2023-01-22
1. Trying to implement the saturation of the control function actually creates issues with the scipy.integrate.odeint function. For some reason, computing the control function separately and then replacing its value generates results that are different, even when the values of the state variables are far from the boundaries (!). However, there is a possible solution, if we make a reasonable assumption on the constraints: if the constraints are in the form (min) <= control\_rule <= (max), we can change the control rule to max(min(control\_rule, max), min). It seems to work.

### 2023-01-19
1. Checking for individual equality using sympy's .equals() can create HUGE issues with complex individuals (as in, hours of evaluation for a single expression). Example: try comparing "sin((((sin(cos(sqrt((-0.8600))))-(log(((0.3135)\*P))\*(sqrt((-0.9027))+cos(P))))-((0.1684)+P))\*(L-P)))" to anything. Possible solution: convert to symbolic expression, then convert to string and directly compare the strings. Maybe less effective, but better than getting stuck.

## TODO
- Maybe, since gplearn's library is not very big, I could just use a local version, to avoid issues.
