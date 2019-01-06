The finite-horizon model is solved in job_search.py`.

Class methods for agent_series:

 1. solve_agent() - solves the consumption and job-search problem, generating the consumption functions and value functions. The consumption function is a list of lists of functions. It has T (number of periods) x (number of employment states per period) functions. The value function is similarly a list of lists of functions.

 2. compute_series() - Compute the agent behaviour for a given employment history and starting assets. Used to generate all the plots we see in the paper.



Parameters:
T_series - length of consumption/search behaviour to generate when calling compute_series()
T_solve - agent lifespan
e - employment history. Each integer indexes an employment state. 
z_vals - income for each employment state. In our code we use z_vals = [1, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.54], so this implies that:
	- Emp. state 1, inc = 1
	- Emp. states 2-7, inc= 0.83
	- Emp. state 8, inc = 0.54
   This corresponds to state 0 being the employed state, states 2-7 being the receiving-UI states, and state and 8 being the exhausted-benefits state. We have multiple states for receiving-UI, to model expectations of UI benefits ending. When simulating agent behaviour, in each unemployed state the agent either progresses to the next state, or finds a job and goes to state 0
a0 - starting assets
beta_var - delta, the geometric discount factor. This is unfortunately named.
beta_hyp - beta, the hyperbolic discount factor
a_size - grid size for solving the bellman equation.
rho - risk aversion
L_ - borrowing limit
constrained - True: the agent constrained by L_; False: the agent is constrained by the natural borrowing limit
Pi_ - state/job transition probability. See note 2 above. This is updated when solving the agent with endogenous search
R - interest rate
Rbor - interest rate when borrowing. I think this is deprecated.
phi - convexity of job search cost
k - constant term in job search cost
