class McGariConsumerType(IndShockConsumerType):
    """
    An agent in the Markov consumption-saving model.  His problem is defined by a sequence
    of income distributions, survival probabilities, discount factors, and permanent
    income growth rates, as well as time invariant values for risk aversion, the
    interest rate, the grid of end-of-period assets, and how he is borrowing constrained.
    """

    time_vary_ = IndShockConsumerType.time_vary_ + ["MrkvArray"]
    time_inv_ = IndShockConsumerType.time_inv_  + ["eta",
                                                   "nu",
                                                   "pssi",
                                                   "B",
                                                   ]

    # Is "Mrkv" a shock or a state?
    shock_vars_ = IndShockConsumerType.shock_vars_ + ["Mrkv"]
    state_vars  = IndShockConsumerType.state_vars  + ["Mrkv"]

    def __init__(self, cycles=1, **kwds):
        IndShockConsumerType.__init__(self, cycles=1, **kwds)
        self.solve_one_period = _solve_ConsMarkov

        if not hasattr(self, "global_markov"):
            self.global_markov = False

    def check_markov_inputs(self):
        """
        Many parameters used by MarkovConsumerType are arrays.  Make sure those arrays are the
        right shape.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        StateCount = self.MrkvArray[0].shape[0]

        # Check that arrays are the right shape
        if not isinstance(self.Rfree, np.ndarray) or self.Rfree.shape != (StateCount,):
            raise ValueError(
                "Rfree not the right shape, it should an array of Rfree of all the states."
            )

        # Check that arrays in lists are the right shape
        for MrkvArray_t in self.MrkvArray:
            if not isinstance(MrkvArray_t, np.ndarray) or MrkvArray_t.shape != (
                StateCount,
                StateCount,
            ):
                raise ValueError(
                    "MrkvArray not the right shape, it should be of the size states*statres."
                )
        for LivPrb_t in self.LivPrb:
            if not isinstance(LivPrb_t, np.ndarray) or LivPrb_t.shape != (StateCount,):
                raise ValueError(
                    "Array in LivPrb is not the right shape, it should be an array of length equal to number of states"
                )
        for PermGroFac_t in self.PermGroFac:
            if not isinstance(PermGroFac_t, np.ndarray) or PermGroFac_t.shape != (
                StateCount,
            ):
                raise ValueError(
                    "Array in PermGroFac is not the right shape, it should be an array of length equal to number of states"
                )

        # Now check the income distribution.
        # Note IncShkDstn is (potentially) time-varying, so it is in time_vary.
        # Therefore it is a list, and each element of that list responds to the income distribution
        # at a particular point in time.  Each income distribution at a point in time should itself
        # be a list, with each element corresponding to the income distribution
        # conditional on a particular Markov state.
        # TODO: should this be a numpy array too?
        for IncShkDstn_t in self.IncShkDstn:
            if not isinstance(IncShkDstn_t, list):
                raise ValueError(
                    "self.IncShkDstn is time varying and so must be a list"
                    + "of lists of Distributions, one per Markov State. Found "
                    + f"{self.IncShkDstn} instead"
                )
            elif len(IncShkDstn_t) != StateCount:
                raise ValueError(
                    "List in IncShkDstn is not the right length, it should be length equal to number of states"
                )

    def pre_solve(self):
        """
        Check to make sure that the inputs that are specific to MarkovConsumerType
        are of the right shape (if arrays) or length (if lists).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        AgentType.pre_solve(self)
        self.check_markov_inputs()

    def update_solution_terminal(self): # Requires Extensive Modification
        """
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        IndShockConsumerType.update_solution_terminal(self)

        # Make replicated terminal period solution: consume all resources, no human wealth, minimum m is 0
        StateCount = self.MrkvArray[0].shape[0]
        self.solution_terminal.cFunc = StateCount * [self.cFunc_terminal_]
        self.solution_terminal.vFunc = StateCount * [self.solution_terminal.vFunc]
        self.solution_terminal.vPfunc = StateCount * [self.solution_terminal.vPfunc]
        self.solution_terminal.vPPfunc = StateCount * [self.solution_terminal.vPPfunc]
        self.solution_terminal.mNrmMin = np.zeros(StateCount)
        self.solution_terminal.hRto = np.zeros(StateCount)
        self.solution_terminal.MPCmax = np.ones(StateCount)
        self.solution_terminal.MPCmin = np.ones(StateCount)

    def initialize_sim(self):
        self.shocks["Mrkv"] = np.zeros(self.AgentCount, dtype=int)
        IndShockConsumerType.initialize_sim(self)
        if (
            self.global_markov
        ):  # Need to initialize markov state to be the same for all agents
            base_draw = Uniform(seed=self.RNG.randint(0, 2 ** 31 - 1)).draw(1)
            Cutoffs = np.cumsum(np.array(self.MrkvPrbsInit))
            self.shocks["Mrkv"] = np.ones(self.AgentCount) * np.searchsorted(
                Cutoffs, base_draw
            ).astype(int)
        self.shocks["Mrkv"] = self.shocks["Mrkv"].astype(int)

    def reset_rng(self):
        """
        Extended method that ensures random shocks are drawn from the same sequence
        on each simulation, which is important for structural estimation.  This
        method is called automatically by initialize_sim().

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        PerfForesightConsumerType.reset_rng(self)

        # Reset IncShkDstn if it exists (it might not because reset_rng is called at init)
        if hasattr(self, "IncShkDstn"):
            T = len(self.IncShkDstn)
            for t in range(T):
                for dstn in self.IncShkDstn[t]:
                    dstn.reset()

    def sim_death(self):
        """
        Determines which agents die this period and must be replaced.  Uses the sequence in LivPrb
        to determine survival probabilities for each agent.

        Parameters
        ----------
        None

        Returns
        -------
        which_agents : np.array(bool)
            Boolean array of size AgentCount indicating which agents die.
        """
        # Determine who dies
        LivPrb = np.array(self.LivPrb)[
            self.t_cycle - 1, self.shocks["Mrkv"]
        ]  # Time has already advanced, so look back one
        DiePrb = 1.0 - LivPrb
        DeathShks = Uniform(seed=self.RNG.randint(0, 2 ** 31 - 1)).draw(
            N=self.AgentCount
        )
        which_agents = DeathShks < DiePrb
        if self.T_age is not None:  # Kill agents that have lived for too many periods
            too_old = self.t_age >= self.T_age
            which_agents = np.logical_or(which_agents, too_old)
        return which_agents

    def sim_birth(self, which_agents):
        """
        Makes new Markov consumer by drawing initial normalized assets, permanent income levels, and
        discrete states. Calls IndShockConsumerType.sim_birth, then draws from initial Markov distribution.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        IndShockConsumerType.sim_birth(
            self, which_agents
        )  # Get initial assets and permanent income
        if (
            not self.global_markov
        ):  # Markov state is not changed if it is set at the global level
            N = np.sum(which_agents)
            base_draws = Uniform(seed=self.RNG.randint(0, 2 ** 31 - 1)).draw(N)
            Cutoffs = np.cumsum(np.array(self.MrkvPrbsInit))
            self.shocks["Mrkv"][which_agents] = np.searchsorted(
                Cutoffs, base_draws
            ).astype(int)

    def get_markov_states(self):
        """
        Draw new Markov states for each agent in the simulated population, using
        the attribute MrkvArray to determine transition probabilities.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        dont_change = (
            self.t_age == 0
        )  # Don't change Markov state for those who were just born (unless global_markov)
        if self.t_sim == 0:  # Respect initial distribution of Markov states
            dont_change[:] = True

        # Determine which agents are in which states right now
        J = self.MrkvArray[0].shape[0]
        MrkvPrev = self.shocks["Mrkv"]
        MrkvNow = np.zeros(self.AgentCount, dtype=int)

        # Draw new Markov states for each agent
        for t in range(self.T_cycle):
            markov_process = MarkovProcess(
                self.MrkvArray[t],
                seed=self.RNG.randint(0, 2 ** 31 - 1)
                )
            right_age = self.t_cycle == t
            MrkvNow[right_age] = markov_process.draw(MrkvPrev[right_age])
        if not self.global_markov:
            MrkvNow[dont_change] = MrkvPrev[dont_change]

        self.shocks["Mrkv"] = MrkvNow.astype(int)

    def get_shocks(self):
        """
        Gets new Markov states and permanent and transitory income shocks for this period.  Samples
        from IncShkDstn for each period-state in the cycle.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.get_markov_states()
        MrkvNow = self.shocks['Mrkv']

        # Now get income shocks for each consumer, by cycle-time and discrete state
        PermShkNow = np.zeros(self.AgentCount)  # Initialize shock arrays
        TranShkNow = np.zeros(self.AgentCount)
        for t in range(self.T_cycle):
            for j in range(self.MrkvArray[t].shape[0]):
                these = np.logical_and(t == self.t_cycle, j == MrkvNow)
                N = np.sum(these)
                if N > 0:
                    IncShkDstnNow = self.IncShkDstn[t - 1][
                        j
                    ]  # set current income distribution
                    PermGroFacNow = self.PermGroFac[t - 1][
                        j
                    ]  # and permanent growth factor

                    # Get random draws of income shocks from the discrete distribution
                    EventDraws = IncShkDstnNow.draw_events(N)
                    PermShkNow[these] = (
                        IncShkDstnNow.X[0][EventDraws] * PermGroFacNow
                    )  # permanent "shock" includes expected growth
                    TranShkNow[these] = IncShkDstnNow.X[1][EventDraws]
        newborn = self.t_age == 0
        PermShkNow[newborn] = 1.0
        TranShkNow[newborn] = 1.0
        self.shocks['PermShk'] = PermShkNow
        self.shocks['TranShk'] = TranShkNow

    def read_shocks_from_history(self):
        """
        A slight modification of AgentType.read_shocks that makes sure that MrkvNow is int, not float.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        IndShockConsumerType.read_shocks_from_history(self)
        self.shocks['Mrkv'] = self.shocks['Mrkv'].astype(int)

    def get_Rfree(self):
        """
        Returns an array of size self.AgentCount with interest factor that varies with discrete state.

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        """
        RfreeNow = self.Rfree[self.shocks['Mrkv']]
        return RfreeNow

    def get_controls(self):
        """
        Calculates consumption for each consumer of this type using the consumption functions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        J = self.MrkvArray[0].shape[0]

        MrkvBoolArray = np.zeros((J, self.AgentCount), dtype=bool)
        for j in range(J):
            MrkvBoolArray[j, :] = j == self.shocks['Mrkv']

        for t in range(self.T_cycle):
            right_t = t == self.t_cycle
            for j in range(J):
                these = np.logical_and(right_t, MrkvBoolArray[j, :])
                cNrmNow[these], MPCnow[these] = (
                    self.solution[t].cFunc[j].eval_with_derivative(self.state_now['mNrm'][these])
                )
        self.controls['cNrm'] = cNrmNow
        self.MPCnow = MPCnow

