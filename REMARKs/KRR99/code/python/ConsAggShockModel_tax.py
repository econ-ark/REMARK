"""
ConsAggShockModel, but with flat income tax rates and lump-sum transfers
"""


import numpy as np
import scipy.stats as stats
from HARK.interpolation import (
    LinearInterp,
    LinearInterpOnInterp1D,
    ConstantFunction,
    IdentityFunction,
    VariableLowerBoundFunc2D,
    BilinearInterp,
    LowerEnvelope2D,
    UpperEnvelope,
    MargValueFuncCRRA,
    ValueFuncCRRA
)
from HARK.utilities import (
    CRRAutility,
    CRRAutilityP,
    CRRAutilityPP,
    CRRAutilityP_inv,
    CRRAutility_invP,
    CRRAutility_inv,
    make_grid_exp_mult,
)
from HARK.distribution import (
    MarkovProcess,
    MeanOneLogNormal,
    Uniform,
    combine_indep_dstns,
    calc_expectation
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,
    init_idiosyncratic_shocks,
)
from HARK import MetricObject, Market, AgentType
from copy import deepcopy
import matplotlib.pyplot as plt


from HARK.ConsumptionSaving.ConsAggShockModel import (AggShockConsumerType,
    CobbDouglasEconomy,
    init_agg_shocks,
    init_cobb_douglas,
    solveConsAggShock,
    AggregateSavingRule
)


class ValueFunc2D(MetricObject):
    """
    A class for representing a value function in models (with CRRA utility).
    """

    distance_criteria = ["cFunc", "CRRA"]

    def __init__(self, cFunc, CRRA):
        """
        Constructor for a new value function object.
        Parameters
        ----------
        cFunc : function
            A real function representing the marginal value function composed
            with the inverse marginal utility function, defined on normalized individual market
            resources and aggregate market resources-to-labor ratio: uP_inv(vPfunc(m,M)).
            Called cFunc because when standard envelope condition applies,
            uP_inv(vPfunc(m,M)) = cFunc(m,M).
        CRRA : float
            Coefficient of relative risk aversion.
        Returns
        -------
        new instance of MargValueFunc
        """
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA

    def __call__(self, m, M):
        return utility(self.cFunc(m, M), gam=self.CRRA)
    

# Make a dictionary to specify an aggregate shocks consumer with taxes and transfers


init_agg_shocks_tax = init_agg_shocks.copy()
init_agg_shocks_tax["tax_rate"] = 0.0

class AggShockConsumerType_tax(AggShockConsumerType):
    """
    Same as AggShockConsumerType, but with taxes and transfers
    """
    def __init__(self, **kwds):
        """
        Makes a new instance of AggShockConsumerType_tax, an extension of
        AggShockConsumerType.  Sets appropriate solver and input lists.
        """

        self.vFunc = None

        params = init_agg_shocks_tax.copy()
        params.update(kwds)

        AggShockConsumerType.__init__(self, **params)

        self.add_to_time_inv("tax_rate")

        self.solve_one_period = solveConsAggShock_tax
        self.update()

    def transition(self):
        """"
        Same as in AggShockConsumerType, but with flat income tax and lump-sum transfers
        """

        pLvlPrev = self.state_prev['pLvl']
        aNrmPrev = self.state_prev['aNrm']
        RfreeNow = self.get_Rfree()

        # Calculate new states: normalized market resources and permanent income level
        pLvlNow = pLvlPrev*self.shocks['PermShk']  # Updated permanent income level
        # Updated aggregate permanent productivity level
        PlvlAggNow = self.state_prev['PlvlAgg']*self.PermShkAggNow
        # "Effective" interest factor on normalized assets
        ReffNow = RfreeNow/self.shocks['PermShk']

        # In AggShockConsumerType class:
        # mNrmNow = bNrmNow + self.shocks['TranShk']  # Market resources after income

        bNrmNow = (1 + (1-self.tax_rate)*(ReffNow-1)) * aNrmPrev
        # Bank balances before labor income and subtracted by taxed portion of capital income

        mNrmNow = bNrmNow + (1-self.tax_rate)*self.shocks['TranShk'] + self.calc_transfers()
        # Market resources, after (taxed) transitory income shock and lump-sum transfers

        return pLvlNow, PlvlAggNow, bNrmNow, mNrmNow, None

    def calc_transfers(self):
        """
        Returns level of (normalized) lump-sum transfers
        """
        aNrmPrev = self.state_prev['aNrm']
        RfreeNow = self.get_Rfree()
        ReffNow = RfreeNow / self.shocks['PermShk']

        # calculates lump-sum transfer by multiplying tax rate to to (capital + labor) income
        taxrevenue = np.mean(self.tax_rate * ((ReffNow - 1) * aNrmPrev + self.shocks['TranShk']))
        transfers = taxrevenue * (1 - 0.05)
        return transfers

    def update_solution_terminal(self):
        """
        Same as update_solution_terminal() in the AggShockConsumerType class, 
        with the addition of the value function vFunc

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cFunc_terminal = BilinearInterp(
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        )
        vPfunc_terminal = MargValueFuncCRRA(cFunc_terminal, self.CRRA)
        vFunc_terminal = ValueFuncCRRA(cFunc_terminal, self.CRRA)
        mNrmMin_terminal = ConstantFunction(0)
        self.solution_terminal = ConsumerSolution(
            cFunc=cFunc_terminal, vPfunc=vPfunc_terminal, vFunc=vFunc_terminal, mNrmMin=mNrmMin_terminal
        )



def solveConsAggShock_tax(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    PermGroFac,
    PermGroFacAgg,
    aXtraGrid,
    BoroCnstArt,
    Mgrid,
    AFunc,
    Rfunc,
    wFunc,
):
    """
    Same as solveConsAggShock in ConsAggShockModel.py,
    with the addition that it calculates a value function in the solution.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to the succeeding one period problem.
    IncShkDstn : distribution.Distribution
        A discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order: 
        idiosyncratic permanent shocks, idiosyncratic transitory
        shocks, aggregate permanent shocks, aggregate transitory shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    PermGroFacAgg : float
        Expected aggregate productivity growth factor.
    aXtraGrid : np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    BoroCnstArt : float
        Artificial borrowing constraint; minimum allowable end-of-period asset-to-
        permanent-income ratio.  Unlike other models, this *can't* be None.
    Mgrid : np.array
        A grid of aggregate market resourses to permanent income in the economy.
    AFunc : function
        Aggregate savings as a function of aggregate market resources.
    Rfunc : function
        The net interest factor on assets as a function of capital ratio k.
    wFunc : function
        The wage rate for labor as a function of capital-to-labor ratio k.
    DeprFac : float
        Capital Depreciation Rate

    Returns
    -------
    solution_now : ConsumerSolution
        The solution to the single period consumption-saving problem.  Includes
        a consumption function cFunc (linear interpolation over linear interpola-
        tions), a marginal value function vPfunc, and a value function vFunc.
    """

    # Unpack next period's solution
    vPfuncNext = solution_next.vPfunc
    vFuncNext = solution_next.vFunc
    mNrmMinNext = solution_next.mNrmMin

    # Unpack the income shocks
    ShkPrbsNext = IncShkDstn.pmf
    PermShkValsNext = IncShkDstn.X[0]
    TranShkValsNext = IncShkDstn.X[1]
    PermShkAggValsNext = IncShkDstn.X[2]
    TranShkAggValsNext = IncShkDstn.X[3]
    ShkCount = ShkPrbsNext.size

    # Make the grid of end-of-period asset values, and a tiled version
    aNrmNow = aXtraGrid
    aCount = aNrmNow.size
    Mcount = Mgrid.size
    aXtra_tiled = np.tile(np.reshape(aNrmNow, (1, aCount, 1)), (Mcount, 1, ShkCount))

    # Make tiled versions of the income shocks
    # Dimension order: Mnow, aNow, Shk
    ShkPrbsNext_tiled = np.tile(
        np.reshape(ShkPrbsNext, (1, 1, ShkCount)), (Mcount, aCount, 1)
    )
    PermShkValsNext_tiled = np.tile(
        np.reshape(PermShkValsNext, (1, 1, ShkCount)), (Mcount, aCount, 1)
    )
    TranShkValsNext_tiled = np.tile(
        np.reshape(TranShkValsNext, (1, 1, ShkCount)), (Mcount, aCount, 1)
    )
    PermShkAggValsNext_tiled = np.tile(
        np.reshape(PermShkAggValsNext, (1, 1, ShkCount)), (Mcount, aCount, 1)
    )
    TranShkAggValsNext_tiled = np.tile(
        np.reshape(TranShkAggValsNext, (1, 1, ShkCount)), (Mcount, aCount, 1)
    )

    # Calculate returns to capital and labor in the next period
    AaggNow_tiled = np.tile(
        np.reshape(AFunc(Mgrid), (Mcount, 1, 1)), (1, aCount, ShkCount)
    )
    kNext_array = AaggNow_tiled / (
        PermGroFacAgg * PermShkAggValsNext_tiled
    )  # Next period's aggregate capital/labor ratio
    kNextEff_array = (
        kNext_array / TranShkAggValsNext_tiled
    )  # Same thing, but account for *transitory* shock
    R_array = Rfunc(kNextEff_array)  # Interest factor on aggregate assets
    Reff_array = (
        R_array / LivPrb
    )  # Effective interest factor on individual assets *for survivors*
    wEff_array = (
        wFunc(kNextEff_array) * TranShkAggValsNext_tiled
    )  # Effective wage rate (accounts for labor supply)
    PermShkTotal_array = (
        PermGroFac * PermGroFacAgg * PermShkValsNext_tiled * PermShkAggValsNext_tiled
    )  # total / combined permanent shock

    Mnext_array = (
        kNext_array * R_array + wEff_array
    )  # next period's aggregate market resources

    # Find the natural borrowing constraint for each value of M in the Mgrid.
    # There is likely a faster way to do this, but someone needs to do the math:
    # is aNrmMin determined by getting the worst shock of all four types?
    aNrmMin_candidates = (
        PermGroFac
        * PermGroFacAgg
        * PermShkValsNext_tiled[:, 0, :]
        * PermShkAggValsNext_tiled[:, 0, :]
        / Reff_array[:, 0, :]
        * (
            mNrmMinNext(Mnext_array[:, 0, :])
            - wEff_array[:, 0, :] * TranShkValsNext_tiled[:, 0, :]
        )
    )
    aNrmMin_vec = np.max(aNrmMin_candidates, axis=1)
    BoroCnstNat_vec = aNrmMin_vec
    aNrmMin_tiled = np.tile(
        np.reshape(aNrmMin_vec, (Mcount, 1, 1)), (1, aCount, ShkCount)
    )
    aNrmNow_tiled = aNrmMin_tiled + aXtra_tiled

    # Calculate market resources next period (and a constant array of capital-to-labor ratio)
    mNrmNext_array = (
        Reff_array * aNrmNow_tiled / PermShkTotal_array
        + TranShkValsNext_tiled * wEff_array
    )

    # Find marginal value next period at every income shock realization and every aggregate market resource gridpoint
    vPnext_array = (
        Reff_array
        * PermShkTotal_array ** (-CRRA)
        * vPfuncNext(mNrmNext_array, Mnext_array)
    )

    # Calculate expectated marginal value at the end of the period at every asset gridpoint
    EndOfPrdvP = DiscFac * LivPrb * np.sum(vPnext_array * ShkPrbsNext_tiled, axis=2)

    # Calculate optimal consumption from each asset gridpoint
    cNrmNow = EndOfPrdvP ** (-1.0 / CRRA)
    mNrmNow = aNrmNow_tiled[:, :, 0] + cNrmNow

    # Loop through the values in Mgrid and make a linear consumption function for each
    cFuncBaseByM_list = []
    for j in range(Mcount):
        c_temp = np.insert(cNrmNow[j, :], 0, 0.0)  # Add point at bottom
        m_temp = np.insert(mNrmNow[j, :] - BoroCnstNat_vec[j], 0, 0.0)
        cFuncBaseByM_list.append(LinearInterp(m_temp, c_temp))
        # Add the M-specific consumption function to the list

    # Construct the overall unconstrained consumption function by combining the M-specific functions
    BoroCnstNat = LinearInterp(
        np.insert(Mgrid, 0, 0.0), np.insert(BoroCnstNat_vec, 0, 0.0)
    )
    cFuncBase = LinearInterpOnInterp1D(cFuncBaseByM_list, Mgrid)
    cFuncUnc = VariableLowerBoundFunc2D(cFuncBase, BoroCnstNat)

    # Make the constrained consumption function and combine it with the unconstrained component
    cFuncCnst = BilinearInterp(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([BoroCnstArt, BoroCnstArt + 1.0]),
        np.array([0.0, 1.0]),
    )
    cFuncNow = LowerEnvelope2D(cFuncUnc, cFuncCnst)

    # Make the minimum m function as the greater of the natural and artificial constraints
    mNrmMinNow = UpperEnvelope(BoroCnstNat, ConstantFunction(BoroCnstArt))

    # Construct the marginal value function using the envelope condition
    vPfuncNow = MargValueFuncCRRA(cFuncNow, CRRA)

    # Construct the marginal value function using the envelope condition
    vFuncNow = ValueFuncCRRA(cFuncNow, CRRA)

    # Pack up and return the solution
    solution_now = ConsumerSolution(
        cFunc=cFuncNow, vPfunc=vPfuncNow, vFunc=vFuncNow, mNrmMin=mNrmMinNow
    )
    return solution_now




# Make a dictionary to specify a Cobb-Douglas economy with income tax rates

init_cobb_douglas_tax = init_cobb_douglas.copy()
init_cobb_douglas_tax["tax_rate"] = 0.00

class CobbDouglasEconomy_tax(CobbDouglasEconomy):
    """
    Same as the CobbDouglasEconomy market class, with the addition
    of a flat income tax rate affecting the steady state level of capital
    per capita and hence aggregate output.
    
    Additional Parameters
    ----------
    tax_rate: float [0,1]
    """

    def __init__(self, agents=None, tolerance=0.0001, act_T=1200, **kwds):
        agents = agents if agents is not None else list()
        params = init_cobb_douglas.copy()
        params["sow_vars"] = [
            "MaggNow",
            "AaggNow",
            "RfreeNow",
            "wRteNow",
            "PermShkAggNow",
            "TranShkAggNow",
            "KtoLnow",
        ]
        params.update(kwds)

        Market.__init__(
            self,
            agents=agents,
            reap_vars=['aLvl', 'pLvl'],
            track_vars=["MaggNow", "AaggNow"],
            dyn_vars=["AFunc"],
            tolerance=tolerance,
            act_T=act_T,
            **params
        )
        self.update()

        # Use previously hardcoded values for AFunc updating if not passed
        # as part of initialization dictionary.  This is to prevent a last
        # minute update to HARK before a release from having a breaking change.
        if not hasattr(self, "tax_rate"):
            self.tax_rate = 0.0

    def update(self):
        """
        Use primitive parameters (and perfect foresight calibrations) to make
        interest factor and wage rate functions (of capital to labor ratio),
        as well as discrete approximations to the aggregate shock distributions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.kSS = (
            (
                self.get_PermGroFacAggLR() ** (self.CRRA) / self.DiscFac
                - (1.0 - self.DeprFac)
            ) / (1 - self.tax_rate)
            / self.CapShare
        ) ** (1.0 / (self.CapShare - 1.0))
        self.KtoYSS = self.kSS ** (1.0 - self.CapShare)
        self.wRteSS = (1.0 - self.CapShare) * self.kSS ** (self.CapShare)
        self.RfreeSS = (
            1.0 + self.CapShare * self.kSS ** (self.CapShare - 1.0) - self.DeprFac
        )
        self.MSS = self.kSS * self.RfreeSS + self.wRteSS
        self.convertKtoY = lambda KtoY: KtoY ** (
            1.0 / (1.0 - self.CapShare)
        )  # converts K/Y to K/L
        self.Rfunc = lambda k: (
            1.0 + self.CapShare * k ** (self.CapShare - 1.0) - self.DeprFac
        )
        self.wFunc = lambda k: ((1.0 - self.CapShare) * k ** (self.CapShare))

        self.sow_init["KtoLnow"] = self.kSS
        self.sow_init["MaggNow"] = self.kSS
        self.sow_init["AaggNow"] = self.kSS
        self.sow_init["RfreeNow"] = self.Rfunc(self.kSS)
        self.sow_init["wRteNow"] = self.wFunc(self.kSS)
        self.sow_init["PermShkAggNow"] = 1.0
        self.sow_init["TranShkAggNow"] = 1.0
        self.make_AggShkDstn()
        self.AFunc = AggregateSavingRule(self.intercept_prev, self.slope_prev)
