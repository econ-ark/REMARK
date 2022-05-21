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

        params = init_agg_shocks_tax.copy()
        params.update(kwds)

        AggShockConsumerType.__init__(self, **params)

        self.add_to_time_inv("tax_rate")

        self.solve_one_period = solveConsAggShock
        self.update()

    # def get_shocks(self):
    #     """
    #     Same as in AggShockConsumerType, but with transitory income shocks decreased by the tax rate.
    #
    #     Parameters
    #     ----------
    #     None
    #
    #     Returns
    #     -------
    #     None
    #     """
    #     IndShockConsumerType.get_shocks(self)  # Update idiosyncratic shocks
    #     self.shocks['TranShk'] = (
    #         self.shocks['TranShk'] * self.TranShkAggNow * self.wRteNow
    #     )
    #     self.shocks['PermShk'] = self.shocks['PermShk'] * self.PermShkAggNow


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

# Make a dictionary to specify a Cobb-Douglas economy with income tax rates

init_cobb_douglas_tax = init_cobb_douglas.copy()
init_cobb_douglas_tax["tax_rate"] = 0.00

class CobbDouglasEconomy_tax(CobbDouglasEconomy):
    """
    A class to represent an economy with a Cobb-Douglas aggregate production
    function over labor and capital, extending HARK.Market.  The "aggregate
    market process" for this market combines all individuals' asset holdings
    into aggregate capital, yielding the interest factor on assets and the wage
    rate for the upcoming period.

    Note: The current implementation assumes a constant labor supply, but
    this will be generalized in the future.

    Parameters
    ----------
    tax_rate:
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
