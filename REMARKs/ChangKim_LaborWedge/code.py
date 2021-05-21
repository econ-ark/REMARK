#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 

@author: Wonsik Ko
"""

import sys

from copy import copy
import random as rm
import numpy as np
from HARK.core import HARKobject
from HARK.utilities import CRRAutilityP, CRRAutilityP_inv, NullFunc, plotFuncs, plotFuncsDer, make_figs
from HARK.distribution import DiscreteDistribution, combineIndepDstns, MeanOneLogNormal
from HARK.interpolation import (
    LinearInterp,
    LinearInterpOnInterp1D,
    VariableLowerBoundFunc2D,
    BilinearInterp,
    ConstantFunction,
    LowerEnvelope2D,
    UpperEnvelope,
    ConstantFunction
)
from HARK.ConsumptionSaving.ConsLaborModel import (
    ConsumerLaborSolution,
    LaborIntMargConsumerType,
)
from HARK.ConsumptionSaving.ConsAggShockModel import (
    AggShockConsumerType,
    AggShockMarkovConsumerType,
    AggregateSavingRule
)
from HARK.ConsumptionSaving.ConsGenIncProcessModel import ValueFunc2D

from HARK.ConsumptionSaving.ConsAggShockModel import MargValueFunc2D
import matplotlib.pyplot as plt
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution, PerfForesightConsumerType
)

#%%

### We created new one-period solution incoporating AggShockMarkovConsumerType and LaborIntMargConsumerType
class ConsumerLaborMarkovSolution(HARKobject):


    distance_criteria = ["vPfunc"]

    def __init__(
        self,
        cFunc=None,
        vFunc=None,
        vPfunc=None,
        vPPfunc=None,
        mNrmMin=None,
        bNrmMin=None,
        hNrm=None,
        MPCmin=None,
        MPCmax=None,
        Lbr=None,
        LbrFunc=None,
    ):
        """
        The constructor for a new ConsumerSolution object.
        Parameters
        ----------
        cFunc : function
            The consumption function for this period, defined over market
            resources: c = cFunc(b).
        LbrFunc : function
            The consumption function for this period, defined over market
            resources: l = LbrFunc(b).
        vFunc : function
            The beginning-of-period value function for this period, defined over
            market resources: v = vFunc(b).
        vPfunc : function
            The beginning-of-period marginal value function for this period,
            defined over market resources: vP = vPfunc(b).
        vPPfunc : function
            The beginning-of-period marginal marginal value function for this
            period, defined over market resources: vPP = vPPfunc(b).
        mNrmMin : float
            The minimum allowable market resources for this period; the consump-
            tion function (etc) are undefined for b < mNrmMin.
        hNrm : float
            Human wealth after receiving income this period: PDV of all future
            income, ignoring mortality.
        MPCmin : float
            Infimum of the marginal propensity to consume this period.
            MPC --> MPCmin as b --> infinity.
        MPCmax : float
            Supremum of the marginal propensity to consume this period.
            MPC --> MPCmax as b --> mNrmMin.
        Returns
        -------
        None
        """
        # Change any missing function inputs to NullFunc
        self.cFunc = cFunc if cFunc is not None else NullFunc()
        self.vFunc = vFunc if vFunc is not None else NullFunc()
        self.vPfunc = vPfunc if vPfunc is not None else NullFunc()
        # vPFunc = NullFunc() if vPfunc is None else vPfunc
        self.vPPfunc = vPPfunc if vPPfunc is not None else NullFunc()
        self.mNrmMin = mNrmMin
        self.bNrmMin = bNrmMin
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax
        self.Lbr = Lbr
        self.LbrFunc = LbrFunc
        
    def appendSolution(self, new_solution):
        """
        Appends one solution to another to create a ConsumerSolution whose
        attributes are lists.  Used in ConsMarkovModel, where we append solutions
        *conditional* on a particular value of a Markov state to each other in
        order to get the entire solution.
        Parameters
        ----------
        new_solution : ConsumerSolution
            The solution to a consumption-saving problem; each attribute is a
            list representing state-conditional values or functions.
        Returns
        -------
        None
        """
        if type(self.cFunc) != list:
            # Then we assume that self is an empty initialized solution instance.
            # Begin by checking this is so.
            assert (
                NullFunc().distance(self.cFunc) == 0
            ), "appendSolution called incorrectly!"

            # We will need the attributes of the solution instance to be lists.  Do that here.
            self.cFunc = [new_solution.cFunc]
            self.vFunc = [new_solution.vFunc]
            self.vPfunc = [new_solution.vPfunc]
            self.vPPfunc = [new_solution.vPPfunc]
            self.mNrmMin = [new_solution.mNrmMin]
            self.bNrmMin = [new_solution.bNrmMin]
            self.Lbr = [new_solution.Lbr]
            self.LbrFunc = [new_solution.LbrFunc]
        else:
            self.cFunc.append(new_solution.cFunc)
            self.vFunc.append(new_solution.vFunc)
            self.vPfunc.append(new_solution.vPfunc)
            self.vPPfunc.append(new_solution.vPPfunc)
            self.mNrmMin.append(new_solution.mNrmMin)
            self.bNrmMin.append(new_solution.bNrmMin)
            self.Lbr.append(new_solution.Lbr)
            self.LbrFunc.append(new_solution.LbrFunc)

def solveConsLaborAggMarkov(
    solution_next,
    IncomeDstn,
    LivPrb,
    LbrCost,
    DiscFac,
    CRRA,
    MrkvArray,
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
    Solve one period of a consumption-saving problem with idiosyncratic and
    aggregate shocks (transitory and permanent).  Moreover, the macroeconomic
    state follows a Markov process that determines the income distribution and
    aggregate permanent growth factor. This is a basic solver that can't handle
    cubic splines, nor can it calculate a value function.
    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to the succeeding one period problem.
    IncomeDstn : [[np.array]]
        A list of lists, each containing five arrays of floats, representing a
        discrete approximation to the income process between the period being
        solved and the one immediately following (in solution_next). Order: event
        probabilities, idisyncratic permanent shocks, idiosyncratic transitory
        shocks, aggregate permanent shocks, aggregate transitory shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    MrkvArray : np.array
        Markov transition matrix between discrete macroeconomic states.
        MrkvArray[i,j] is probability of being in state j next period conditional
        on being in state i this period.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period,
        for the *individual*'s productivity.
    PermGroFacAgg : [float]
        Expected aggregate productivity growth in each Markov macro state.
    aXtraGrid : np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    BoroCnstArt : float
        Artificial borrowing constraint; minimum allowable end-of-period asset-to-
        permanent-income ratio.  Unlike other models, this *can't* be None.
    Mgrid : np.array
        A grid of aggregate market resourses to permanent income in the economy.
    AFunc : [function]
        Aggregate savings as a function of aggregate market resources, for each
        Markov macro state.
    Rfunc : function
        The net interest factor on assets as a function of capital ratio k.
    wFunc : function
        The wage rate for labor as a function of capital-to-labor ratio k.
    Returns
    -------
    solution_now : ConsumerSolution
        The solution to the single period consumption-saving problem.  Includes
        a consumption function cFunc (linear interpolation over linear interpola-
        tions) and marginal value function vPfunc.
    """
    
    frac = 1.0 / (1.0 + LbrCost)
    if CRRA <= frac * LbrCost:
        print(
        "Error: make sure CRRA coefficient is strictly greater than alpha/(1+alpha)."
        )
        sys.exit()
    
    
    # Get sizes of grids
    aXtraCount = aXtraGrid.size
    bNrmGrid = aXtraGrid
    Mcount = Mgrid.size
    StateCount = MrkvArray.shape[0]

    EndOfPrdvPfunc_cond = []
    BoroCnstNat_cond = []
        
   
    for j in range(StateCount):
        # Unpack next period's solution
        vPfuncNext = solution_next.vPfunc[j] 
        bNrmMinNext = solution_next.bNrmMin[j]

        # Unpack the income shocks
        ShkPrbsNext = IncomeDstn[j].pmf
        PermShkValsNext = IncomeDstn[j].X[0]
        TranShkValsNext = IncomeDstn[j].X[1]
        PermShkAggValsNext = IncomeDstn[j].X[2]
        TranShkAggValsNext = IncomeDstn[j].X[3]
        ShkCount = ShkPrbsNext.size

                    
        uPinv = lambda X: CRRAutilityP_inv(X, gam=CRRA)

        # Replicated shock values for each b_t and M_t state
        
        aXtra_tiled = np.tile(
            np.reshape(aXtraGrid, (1, aXtraCount, 1)), (Mcount, 1, ShkCount) 
        )
    
        
        bNrmGrid_tiled = np.tile(
            np.reshape(bNrmGrid, (1, aXtraCount, 1)), (Mcount, 1, ShkCount) 
        )
        Mgrid_tiled = np.tile(
            np.reshape(Mgrid, (Mcount, 1, 1)), (1, aXtraCount, ShkCount) 
        )
    
        TranShkValsNext_tiled = np.tile(
            np.reshape(TranShkValsNext, (1, 1, ShkCount)), (Mcount, aXtraCount, 1)
        )
      
        ShkPrbsNext_tiled = np.tile(
            np.reshape(ShkPrbsNext, (1, 1, ShkCount)), (Mcount, aXtraCount, 1)
        )
        
        TranShkAggValsNext_tiled = np.tile(
            np.reshape(TranShkAggValsNext, (1, 1, ShkCount)), (Mcount, aXtraCount, 1)
        )
        
        PermShkAggValsNext_tiled = np.tile(
            np.reshape(PermShkAggValsNext, (1, 1, ShkCount)), (Mcount, aXtraCount, 1)
        )
        
        # Construct a function that gives marginal value of next period's bank balances *just before* the transitory shock arrives
        # Next period's marginal value at every transitory shock and every bank balances gridpoint
        vPNext = vPfuncNext(bNrmGrid_tiled,Mgrid_tiled)

        # Integrate out the transitory shocks (in TranShkVals direction) to get expected vP just before the transitory shock
        vPbarNext = np.sum(vPNext * ShkPrbsNext_tiled, axis=2)

    
        # Transformed marginal value through the inverse marginal utility function to "decurve" it
        vPbarNvrsNext = uPinv(vPbarNext)
        vPbarNvrsNext_temp = np.insert(vPbarNvrsNext, 0, 0.0, axis=0)
        vPbarNvrsNext_temp2 = np.insert(vPbarNvrsNext_temp, 0, 0.0, axis=1)
        # Linear interpolation over b_{t+1}, adding a point at minimal value of b = 0.
        vPbarNvrsFuncNext = BilinearInterp(
            np.transpose(vPbarNvrsNext_temp2), np.insert(bNrmGrid, 0, 0.0), np.insert(Mgrid, 0, 0.0)  
        )

        # "Recurve" the intermediate marginal value function through the marginal utility function
        vPbarFuncNext = MargValueFunc2D(vPbarNvrsFuncNext, CRRA)
        
        # Get next period's bank balances at each permanent shock from each end-of-period asset values
        # Replicated grid of a_t values for each permanent (productivity) shock
        aNrmGrid_tiled = np.tile(np.reshape(aXtraGrid, (1, aXtraCount, 1)), (Mcount, 1, ShkCount))    
        
        # Replicated permanent shock values for each b_t and M_t value
        PermShkValsNext_tiled = np.tile(
            np.reshape(PermShkValsNext, (1,1, ShkCount)), (Mcount, aXtraCount, 1)
        )
    
        # Replicated permanent shock probabilities for each b_t and M_t value
        PermShkPrbsNext_tiled = np.tile(
            np.reshape(ShkPrbsNext, (1, 1, ShkCount)), (Mcount, aXtraCount, 1)
        )
    
        # Make a tiled grid of end-of-period aggregate assets.  These lines use
        # next prd state j's aggregate saving rule to get a relevant set of Aagg,
        # which will be used to make an interpolated EndOfPrdvP_cond function.
        # After constructing these functions, we will use the aggregate saving
        # rule for *current* state i to get values of Aagg at which to evaluate
        # these conditional marginal value functions.  In the strange, maybe even
        # impossible case where the aggregate saving rules differ wildly across
        # macro states *and* there is "anti-persistence", so that the macro state
        # is very likely to change each period, then this procedure will lead to
        # an inaccurate solution because the grid of Aagg values on which the
        # conditional marginal value functions are constructed is not relevant
        # to the values at which it will actually be evaluated.
        
        AaggGrid = AFunc[j](Mgrid)
        AaggNow_tiled = np.tile(
            np.reshape(AaggGrid, (Mcount, 1, 1)), (1, aXtraCount, ShkCount)
        )
        kNowEff_array = AaggNow_tiled[:,:,j]
        ReffNow_array = Rfunc(AaggNow_tiled[:,:,j])
        wEffNow_array = wFunc(AaggNow_tiled[:,:,j])
        # Calculate returns to capital and labor in the next period
        kNext_array = AaggNow_tiled / (
            PermGroFacAgg[j] * PermShkAggValsNext_tiled
        )  # Next period's aggregate capital to labor ratio
        kNextEff_array = (
            kNext_array / TranShkAggValsNext_tiled
        )  # Same thing, but account for *transitory* shock
        R_array = Rfunc(kNextEff_array)  # Interest factor on aggregate assets
        Reff_array = (
            R_array / LivPrb
        )  # Effective interest factor on individual assets *for survivors*
        wEff_array = (
            wFunc(kNextEff_array) * TranShkAggValsNext_tiled * TranShkValsNext_tiled
        )  # Effective wage rate (accounts for labor supply) # aggregate wage doesn't matter for replication.
        PermShkTotal_array = (
            PermGroFac
            * PermGroFacAgg[j]
            * PermShkValsNext_tiled
            * PermShkAggValsNext_tiled
        )  # total / combined permanent shock
        
        Mnext_array = (
            kNext_array * R_array + wEff_array
        )

        # Find the natural borrowing constraint for each value of M in the Mgrid.
        # There is likely a faster way to do this, but someone needs to do the math:
        # is aNrmMin determined by getting the worst shock of all four types?
        aNrmMin_candidates = (
            PermGroFac
            * PermGroFacAgg[j]
            * PermShkValsNext_tiled[:, 0, :]
            * PermShkAggValsNext_tiled[:, 0, :]
            / Reff_array[:, 0, :]
            * (
                bNrmMinNext(Mnext_array[:, 0, :])
            )
        )
        aNrmMin_vec = np.max(aNrmMin_candidates, axis=1)

        BoroCnstNat_vec = aNrmMin_vec
        aNrmMin_tiled = np.tile(
            np.reshape(aNrmMin_vec, (Mcount, 1, 1)), (1, aXtraCount, ShkCount)
        )
    
        aNrmNow_tiled = aNrmMin_tiled + aXtra_tiled
    
        
        bNrmNext = (Reff_array / (PermGroFac * PermShkValsNext_tiled)) * aNrmGrid_tiled
    
        # Calculate marginal value of end-of-period assets at each b_t gridpoint
        # Get marginal value of bank balances next period at each shock
        vPbarNext = (
            Reff_array 
            * (PermGroFac * PermShkValsNext_tiled) ** (-CRRA) 
            * vPbarFuncNext(bNrmNext,Mnext_array)
        )

        # Take expectation across permanent income shocks
        EndOfPrdvP = (
            DiscFac
            * LivPrb
            * np.sum(vPbarNext * PermShkPrbsNext_tiled, axis=2)
        )
        # Make the conditional end-of-period marginal value function
        BoroCnstNat = LinearInterp(
            np.insert(AaggGrid, 0, 0.0), np.insert(BoroCnstNat_vec, 0, 0.0)
        )

        EndOfPrdvPnvrs = np.concatenate(
            (np.zeros((Mcount, 1)), EndOfPrdvP ** (-1.0 / CRRA)), axis=1
        )

        EndOfPrdvPnvrsFunc_base = BilinearInterp(
            np.transpose(EndOfPrdvPnvrs), np.insert(aXtraGrid, 0, 0.0), AaggGrid
        )

        EndOfPrdvPnvrsFunc = VariableLowerBoundFunc2D(
            EndOfPrdvPnvrsFunc_base, BoroCnstNat
        )

        EndOfPrdvPfunc_cond.append(MargValueFunc2D(EndOfPrdvPnvrsFunc, CRRA))
        BoroCnstNat_cond.append(BoroCnstNat)
    # Prepare some objects that are the same across all current states
    aXtra_tiled = np.tile(np.reshape(aXtraGrid, (1, aXtraCount)), (Mcount, 1))
    cFuncCnst = BilinearInterp(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([BoroCnstArt, BoroCnstArt + 1.0]),
        np.array([0.0, 1.0]),
    )    
    # Now loop through *this* period's discrete states, calculating end-of-period
    # marginal value (weighting across state transitions), then construct consumption
    # and marginal value function for each state.

    cFuncNow = []
    LbrFuncNow = []
    vPfuncNow = []
    bNrmMinNow = []   
    for i in range(StateCount):
        # Find natural borrowing constraint for this state by Aagg
        AaggNow = AFunc[i](Mgrid)
        aNrmMin_candidates = np.zeros((StateCount, Mcount)) + np.nan
        for j in range(StateCount):
            if MrkvArray[i, j] > 0.0:  # Irrelevant if transition is impossible
                aNrmMin_candidates[j, :] = BoroCnstNat_cond[j](AaggNow)

        aNrmMin_vec = np.nanmax(aNrmMin_candidates, axis=0)
        BoroCnstNat_vec = aNrmMin_vec

        # Make tiled grids of aNrm and Aagg
        aNrmMin_tiled = np.tile(np.reshape(aNrmMin_vec, (Mcount, 1)), (1, aXtraCount))
        aNrmNow_tiled = aNrmMin_tiled + aXtra_tiled
        AaggNow_tiled = np.tile(np.reshape(AaggNow, (Mcount, 1)), (1, aXtraCount))

        # Loop through feasible transitions and calculate end-of-period marginal value
        EndOfPrdvP_temp = np.zeros((Mcount, aXtraCount))
        for j in range(StateCount):
            if MrkvArray[i, j] > 0.0:
                temp = EndOfPrdvPfunc_cond[j](aNrmNow_tiled, AaggNow_tiled)
                EndOfPrdvP_temp += MrkvArray[i, j] * temp


        # Compute scaling factor for each transitory shock
        TranShkScaleFac_temp = (
           frac
           * (wEff_array[:,:,i]) ** (LbrCost * frac)
           * (LbrCost ** (-LbrCost * frac) + LbrCost ** frac)
        )

       
        EndOfPrdvP = np.multiply(EndOfPrdvP_temp, TranShkScaleFac_temp)

        for k in range(Mcount):
            for l in range(aXtraCount):
                if (EndOfPrdvP[k,l] < 0):
                    EndOfPrdvP[k,l] = 10 ** (-5)

        # Use the first order condition to compute an array of "composite good" x_t values corresponding to (a_t,theta_t) values
        xNow = EndOfPrdvP ** (-1.0 / (CRRA - LbrCost * frac))

        xNowPow = xNow ** frac  
        
        # Find optimal consumption from optimal composite good
        cNrmNow = (((wEffNow_array) / LbrCost) ** (LbrCost * frac)) * xNowPow
        
        # Consumption is infinity in a bad state (when transitory shock is 0) assign specific values to consumption
        # This code line probablly makes problem for consumption function!!
        for k in range(Mcount):
            for l in range(aXtraCount):
                if (cNrmNow[k,l] > 10**100):
                    cNrmNow[k,l] = bNrmGrid_tiled[k,l,i]

        LsrNow = (LbrCost / ( wEffNow_array)) ** frac * xNowPow
        # set leisure to be 1 (labor to be 0) in a bad state
        if i==0:
            for k in range(Mcount):
                for l in range(aXtraCount):
                    if (LsrNow[k,l] > 10^10):
                        LsrNow[k,l] = 1

        # Agent cannot choose to work a negative amount of time. When this occurs, set
        # leisure to one and recompute consumption using simplified first order condition.
        # Find where labor would be negative if unconstrained
        violates_labor_constraint = LsrNow > 1.0
        cNrmNow[violates_labor_constraint] = uPinv(
            EndOfPrdvP[violates_labor_constraint]
        )
        LsrNow[violates_labor_constraint] = 1.0  # Set up z=1, upper limit           
        
        # Calculate the endogenous bNrm states by inverting the within-period transition
        aNrmNow_rep = np.tile(np.reshape(aXtraGrid, (1, aXtraCount)),(Mcount,1))
        bNrmNow = (
            aNrmNow_rep
            - wEffNow_array * TranShkValsNext[i]
            + cNrmNow
            + wEffNow_array * TranShkValsNext[i] * LsrNow
        )
    
        LbrNow = 1.0 - LsrNow  # Labor is the complement of leisure

        
        # Get (pseudo-inverse) marginal value of bank balances using end of period
        # marginal value of assets (envelope condition), adding a column of zeros
        # zeros on the left edge, representing the limit at the minimum value of b_t.
        vPnvrsNowArray = np.concatenate(
            (np.zeros((Mcount ,1)), uPinv(EndOfPrdvP_temp)), axis=1
        )

        # Construct consumption and marginal value functions for this period
        bNrmMinNow_temp = LinearInterp(Mgrid, bNrmNow[:, 0], lower_extrap=False )

        # Now loop through *this* period's discrete states, calculating end-of-period
        # marginal value (weighting across state transitions), then construct consumption
        # and marginal value function for each state.
        cFuncBaseByM_list = []
        LbrFuncBaseByM_list = []
        for n in range(Mcount):
             c_temp = np.insert(cNrmNow[n, :], 0, 0.0)  # Add point at bottom
             l_temp = np.insert(LbrNow[n, :], 0, 0.0)  # Add point at bottom
             b_temp = np.insert(bNrmNow[n, :] , 0, 0.0)
             cFuncBaseByM_list.append(LinearInterp(b_temp, c_temp))
             LbrFuncBaseByM_list.append(LinearInterp(b_temp, l_temp))
            
        # Add the M-specific consumption function to the list
        # Construct the unconstrained consumption function by combining the M-specific functions
        BoroCnstNat = LinearInterp(
            np.insert(Mgrid, 0, 0.0), np.insert(BoroCnstNat_vec, 0, 0.0)
        )   

        cFuncBase = LinearInterpOnInterp1D(cFuncBaseByM_list, Mgrid)
        cFuncUnc = VariableLowerBoundFunc2D(cFuncBase, BoroCnstNat)
        
        cFuncUnc_temp = VariableLowerBoundFunc2D(cFuncBase, BoroCnstNat)

        bNrmMinNow.append(bNrmMinNow_temp)

        LbrFuncBase = LinearInterpOnInterp1D(LbrFuncBaseByM_list, Mgrid) # is it okay to do this?
        vPnvrsFuncNowBase = LinearInterpOnInterp1D(vPnvrsNowArray, Mgrid)
        cFuncNow_temp = LowerEnvelope2D(cFuncBase, cFuncCnst)
        # Combine the constrained consumption function with unconstrained component
        cFuncNow.append(cFuncNow_temp)
        LbrFuncNow.append(LbrFuncBase)
        vPfuncNow.append(MargValueFunc2D(cFuncNow_temp, CRRA))   

        
  # Make a solution object for this period and return it
    solution = ConsumerLaborMarkovSolution(
       cFunc=cFuncNow, LbrFunc=LbrFuncNow, vPfunc=vPfuncNow, bNrmMin=bNrmMinNow
    )
    return solution

#%%
### We created new consumer agent type incoporating AggShockMarkovConsumerType and LaborIntMargConsumerType
class LaborMarkovConsumerType(AggShockMarkovConsumerType):

    """
    A class representing agents who make a decision each period about how much
    to consume vs save and how much labor to supply (as a fraction of their time).
    They get CRRA utility from a composite good x_t = c_t*z_t^alpha, and discount
    future utility flows at a constant factor.
    """

    time_vary_ = copy(AggShockMarkovConsumerType.time_vary_)
    time_inv_ = copy(AggShockMarkovConsumerType.time_inv_)
    time_inv_ += ["StateCount"]
    
    def __init__(self, cycles=0, **kwds):
        """
        Instantiate a new consumer type with given data.
        See init_labor_intensive for a dictionary of
        the keywords that should be passed to the constructor.
        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.
        Returns
        -------
        None
        """
        
        params = init_labor_markov.copy()
        params.update(kwds)
        AggShockMarkovConsumerType.__init__(self, cycles=0, **kwds)
        self.solveOnePeriod = solveConsLaborAggMarkov
        self.__dict__.update(kwds) 


    def update(self):
        """
        Update the income process, the assets grid, and the terminal solution.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        
        self.updateLbrCost()
        self.updateIncomeProcess()
        self.updateAssetsGrid()


    def updateLbrCost(self):
        """
        Construct the age-varying cost of working LbrCost using the attribute LbrCostCoeffs.
        This attribute should be a 1D array of arbitrary length, representing polynomial
        coefficients (over t_cycle) for (log) LbrCost.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        Coeffs = self.LbrCostCoeffs
        N = len(Coeffs)
        age_vec = np.arange(self.T_cycle)
        LbrCostBase = np.zeros(self.T_cycle)
        for n in range(N):
            LbrCostBase += Coeffs[n] * age_vec ** n
        LbrCost = np.exp(LbrCostBase)
        self.LbrCost = LbrCost.tolist()
        self.addToTimeVary("LbrCost")
        
        
    def updateSolutionTerminal(self):
        """
        Updates the terminal period solution and solves for optimal consumption
        and labor when there is no future.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        ShkCount = 2
        aXtraCount = self.aXtraGrid.size
        Mcount = self.MgridBase.size
        AFunc_all = []
        for i in range(self.StateCount):
            AFunc_all.append(
                AggregateSavingRule(self.intercept_prev, self.slope_prev)
            )
        self.AFunc = AFunc_all
        
        self.Rfunc = lambda k: (
            1.0 + self.CapShare * k ** (self.CapShare - 1.0) - self.DeprFac
        )
        self.wFunc = lambda k: ((1.0 - self.CapShare) * k ** (self.CapShare))
        t = -1
        
        LbrFunc_terminal = []
        cFunc_terminal = []
        vPfunc_terminal = []
        bNrmMin_terminal = []

        for j in range(self.StateCount):
            TranShkGrid = self.IncomeDstn[0][j].X[1]
            LbrCost = self.LbrCost[t]
            AaggGrid = self.AFunc[j](self.MgridBase)
            AaggNow_tiled = np.tile(
                np.reshape(AaggGrid, (Mcount, 1, 1)), (1, aXtraCount, ShkCount)
            )
            kNowEff_array = AaggNow_tiled[:,:,j]
            ReffNow_array = self.Rfunc(AaggNow_tiled[:,:,j])
            wEffNow_array = self.wFunc(AaggNow_tiled[:,:,j])
            WageRte = wEffNow_array
            
            bNrmGrid = self.aXtraGrid
            bNrmCount = bNrmGrid.size  # 201
            
            TranShkCount = TranShkGrid.size  # = (7,)
            bNrmGridTerm = np.tile(
                np.reshape(bNrmGrid, (1, bNrmCount)), (Mcount, 1)
            )  # Replicated bNrmGrid for each transitory shock theta_t
    
            # Array of labor (leisure) values for terminal solution
            LsrTerm = np.minimum(
                (LbrCost / (1.0 + LbrCost))
                * (bNrmGridTerm / (WageRte * TranShkGrid[j]) + 1.0),
                1.0,
            )
            if i==0:
                LsrTerm[:, 0] = 1.0
            LbrTerm = 1.0 - LsrTerm
            
            # Calculate market resources in terminal period, which is consumption
            mNrmTerm = bNrmGridTerm + LbrTerm * WageRte * TranShkGrid[j]
            cNrmTerm = mNrmTerm  # Consume everything we have
            

            
            # Make a bilinear interpolation to represent the labor and consumption functions
            LbrFunc_terminal.append(BilinearInterp(LbrTerm, self.MgridBase , bNrmGrid))
            cFunc_terminal.append(BilinearInterp(cNrmTerm, self.MgridBase, bNrmGrid))
            
            # Compute the effective consumption value using consumption value and labor value at the terminal solution
            xEffTerm = LsrTerm ** LbrCost * cNrmTerm
            vNvrsFunc_terminal = BilinearInterp(xEffTerm, self.MgridBase, bNrmGrid)
            vFunc_terminal = ValueFunc2D(vNvrsFunc_terminal, self.CRRA)
    
            # Using the envelope condition at the terminal solution to estimate the marginal value function
            vPterm = LsrTerm ** LbrCost * CRRAutilityP(xEffTerm, gam=self.CRRA)
            vPnvrsTerm = CRRAutilityP_inv(
                vPterm, gam=self.CRRA
            )  # Evaluate the inverse of the CRRA marginal utility function at a given marginal value, vP
    
            vPnvrsFunc_terminal = BilinearInterp(vPnvrsTerm, self.MgridBase, bNrmGrid)

            vPfunc_terminal.append(MargValueFunc2D(vPnvrsFunc_terminal, self.CRRA))
                       
            
            # Get the Marginal Value function

            bNrmMin_terminal.append(
                ConstantFunction(
                    0.0
                )
            )
            # Trivial function that return the same real output for any input
        self.solution_terminal = ConsumerLaborMarkovSolution(
            cFunc=cFunc_terminal,
            LbrFunc=LbrFunc_terminal,
            vFunc=vFunc_terminal,
            vPfunc=vPfunc_terminal,
            bNrmMin=bNrmMin_terminal,
        )
    

        
    def getControls(self):
        """
        Calculates consumption for each consumer of this type using the consumption functions.
        For this AgentType class, MrkvNow is the same for all consumers.  However, in an
        extension with "macroeconomic inattention", consumers might misperceive the state
        and thus act as if they are in different states.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        LbrNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        MaggNow = self.getMaggNow()
        MrkvNow = self.getMrkvNow()

        StateCount = self.MrkvArray.shape[0]
        MrkvBoolArray = np.zeros((StateCount, self.AgentCount), dtype=bool)
        for i in range(StateCount):
            MrkvBoolArray[i, :] = i == MrkvNow

        for t in range(self.T_cycle):
            these = t == self.t_cycle
            for i in range(StateCount):
                those = np.logical_and(these, MrkvBoolArray[i, :])
                cNrmNow[those] = self.solution[t].cFunc[i](
                    self.state_now["bNrmNow"][those], MaggNow[those]
                )
                LbrNow[those] = self.solution[t].LbrFunc[i](
                    self.state_now["bNrmNow"][those], MaggNow[those]
                )
                # Marginal propensity to consume
                MPCnow[those] = (
                    self.solution[t]
                    .cFunc[i]
                    .derivativeX(self.state_now["bNrmNow"][those], MaggNow[those])
                )
        self.controls["cNrmNow"] = cNrmNow
        self.controls["LbrNow"] = LbrNow
        self.MPCnow = MPCnow
        return None

init_labor_markov = {
    "T_cycle": 1,
    "DiscFac": 0.99,
    "CRRA": 1.0,
    "LbrInd": 1.0,
    "aMin": 0.001,
    "aMax": 50.0,
    "aCount": 32,
    "aNestFac": 2,
    "MgridBase": np.array(
        [0.1, 0.3, 0.6, 0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.6, 2.0, 3.0]
    ),
    "AgentCount": 5000,
    "LbrCostCoeffs": -1.0,
    "StateCount": 2,
    "intercept_prev": 0.0,  # Intercept of aggregate savings function
    "slope_prev": 1.0  # Slope of aggregate savings function
}


#%%


RWAgentDictionary = { 
    "CapShare": 0.36,
    "DeprFac": 0.025,                        # Depreciation factor
    "intercept_prev": 0.0,                  # Intercept of aggregate savings function
    "slope_prev": 1.0,                       # Slope of aggregate savings function
    "StateCount": 2,    
    "LivPrb" : [1.0],                      # Survival probability
    "AgentCount" : 10000,                  # Number of agents of this type (only matters for simulation)
    "aNrmInitMean" : 0.0,                  # Mean of log initial assets (only matters for simulation)
    "aNrmInitStd"  : 0.0,                  # Standard deviation of log initial assets (only for simulation)
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income (only matters for simulation)
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income (only matters for simulation)
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor (only matters for simulation)
    "T_age" : None,                        # Age after which simulated agents are automatically killed
    "T_cycle" : 1,
    "LbrCostCoeffs": [-1.0],                 # Number of periods in the cycle for this agent type
# Parameters for constructing the "assets above minimum" grid
    "aXtraMin" : 0.001,                    # Minimum end-of-period "assets above minimum" value
    "aXtraMax" : 20,                       # Maximum end-of-period "assets above minimum" value               
    "aXtraExtra" : [None],                 # Some other value of "assets above minimum" to add to the grid
    "aXtraNestFac" : 3,                    # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraCount" : 24,                     # Number of points in the grid of "assets above minimum"
# Parameters describing the income process
    "PermShkCount" : 1,                    # Number of points in discrete approximation to permanent income shocks - no shocks of this kind!
    "TranShkCount" : 1,                    # Number of points in discrete approximation to transitory income shocks - no shocks of this kind!
    "PermShkStd" : [0.],                   # Standard deviation of log permanent income shocks - no shocks of this kind!
    "TranShkStd" : [0.],                   # Standard deviation of log transitory income shocks - no shocks of this kind!
    "UnempPrb" : 0.0,                      # Probability of unemployment while working - no shocks of this kind!
    "UnempPrbRet" : 0.00,                  # Probability of "unemployment" while retired - no shocks of this kind!
    "IncUnemp" : 0.0,                      # Unemployment benefits replacement rate
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "tax_rate" : 0.0,                      # Flat income tax rate
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "BoroCnstArt" : 0.0,                   # Artificial borrowing constraint; imposed minimum level of end-of period assets   
    "PermGroFac" : [1.0],                  # Permanent income growth factor
# New Parameters that we need now    
    'MgridBase': np.array([0.1,0.3,0.6,
                           0.8,0.9,0.98,
                           1.0,1.02,1.1,
                           1.2,1.6,2.0,
                           3.0]),          # Grid of capital-to-labor-ratios (factors) 
    'PermShkAggStd' : [0.0,0.0],           # Standard deviation of log aggregate permanent shocks by state. No continous shocks in a state.
    'TranShkAggStd' : [0.0,0.0],           # Standard deviation of log aggregate transitory shocks by state. No continuous shocks in a state.
    'PermGroFacAgg' : 1.0,

    "prb_eg": 0.96,         # Probability of   employment in the good state
    "prb_ug": 0.04,     # Probability of unemployment in the good state
    "prb_eb": 0.90,         # Probability of   employment in the bad state
    "prb_ub": 0.10,     # Probability of unemployment in the bad state
    "p_ind" : 1,            # Persistent component of income is always 1
    "ell_ug" : 0,
    "ell_ub" : 0,   # Labor supply is zero for unemployed consumers in either agg state
    "ell_eg" : 1.0/0.96,   # Labor supply for employed consumer in good state
    "ell_eb" : 1.0/0.90   

}

# Here we state just the "interesting" parts of the consumer's specification

RWAgentDictionary['CRRA']    = 1.0      # Relative risk aversion 
RWAgentDictionary['DiscFac'] = 0.99     # Intertemporal discount factor
RWAgentDictionary['cycles']  = 0        # cycles=0 means consumer is infinitely lived

# KS assume that 'good' and 'bad' times are of equal expected duration
# The probability of a change in the aggregate state is p_change=0.125
p_change=0.125
p_remain=1-p_change

# Now we define macro transition probabilities for AggShockMarkovConsumerType
#   [i,j] is probability of being in state j next period conditional on being in state i this period. 
# In both states, there is 0.875 chance of staying, 0.125 chance of switching
AggMrkvArray = \
np.array([[p_remain,p_change],  # Probabilities of states 0 and 1 next period if in state 0
          [p_change,p_remain]]) # Probabilities of states 0 and 1 next period if in state 1
RWAgentDictionary['MrkvArray'] = AggMrkvArray

# %%
# Create the Krusell-Smith agent as an instance of AggShockMarkovConsumerType 
RWAgent = LaborMarkovConsumerType(**RWAgentDictionary)



# %% code_folding=[]
# Construct the income distribution for the Krusell-Smith agent
prb_eg = 0.96         # Probability of   employment in the good state
prb_ug = 1-prb_eg     # Probability of unemployment in the good state
prb_eb = 0.90         # Probability of   employment in the bad state
prb_ub = 1-prb_eb     # Probability of unemployment in the bad state
p_ind  = 1            # Persistent component of income is always 1
ell_ug = ell_ub = 0   # Labor supply is zero for unemployed consumers in either agg state
ell_eg = 1.0/prb_eg   # Labor supply for employed consumer in good state
ell_eb = 1.0/prb_eb   # 1=pe_g*ell_ge+pu_b*ell_gu=pe_b*ell_be+pu_b*ell_gu

# IncomeDstn is a list of lists, one for each aggregate Markov state
# Each contains three arrays of floats, representing a discrete approximation to the income process. 
# Order: 
#   state probabilities 
#   idiosyncratic persistent income level by state (KS have no persistent shocks p_ind is always 1.0)
#   idiosyncratic transitory income level by state


RWAgent.IncomeDstn[0] = [
     DiscreteDistribution(np.array([prb_ug,prb_eg]), 
                          [np.array([p_ind,p_ind]),
                           np.array([ell_ug,ell_eg])]), # Agg state good
     DiscreteDistribution(np.array([prb_ub,prb_eb]),
                          [np.array([p_ind,p_ind]),
                           np.array([ell_ub,ell_eb])])  # Agg state bad
]


# %% [markdown]
# Up to this point, individual agents do not have enough information to solve their decision problem yet. What is missing are beliefs about the endogenous macro variables $r$ and $w$, both of which are functions of $\bar{k}$. 

# %% [markdown]
# #### The Aggregate Economy

# %% code_folding=[]
from HARK.ConsumptionSaving.ConsAggShockModel import CobbDouglasMarkovEconomy

RWEconomyDictionary = {
    'PermShkAggCount': 1, 
    'TranShkAggCount': 1, 
    'PermShkAggStd': [0.0,0.0], 
    'TranShkAggStd': [0.0,0.0], 
    'DeprFac': 0.025, # Depreciation factor
    'DiscFac': 0.99,
    'CRRA': 1.0,
    'PermGroFacAgg': [1.0,1.0],
    'AggregateL':1.0, # Fix aggregate labor supply at 1.0 - makes interpretation of z easier
    'act_T':1200, # Number of periods for economy to run in simulation
    'intercept_prev': [0.5,0.5], # Make some initial guesses at linear savings rule intercepts for each state
    'slope_prev': [0.5,0.5], # Make some initial guesses at linear savings rule slopes for each state
    'MrkvNow_init': 0   # Pick a state to start in (we pick the first state)
}

# The 'interesting' parts of the CobbDouglasMarkovEconomy
RWEconomyDictionary['CapShare']  = 0.36
RWEconomyDictionary['MrkvArray'] = AggMrkvArray

RWEconomy = CobbDouglasMarkovEconomy(agents = [RWAgent], **RWEconomyDictionary) # Combine production and consumption sides into an "Economy"



# %% code_folding=[]
# Calibrate the magnitude of the aggregate shocks

Tran_g = 1.01 # Productivity z in the good aggregate state
Tran_b = 0.99 # and the bad state

# The HARK framework allows permanent shocks
Perm_g = Perm_b = 1.0 # KS assume there are no aggregate permanent shocks

# Aggregate productivity shock distribution by state.
# First element is probabilities of different outcomes, given the state you are in. 
# Second element is agg permanent shocks (here we don't have any, so just they are just 1.).
# Third  element is agg transitory shocks, which are calibrated the same as in Krusell Smith.

RWAggShkDstn = [
     DiscreteDistribution(np.array([1.0]), [np.array([Perm_g]), np.array([Tran_g])]), # Aggregate good
     DiscreteDistribution(np.array([1.0]), [np.array([Perm_b]), np.array([Tran_b])])  # Aggregate bad
]

RWEconomy.AggShkDstn = RWAggShkDstn

RWAgent.getEconomyData(RWEconomy)

RWEconomy.makeAggShkHist()  # Simulate a history of aggregate shocks




# %% code_folding=[]


RWEconomy.tolerance = 0.01

# Solve macro problem by finding a fixed point for beliefs

RWEconomy.solve() # Solve the economy using the market method. 
# i.e. guess the saving function, and iterate until a fixed point
RWAgent.solve()

    
# %% Create Figures

import sys
import os

# Find pathname to this file:
my_file_path = os.path.dirname(os.path.abspath("code.py"))
figures_dir = os.path.join(my_file_path, "Figures") 

#Graph 1    
bottom = 0.1
top = 2 * RWEconomy.kSS
x = np.linspace(bottom, top, 1000, endpoint=True)
y0 = RWEconomy.AFunc[0](x)
y1 = RWEconomy.AFunc[1](x)
f = plt.figure()    
plt.plot(x, y0)
plt.plot(x, y1)
plt.xlim([bottom, top])
make_figs('aggregate_savings', True, False, '../../Figures')
f.savefig(os.path.join(figures_dir, 'ray1.pdf'))
f.savefig(os.path.join(figures_dir, 'ray1.png'))
f.savefig(os.path.join(figures_dir, 'ray1.svg'))

#Graph 2    
RWAgent.unpackcFunc()
m_grid = np.linspace(0,10,200)
RWAgent.unpackcFunc()
f = plt.figure()    
for M in RWAgent.Mgrid:
    c_at_this_M = RWAgent.solution[0].cFunc[0](m_grid,M*np.ones_like(m_grid)) #Have two consumption functions, check this
    plt.plot(m_grid,c_at_this_M)
make_figs('consumption_function', True, False, '../../Figures')
f.savefig(os.path.join(figures_dir, 'ray2.pdf'))
f.savefig(os.path.join(figures_dir, 'ray2.png'))
f.savefig(os.path.join(figures_dir, 'ray2.svg'))

#Graph 3
RWAgent.unpackcFunc()
m_grid = np.linspace(0,10,200)
RWAgent.unpackcFunc()
f = plt.figure()    
for M in RWAgent.Mgrid:
    c_at_this_M = RWAgent.solution[0].cFunc[1](m_grid,M*np.ones_like(m_grid)) #Have two consumption functions, check this
    plt.plot(m_grid,c_at_this_M)
make_figs('consumption_function', True, False, '../../Figures')
f.savefig(os.path.join(figures_dir, 'ray3.pdf'))
f.savefig(os.path.join(figures_dir, 'ray3.png'))
f.savefig(os.path.join(figures_dir, 'ray3.svg'))

#Graph 4    
m_grid = np.linspace(0,10,200)
RWAgent.unpack('LbrFunc')
f = plt.figure()    
for M in RWAgent.Mgrid:
    l_at_this_M = RWAgent.solution[0].LbrFunc[0](m_grid,M*np.ones_like(m_grid)) #Have two consumption functions, check this
    plt.plot(m_grid,l_at_this_M)
make_figs('labor_function', True, False, '../../Figures')
f.savefig(os.path.join(figures_dir, 'ray4.pdf'))
f.savefig(os.path.join(figures_dir, 'ray4.png'))
f.savefig(os.path.join(figures_dir, 'ray4.svg'))

#Graph 5    
m_grid = np.linspace(0,10,200)
RWAgent.unpack('LbrFunc')
f = plt.figure()    
for M in RWAgent.Mgrid:
    l_at_this_M = RWAgent.solution[0].LbrFunc[1](m_grid,M*np.ones_like(m_grid)) #Have two consumption functions, check this
    plt.plot(m_grid,l_at_this_M)
make_figs('labor_function', True, False, '../../Figures')
f.savefig(os.path.join(figures_dir, 'ray5.pdf'))
f.savefig(os.path.join(figures_dir, 'ray5.png'))
f.savefig(os.path.join(figures_dir, 'ray5.svg'))

# #Optional Wealth Distribution Graph
# sim_wealth = RWEconomy.reap_state['aLvlNow'][0]

# print("The mean of individual wealth is "+ str(sim_wealth.mean()) + ";\n the standard deviation is "
#       + str(sim_wealth.std())+";\n the median is " + str(np.median(sim_wealth)) +".")

# # Tools for plotting simulated vs actual wealth distributions
# from HARK.utilities import getLorenzShares, getPercentiles

# # The cstwMPC model conveniently has data on the wealth distribution 
# # from the U.S. Survey of Consumer Finances
# from HARK.datasets import load_SCF_wealth_weights
# SCF_wealth, SCF_weights = load_SCF_wealth_weights()

# # Construct the Lorenz curves and plot them

# pctiles = np.linspace(0.001,0.999,15)
# SCF_Lorenz_points = getLorenzShares(SCF_wealth,weights=SCF_weights,percentiles=pctiles)
# sim_Lorenz_points = getLorenzShares(sim_wealth,percentiles=pctiles)

# # Plot 
# f = plt.figure()    
# plt.figure(figsize=(5,5))
# plt.title('Wealth Distribution')
# plt.plot(pctiles,SCF_Lorenz_points,'--k',label='SCF')
# plt.plot(pctiles,sim_Lorenz_points,'-b',label='Benchmark RW')
# plt.plot(pctiles,pctiles,'g-.',label='45 Degree')
# plt.xlabel('Percentile of net worth')
# plt.ylabel('Cumulative share of wealth')
# plt.legend(loc=2)
# plt.ylim([0,1])
# make_figs('wealth_distribution_1', True, False, '../../Figures')
# plt.show()
# f.savefig(os.path.join(figures_dir, 'ray6.pdf'))
# f.savefig(os.path.join(figures_dir, 'ray6.png'))
# f.savefig(os.path.join(figures_dir, 'ray6.svg'))