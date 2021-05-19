

from __future__ import division, print_function
from __future__ import absolute_import
from builtins import range
import numpy as np
from HARK.core import AgentType, HARKobject
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsIndShockSolver,
    ValueFunc,
    MargValueFunc,
    IndShockConsumerType,
    PerfForesightConsumerType,
)

from HARK.distribution import DiscreteDistribution, Uniform
from HARK.interpolation import CubicInterp, LowerEnvelope, LinearInterp
from HARK.utilities import (
    CRRAutility,
    CRRAutilityP,
    CRRAutilityPP,
    CRRAutilityP_inv,
    CRRAutility_invP,
    CRRAutility_inv,
    CRRAutilityP_invP,
)
from scipy.io import loadmat
from HARK.utilities import  NullFunc, plotFuncs


"""
Authors: William Du and Tung-Sheng Hsieh

base: Python 3.7.3

This code is an alteration of ConsMarkovModel.Py 

The version of the ConsMarkovModel.Py used was last updated October 21st 2020

Note** GL denotes Guerrieri Lorenzoni


There are three main classes in this code:
    
    GLConsumerSolution is the usual Consumer Solution with the addition of the labor supply function.
    - appendSolution
    
    GLSolver solves a single period of the consumption-labor-saving problem with stochastic transition between discrete states.
    - defBoundaryGL
    - calcEndOfPrdvPGL
    - getPointsForInterpolationGL
    - makeSolutionGL
    
    GLConsumerType represents the agent in the consumption-labor-saving model.
    - updateSolutionTerminal
    
Underneath each class are the functions that were edited. If a function in the original ConMarkovModel.py was edited,
then it will likely have a GL at its name. 
    
"""






"""
Classes to solve and simulate consumption-savings model with a discrete, exogenous,
stochastic Markov state.  The only solver here extends ConsIndShockModel to
include a Markov state; the interest factor, permanent growth factor, and income
distribution can vary with the discrete state.
"""




__all__ = ["GLSolver", "GLConsumerType"]

utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP

class GLConsumerSolution(HARKobject):
    """
    A class representing the solution of a single period of a consumption-saving
    problem.  The solution must include a consumption function and marginal
    value function.
    Here and elsewhere in the code, Nrm indicates that variables are normalized
    by permanent income.
    """

    distance_criteria = ["vPfunc"]

    def __init__(
        self,
        cFunc=None,
        LFunc=None,
        vFunc=None,
        vPfunc=None,
        vPPfunc=None,
        mNrmMin=None,
        hNrm=None,
        MPCmin=None,
        MPCmax=None,
    ):
        """
        The constructor for a new ConsumerSolution object.
        Parameters
        ----------
        cFunc : function
            The consumption function for this period, defined over bond
            holdings: c = cFunc(b).
        LFunc: function
            The Labor Supply function for this period, defined over bond
            holdings: l = LFunc(b)
        vFunc : function
            The beginning-of-period value function for this period, defined over
            market resources: v = vFunc(m).
        vPfunc : function
            The beginning-of-period marginal value function for this period,
            defined over bond holdings: vP = vPfunc(b).
        vPPfunc : function
            The beginning-of-period marginal marginal value function for this
            period, defined over market resources: vPP = vPPfunc(m).
        mNrmMin : float
            The minimum allowable market resources for this period; the consump-
            tion function (etc) are undefined for m < mNrmMin.
        hNrm : float
            Human wealth after receiving income this period: PDV of all future
            income, ignoring mortality.
        MPCmin : float
            Infimum of the marginal propensity to consume this period.
            MPC --> MPCmin as m --> infinity.
        MPCmax : float
            Supremum of the marginal propensity to consume this period.
            MPC --> MPCmax as m --> mNrmMin.
        Returns
        -------
        None
        """
        # Change any missing function inputs to NullFunc
        self.cFunc = cFunc if cFunc is not None else NullFunc()
        self.LFunc = LFunc if LFunc is not None else NullFunc()

        self.vFunc = vFunc if vFunc is not None else NullFunc()
        self.vPfunc = vPfunc if vPfunc is not None else NullFunc()
        # vPFunc = NullFunc() if vPfunc is None else vPfunc
        self.vPPfunc = vPPfunc if vPPfunc is not None else NullFunc()
        self.mNrmMin = mNrmMin
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax
        
    
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
            self.LFunc = [new_solution.LFunc]
            self.vFunc = [new_solution.vFunc]
            self.vPfunc = [new_solution.vPfunc]
            self.vPPfunc = [new_solution.vPPfunc]
            self.mNrmMin = [new_solution.mNrmMin]
        else:
            self.cFunc.append(new_solution.cFunc)
            self.LFunc.append(new_solution.LFunc)
            self.vFunc.append(new_solution.vFunc)
            self.vPfunc.append(new_solution.vPfunc)
            self.vPPfunc.append(new_solution.vPPfunc)
            self.mNrmMin.append(new_solution.mNrmMin)


class GLSolver(ConsIndShockSolver):
    """
    A class to solve a single period consumption-saving problem with risky income
    and stochastic transitions between discrete states, in a Markov fashion.
    Extends ConsIndShockSolver, with identical inputs but for a discrete
    Markov state, whose transition rule is summarized in MrkvArray.  Markov
    states can differ in their interest factor, permanent growth factor, live probability, and
    income distribution, so the inputs Rfree, PermGroFac, IncomeDstn, and LivPrb are
    now arrays or lists specifying those values in each (succeeding) Markov state.
    """

    def __init__(
        self,
        solution_next,
        IncomeDstn_list,
        LivPrb,
        DiscFac,
        CRRA,
        Rfree_list,
        PermGroFac_list,
        MrkvArray,
        BoroCnstArt,
        aXtraGrid,
        vFuncBool,
        CubicBool,
        eta,
        nu,
        pssi,
        B,
    ):
        """
        Constructor for a new solver for a one period problem with risky income
        and transitions between discrete Markov states.  In the descriptions below,
        N is the number of discrete states.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn_list : [[np.array]]
            A length N list of income distributions in each succeeding Markov
            state.  Each income distribution contains three arrays of floats,
            representing a discrete approximation to the income process at the
            beginning of the succeeding period. Order: event probabilities,
            permanent shocks, transitory shocks.
        LivPrb : np.array
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period for each Markov state.
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree_list : np.array
            Risk free interest factor on end-of-period assets for each Markov
            state in the succeeding period.
        PermGroFac_list : np.array
            Expected permanent income growth factor at the end of this period
            for each Markov state in the succeeding period.
        MrkvArray : np.array
            An NxN array representing a Markov transition matrix between discrete
            states.  The i,j-th element of MrkvArray is the probability of
            moving from state i in period t to state j in period t+1.
        BoroCnstArt: float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
            rowing constraint.
        aXtraGrid: np.array
            Array of "extra" end-of-period asset values-- assets above the
            absolute minimum acceptable level.
        vFuncBool: boolean
            An indicator for whether the value function should be computed and
            included in the reported solution.
        CubicBool: boolean
            An indicator for whether the solver should use cubic or linear inter-
            polation.
        eta : float
           Coefficient for the Curvature of utility from leisure
        nu : float
            UI benefits
        pssi : float
            Disutility from labor as if representative agent
        B : float
            Bond Supply
        
            
    
        Returns
        -------
        None
        """
        # Set basic attributes of the problem

        self.assignParameters(
            solution_next=solution_next,
            IncomeDstn_list=IncomeDstn_list,
            LivPrb=LivPrb,
            DiscFac=DiscFac,
            CRRA=CRRA,
            BoroCnstArt=BoroCnstArt,
            aXtraGrid=aXtraGrid,
            vFuncBool=vFuncBool,
            CubicBool=CubicBool,
            Rfree_list=Rfree_list,
            PermGroFac_list=PermGroFac_list,
            MrkvArray=MrkvArray,
            StateCount=MrkvArray.shape[0],
            eta=eta,
            nu=nu,
            pssi=pssi,
            B=B,
        )
        self.defUtilityFuncs()

    def solve(self):
        """
        Solve the one period problem of the consumption-saving model with a Markov state.

        Parameters
        ----------
        none

        Returns
        -------
        solution : ConsumerSolution
            The solution to the single period consumption-labor-saving problem. Includes
            a consumption function cFunc (using cubic or linear splines), a labor supply function,
            a marginal value function vPfunc, a minimum acceptable level of normalized
            market resources mNrmMin, normalized human wealth hNrm, and bounding
            MPCs MPCmin and MPCmax.  It might also have a value function vFunc
            and marginal marginal value function vPPfunc.  All of these attributes
            are lists or arrays, with elements corresponding to the current
            Markov state.  E.g. solution.cFunc[0] is the consumption function
            when in the i=0 Markov state this period.
        """
        # Find the natural borrowing constraint in each current state
        self.defBoundaryGL()

        # Initialize end-of-period (marginal) value functions
        self.EndOfPrdvFunc_list = []
        self.EndOfPrdvPfunc_list = []
        self.ExIncNextAll = (
            np.zeros(self.StateCount) + np.nan
        )  # expected income conditional on the next state
        self.WorstIncPrbAll = (
            np.zeros(self.StateCount) + np.nan
        )  # probability of getting the worst income shock in each next period state

        # Loop through each next-period-state and calculate the end-of-period
        # (marginal) value function
        for j in range(self.StateCount):
            # Condition values on next period's state (and record a couple for later use)
            self.conditionOnState(j)
            self.ExIncNextAll[j] = np.dot(
                self.ShkPrbsNext, self.PermShkValsNext * self.TranShkValsNext
            )
            self.WorstIncPrbAll[j] = self.WorstIncPrb

            # Construct the end-of-period marginal value function conditional
            # on next period's state and add it to the list of value functions
            EndOfPrdvPfunc_cond = self.makeEndOfPrdvPfuncCond()
            self.EndOfPrdvPfunc_list.append(EndOfPrdvPfunc_cond)

            # Construct the end-of-period value functional conditional on next
            # period's state and add it to the list of value functions
            if self.vFuncBool:
                EndOfPrdvFunc_cond = self.makeEndOfPrdvFuncCond()
                self.EndOfPrdvFunc_list.append(EndOfPrdvFunc_cond)

        # EndOfPrdvP_cond is EndOfPrdvP conditional on *next* period's state.
        # Take expectations to get EndOfPrdvP conditional on *this* period's state.
        self.calcEndOfPrdvPGL()

        # Calculate the bounding MPCs and PDV of human wealth for each state
        self.calcHumWealthAndBoundingMPCs()

        # Find consumption and market resources corresponding to each end-of-period
        # assets point for each state (and add an additional point at the lower bound)
        aNrm = (
            np.asarray(self.aXtraGrid)[np.newaxis, :]
            + np.array(self.BoroCnstNat_list)[:, np.newaxis]
        )
        
        self.getPointsForInterpolationGL(self.EndOfPrdvP, aNrm)
        
        # Package and return the solution for this period
        self.BoroCnstNat = self.BoroCnstNat_list
        solution = self.makeSolutionGL(self.cNow, self.Bnow, self.Nnow)
        return solution
    
    
    def getPointsForInterpolationGL(self, EndOfPrdvP, aNrmNow):
        """
        Finds interpolation points (c,m) for the consumption function.
        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aNrmNow : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.
        Returns
        -------
        c_for_interpolation : np.array
            Consumption points for interpolation.
        m_for_interpolation : np.array
            Corresponding market resource points for interpolation.
        """
        
        # minimum consumption value for each state
        Matlabcl=loadmat('cl')
        cldata=list(Matlabcl.items())
        cldata=np.array(cldata)
        cl=cldata[3,1].reshape(13,1)
        
        # import Income Process
        Matlabdict = loadmat('inc_process.mat')
        data = list(Matlabdict.items())
        data_array=np.asarray(data)
        x=data_array[3,1] #log productivity
        Pr=data_array[4,1]
        pr = data_array[5,1]
        
        # vector where each element is a level of productivity, The first element is productivity of effectively zero (unemployed)
        theta = np.concatenate((np.array([1e-100000]).reshape(1,1),np.exp(x).reshape(1,12)),axis=1).reshape(13,)
        
        fin   = 0.8820 # job-finding probability
        sep   = 0.0573 # separation probability
        cmin= 1e-6     # lower bound on consumption 
        
        
        #constructing transition Matrix
        G=np.array([1-fin]).reshape(1,1)
        A = np.concatenate((G, fin*pr), axis=1)
        K= sep**np.ones(12).reshape(12,1)
        D=np.concatenate((K,np.multiply((1-sep),Pr)),axis=1)
        Pr = np.concatenate((A,D)) # Markov Array
        
        
        # find new invariate distribution
        pr = np.concatenate([np.array([0]).reshape(1,1), pr],axis=1)
        
        dif = 1
        while dif > 1e-5:
            pri = pr.dot(Pr)
            dif = np.amax(np.absolute(pri-pr))
            pr  = pri
    
        
    
        fac = ((self.pssi / theta)** (1/self.eta)).reshape(13,)  # parameter for calculating Nnow
        tau = (self.nu*pr[0,0] + (self.Rfree-1)/(self.Rfree)*self.B) / (1 - pr[0,0]) # labor tax
        z = np.insert(-tau*np.ones(12),0,self.nu).reshape(13,1) # full transfer scheme
        
        #diagonalize for computational purposes
        facMat = np.diag(fac) 
        thetaMat = np.diag(theta)
        
        
        self.Bgrid_rep=np.tile(self.Bgrid,(13,1))
              
        #Endogenous Gridpoints Method
        cNow = self.uPinv(EndOfPrdvP) #FOC for consumption
        Nnow = np.maximum(1-(facMat.dot(cNow**(self.CRRA/self.eta))),0)  #labor supply FOC
        Bnow = (self.Bgrid_rep/(self.Rfree)) + cNow - thetaMat.dot(Nnow) - z # Budget constraint
        
        #Constrained
        for i in range(13):
            if Bnow[i,0] < self.BoroCnstArt:
                c_c = np.linspace(cl[i,0], cNow[i,0], 6) 
                n_c = np.maximum(1 - fac[i]*(c_c**(self.CRRA/self.eta)),0)  # labor supply
                b_c = self.BoroCnstArt/self.Rfree + c_c - theta[i]*n_c - z[i] # budget
                Bnow[i] = np.concatenate([b_c[0:5], Bnow[i,5:183]])
                Nnow[i]= np.concatenate([n_c[0:5], Nnow[i,5:183]])
                cNow[i] = np.concatenate([c_c[0:5], cNow[i,5:183]])
      
        cNow=np.maximum(cNow,cmin)

        # Limiting consumption is zero as m approaches mNrmMin
        c_for_interpolation = np.insert(cNow, 0, cmin, axis=-1)
        m_for_interpolation = np.insert(Bnow, 0,  self.BoroCnstArt, axis=-1)

        # Storage
        self.cNow = cNow
        self.Nnow = Nnow 
        self.Bnow = Bnow
        
        

        return c_for_interpolation, m_for_interpolation

    def defBoundaryGL(self):
        
        """
        Find the borrowing constraint for each current state and save it as an
        attribute of self for use by other methods.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        
        self.BoroCnstNatAll = np.zeros(self.StateCount) + np.nan
        # Find the natural borrowing constraint conditional on next period's state
        for j in range(self.StateCount):
            PermShkMinNext = np.min(self.IncomeDstn_list[j].X[0])
            TranShkMinNext = np.min(self.IncomeDstn_list[j].X[1])
            
            
            self.BoroCnstNatAll[j] = (
                (self.solution_next.mNrmMin[j] - TranShkMinNext - .1670 ) # .1670 = z[0]: the level of unemployment benefits
                * (self.PermGroFac_list[j] * PermShkMinNext)
                / self.Rfree_list[j]
            )
            
        self.BoroCnstNat_list = np.zeros(self.StateCount) + np.nan
        self.mNrmMin_list = np.zeros(self.StateCount) + np.nan
        self.BoroCnstDependency = np.zeros((self.StateCount, self.StateCount)) + np.nan
        # The natural borrowing constraint in each current state is the *highest*
        # among next-state-conditional natural borrowing constraints that could
        # occur from this current state.
        for i in range(self.StateCount):
            possible_next_states = self.MrkvArray[i, :] > 0
            self.BoroCnstNat_list[i] = np.max(self.BoroCnstNatAll[possible_next_states])

            # Explicitly handle the "None" case:
            if self.BoroCnstArt is None:
                self.mNrmMin_list[i] = self.BoroCnstNat_list[i]
            else:
                self.mNrmMin_list[i] = np.max(
                    [self.BoroCnstNat_list[i], self.BoroCnstArt]
                )
            self.BoroCnstDependency[i, :] = (
                self.BoroCnstNat_list[i] == self.BoroCnstNatAll
            )
        # Also creates a Boolean array indicating whether the natural borrowing
        # constraint *could* be hit when transitioning from i to j.

    def conditionOnState(self, state_index):
        """
        Temporarily assume that a particular Markov state will occur in the
        succeeding period, and condition solver attributes on this assumption.
        Allows the solver to construct the future-state-conditional marginal
        value function (etc) for that future state.

        Parameters
        ----------
        state_index : int
            Index of the future Markov state to condition on.

        Returns
        -------
        none
        """
        # Set future-state-conditional values as attributes of self
        self.IncomeDstn = self.IncomeDstn_list[state_index]
        self.Rfree = self.Rfree_list[state_index]
        self.PermGroFac = self.PermGroFac_list[state_index]
        self.vPfuncNext = self.solution_next.vPfunc[state_index]
        self.mNrmMinNow = self.mNrmMin_list[state_index]
        self.BoroCnstNat = self.BoroCnstNatAll[state_index]
        self.setAndUpdateValues(
            self.solution_next, self.IncomeDstn, self.LivPrb, self.DiscFac
        )
        self.DiscFacEff = (
            self.DiscFac
        )  # survival probability LivPrb represents probability from
        # *current* state, so DiscFacEff is just DiscFac for now

        # These lines have to come after setAndUpdateValues to override the definitions there
        self.vPfuncNext = self.solution_next.vPfunc[state_index]
        if self.CubicBool:
            self.vPPfuncNext = self.solution_next.vPPfunc[state_index]
        if self.vFuncBool:
            self.vFuncNext = self.solution_next.vFunc[state_index]

    def calcEndOfPrdvPP(self):
        """
        Calculates end-of-period marginal marginal value using a pre-defined
        array of next period market resources in self.mNrmNext.

        Parameters
        ----------
        none

        Returns
        -------
        EndOfPrdvPP : np.array
            End-of-period marginal marginal value of assets at each value in
            the grid of assets.
        """
        EndOfPrdvPP = (
            self.DiscFacEff
            * self.Rfree
            * self.Rfree
            * self.PermGroFac ** (-self.CRRA - 1.0)
            * np.sum(
                self.PermShkVals_temp ** (-self.CRRA - 1.0)
                * self.vPPfuncNext(self.mNrmNext)
                * self.ShkPrbs_temp,
                axis=0,
            )
        )
        return EndOfPrdvPP

    def makeEndOfPrdvFuncCond(self):
        """
        Construct the end-of-period value function conditional on next period's
        state.  NOTE: It might be possible to eliminate this method and replace
        it with ConsIndShockSolver.makeEndOfPrdvFunc, but the self.X_cond
        variables must be renamed.

        Parameters
        ----------
        none

        Returns
        -------
        EndofPrdvFunc_cond : ValueFunc
            The end-of-period value function conditional on a particular state
            occuring in the next period.
        """
        VLvlNext = (
            self.PermShkVals_temp ** (1.0 - self.CRRA)
            * self.PermGroFac ** (1.0 - self.CRRA)
        ) * self.vFuncNext(self.mNrmNext)
        EndOfPrdv_cond = self.DiscFacEff * np.sum(VLvlNext * self.ShkPrbs_temp, axis=0)
        EndOfPrdvNvrs_cond = self.uinv(EndOfPrdv_cond)
        EndOfPrdvNvrsP_cond = self.EndOfPrdvP_cond * self.uinvP(EndOfPrdv_cond)
        EndOfPrdvNvrs_cond = np.insert(EndOfPrdvNvrs_cond, 0, 0.0)
        EndOfPrdvNvrsP_cond = np.insert(EndOfPrdvNvrsP_cond, 0, EndOfPrdvNvrsP_cond[0])
        aNrm_temp = np.insert(self.aNrm_cond, 0, self.BoroCnstNat)
        EndOfPrdvNvrsFunc_cond = CubicInterp(
            aNrm_temp, EndOfPrdvNvrs_cond, EndOfPrdvNvrsP_cond
        )
        EndofPrdvFunc_cond = ValueFunc(EndOfPrdvNvrsFunc_cond, self.CRRA)
        return EndofPrdvFunc_cond

    def calcEndOfPrdvPcond(self):
        """
        Calculate end-of-period marginal value of assets at each point in aNrmNow
        conditional on a particular state occuring in the next period.

        Parameters
        ----------
        None

        Returns
        -------
        EndOfPrdvP : np.array
            A 1D array of end-of-period marginal value of assets.
        """
        
        #EndOfPrdvPcond = ConsIndShockSolver.calcEndOfPrdvP(self)

        EndOfPrdvPcond = ConsIndShockSolver.calcEndOfPrdvP(self)         
        return EndOfPrdvPcond

    def makeEndOfPrdvPfuncCond(self):
        """
        Construct the end-of-period marginal value function conditional on next
        period's state.

        Parameters
        ----------
        None

        Returns
        -------
        EndofPrdvPfunc_cond : MargValueFunc
            The end-of-period marginal value function conditional on a particular
            state occuring in the succeeding period.
        """
        # Get data to construct the end-of-period marginal value function (conditional on next state)
        self.aNrm_cond = self.prepareToCalcEndOfPrdvP()
        self.EndOfPrdvP_cond = self.calcEndOfPrdvPcond()
        EndOfPrdvPnvrs_cond = self.uPinv(
            self.EndOfPrdvP_cond
        )  # "decurved" marginal value
        if self.CubicBool:
            EndOfPrdvPP_cond = self.calcEndOfPrdvPP()
            EndOfPrdvPnvrsP_cond = EndOfPrdvPP_cond * self.uPinvP(
                self.EndOfPrdvP_cond
            )  # "decurved" marginal marginal value

        # Construct the end-of-period marginal value function conditional on the next state.
        if self.CubicBool:
            EndOfPrdvPnvrsFunc_cond = CubicInterp(
                self.aNrm_cond,
                EndOfPrdvPnvrs_cond,
                EndOfPrdvPnvrsP_cond,
                lower_extrap=True,
            )
        else:
            EndOfPrdvPnvrsFunc_cond = LinearInterp(
                self.aNrm_cond, EndOfPrdvPnvrs_cond, lower_extrap=True
            )
        EndofPrdvPfunc_cond = MargValueFunc(
            EndOfPrdvPnvrsFunc_cond, self.CRRA
        )  # "recurve" the interpolated marginal value function
        return EndofPrdvPfunc_cond

    def calcEndOfPrdvPGL(self):
        """
        Calculates end of period marginal value (and marginal marginal) value
        at each aXtra gridpoint for each current state, unconditional on the
        future Markov state (i.e. weighting conditional end-of-period marginal
        value by transition probabilities).

        Parameters
        ----------
        none

        Returns
        -------
        none
        """

        
        # Find unique values of minimum acceptable end-of-period assets (and the
        # current period states for which they apply).
        aNrmMin_unique, state_inverse = np.unique(
            self.BoroCnstNat_list, return_inverse=True
        )
        self.possible_transitions = self.MrkvArray > 0

        # Calculate end-of-period marginal value (and marg marg value) at each
        # asset gridpoint for each current period state
        EndOfPrdvP = np.zeros((self.StateCount, self.aXtraGrid.size))
        EndOfPrdvPP = np.zeros((self.StateCount, self.aXtraGrid.size))
        for k in range(aNrmMin_unique.size):
            aNrmMin = aNrmMin_unique[k]  # minimum assets for this pass
            which_states = (
                state_inverse == k
            )  # the states for which this minimum applies
            
           # Construct Asset Grid 
            Bgrid_uc = -2+((np.array(range(0,200))/200)**2)*52  # asset grid used in authors' code
            self.Bgrid=[]
            for i in range(200):
                if  Bgrid_uc[i] > self.BoroCnstArt:
                    self.Bgrid.append(Bgrid_uc[i])
                    
            self.Bgrid = np.array(self.Bgrid).reshape(1,len(self.Bgrid))
            aGrid = self.Bgrid #asset grid for this pass
            
            #aGrid = aNrmMin + self.aXtraGrid # assets grid for this pass
            
            EndOfPrdvP_all = np.zeros((self.StateCount, self.aXtraGrid.size))
            EndOfPrdvPP_all = np.zeros((self.StateCount, self.aXtraGrid.size))
            for j in range(self.StateCount):
                if np.any(
                    np.logical_and(self.possible_transitions[:, j], which_states)
                ):  # only consider a future state if one of the relevant states could transition to it
                    EndOfPrdvP_all[j, :] = self.EndOfPrdvPfunc_list[j](aGrid)
                    if (
                        self.CubicBool
                    ):  # Add conditional end-of-period (marginal) marginal value to the arrays
                        EndOfPrdvPP_all[j, :] = self.EndOfPrdvPfunc_list[j].derivative(
                            aGrid
                        )
            # Weight conditional marginal (marginal) values by transition probs
            # to get unconditional marginal (marginal) value at each gridpoint.
           
            EndOfPrdvP_temp = np.dot(self.MrkvArray, EndOfPrdvP_all)
          
            
            EndOfPrdvP[which_states, :] = EndOfPrdvP_temp[
                which_states, :
            ]  # only take the states for which this asset minimum applies
            if self.CubicBool:
                EndOfPrdvPP_temp = np.dot(self.MrkvArray, EndOfPrdvPP_all)
                EndOfPrdvPP[which_states, :] = EndOfPrdvPP_temp[which_states, :]

        # Store the results as attributes of self, scaling end of period marginal value by survival probability from each current state
        LivPrb_tiled = np.tile(
            np.reshape(self.LivPrb, (self.StateCount, 1)), (1, self.aXtraGrid.size)
        )
        self.EndOfPrdvP = LivPrb_tiled * EndOfPrdvP    #---------------############################################
        if self.CubicBool:
            self.EndOfPrdvPP = LivPrb_tiled * EndOfPrdvPP

    def calcHumWealthAndBoundingMPCs(self):
        """
        Calculates human wealth and the maximum and minimum MPC for each current
        period state, then stores them as attributes of self for use by other methods.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        # Upper bound on MPC at lower m-bound
        WorstIncPrb_array = self.BoroCnstDependency * np.tile(
            np.reshape(self.WorstIncPrbAll, (1, self.StateCount)), (self.StateCount, 1)
        )
        temp_array = self.MrkvArray * WorstIncPrb_array
        WorstIncPrbNow = np.sum(
            temp_array, axis=1
        )  # Probability of getting the "worst" income shock and transition from each current state
        ExMPCmaxNext = (
            np.dot(
                temp_array,
                self.Rfree_list ** (1.0 - self.CRRA)
                * self.solution_next.MPCmax ** (-self.CRRA),
            )
            / WorstIncPrbNow
        ) ** (-1.0 / self.CRRA)
        DiscFacEff_temp = self.DiscFac * self.LivPrb
        self.MPCmaxNow = 1.0 / (
            1.0
            + ((DiscFacEff_temp * WorstIncPrbNow) ** (1.0 / self.CRRA)) / ExMPCmaxNext
        )
        self.MPCmaxEff = self.MPCmaxNow
        self.MPCmaxEff[self.BoroCnstNat_list < self.mNrmMin_list] = 1.0
        # State-conditional PDV of human wealth
        hNrmPlusIncNext = self.ExIncNextAll + self.solution_next.hNrm
        self.hNrmNow = np.dot(
            self.MrkvArray, (self.PermGroFac_list / self.Rfree_list) * hNrmPlusIncNext
        )
        # Lower bound on MPC as m gets arbitrarily large
        temp = (
            DiscFacEff_temp
            * np.dot(
                self.MrkvArray,
                self.solution_next.MPCmin ** (-self.CRRA)
                * self.Rfree_list ** (1.0 - self.CRRA),
            )
        ) ** (1.0 / self.CRRA)
        self.MPCminNow = 1.0 / (1.0 + temp)

    def makeSolutionGL(self, cNrm, BNrm, NNrm):
        """
        Construct an object representing the solution to this period's problem.

        Parameters
        ----------
        cNrm : np.array
            Array of normalized consumption values for interpolation.  Each row
            corresponds to a Markov state for this period.
        mNrm : np.array
            Array of normalized market resource values for interpolation.  Each
            row corresponds to a Markov state for this period.

        Returns
        -------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem. Includes
            a consumption function cFunc (using cubic or linear splines), a marg-
            inal value function vPfunc, a minimum acceptable level of normalized
            market resources mNrmMin, normalized human wealth hNrm, and bounding
            MPCs MPCmin and MPCmax.  It might also have a value function vFunc
            and marginal marginal value function vPPfunc.  All of these attributes
            are lists or arrays, with elements corresponding to the current
            Markov state.  E.g. solution.cFunc[0] is the consumption function
            when in the i=0 Markov state this period.
        """
        solution = (
            GLConsumerSolution()
        )  # An empty solution to which we'll add state-conditional solutions
        # Calculate the MPC at each market resource gridpoint in each state (if desired)
        if self.CubicBool:
            dcda = self.EndOfPrdvPP / self.uPP(np.array(self.cNow))
            MPC = dcda / (dcda + 1.0)
            self.MPC_temp = np.hstack(
                (np.reshape(self.MPCmaxNow, (self.StateCount, 1)), MPC)
            )
            interpfunc = self.makeCubiccFunc
        else:
            interpfunc = self.makeLinearcFunc

        # Loop through each current period state and add its solution to the overall solution
        for i in range(self.StateCount):
            # Set current-period-conditional human wealth and MPC bounds
            self.hNrmNow_j = self.hNrmNow[i]
            self.MPCminNow_j = self.MPCminNow[i]
            if self.CubicBool:
                self.MPC_temp_j = self.MPC_temp[i, :]

            # Construct the consumption function by combining the constrained and unconstrained portions
            self.cFuncNowCnst = LinearInterp(
                [self.mNrmMin_list[i], self.mNrmMin_list[i] + 1.0], [0.0, 1.0]
            )
            
            cFuncNowUnc = interpfunc(BNrm[i, :], cNrm[i, :])
            cFuncNow = LowerEnvelope(cFuncNowUnc, self.cFuncNowCnst)
            
            LFuncNow = interpfunc(BNrm[i,:], NNrm[i,:])

            # Make the marginal value function and pack up the current-state-conditional solution
            vPfuncNow = MargValueFunc(cFuncNow, self.CRRA)
            solution_cond = GLConsumerSolution(
                cFunc=cFuncNow, LFunc=LFuncNow, vPfunc=vPfuncNow, mNrmMin=self.mNrmMinNow
            )
            if (
                self.CubicBool
            ):  # Add the state-conditional marginal marginal value function (if desired)
                solution_cond = self.addvPPfunc(solution_cond)

            # Add the current-state-conditional solution to the overall period solution
            solution.appendSolution(solution_cond)

        # Add the lower bounds of market resources, MPC limits, human resources,
        # and the value functions to the overall solution
        solution.mNrmMin = self.mNrmMin_list
        solution = self.addMPCandHumanWealth(solution)
        if self.vFuncBool:
            vFuncNow = self.makevFunc(solution)
            solution.vFunc = vFuncNow

        # Return the overall solution to this period
        return solution

    def makeLinearcFunc(self, BNrm, cNrm):
        """
        Make a linear interpolation to represent the (unconstrained) consumption
        function conditional on the current period state.

        Parameters
        ----------
        mNrm : np.array
            Array of normalized market resource values for interpolation.
        cNrm : np.array
            Array of normalized consumption values for interpolation.

        Returns
        -------
        cFuncUnc: an instance of HARK.interpolation.LinearInterp
        """
        cFuncUnc = LinearInterp(
            BNrm, cNrm, self.MPCminNow_j * self.hNrmNow_j, self.MPCminNow_j
        )
        return cFuncUnc

    def makeCubiccFunc(self, mNrm, cNrm):
        """
        Make a cubic interpolation to represent the (unconstrained) consumption
        function conditional on the current period state.

        Parameters
        ----------
        mNrm : np.array
            Array of normalized market resource values for interpolation.
        cNrm : np.array
            Array of normalized consumption values for interpolation.

        Returns
        -------
        cFuncUnc: an instance of HARK.interpolation.CubicInterp
        """
        cFuncUnc = CubicInterp(
            mNrm,
            cNrm,
            self.MPC_temp_j,
            self.MPCminNow_j * self.hNrmNow_j,
            self.MPCminNow_j,
        )
        return cFuncUnc

    def makevFunc(self, solution):
        """
        Construct the value function for each current state.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem. Must
            have a consumption function cFunc (using cubic or linear splines) as
            a list with elements corresponding to the current Markov state.  E.g.
            solution.cFunc[0] is the consumption function when in the i=0 Markov
            state this period.

        Returns
        -------
        vFuncNow : [ValueFunc]
            A list of value functions (defined over normalized market resources
            m) for each current period Markov state.
        """
        vFuncNow = []  # Initialize an empty list of value functions
        # Loop over each current period state and construct the value function
        for i in range(self.StateCount):
            # Make state-conditional grids of market resources and consumption
            mNrmMin = self.mNrmMin_list[i]
            mGrid = mNrmMin + self.aXtraGrid
            cGrid = solution.cFunc[i](mGrid)
            aGrid = mGrid - cGrid

            # Calculate end-of-period value at each gridpoint
            EndOfPrdv_all = np.zeros((self.StateCount, self.aXtraGrid.size))
            for j in range(self.StateCount):
                if self.possible_transitions[i, j]:
                    EndOfPrdv_all[j, :] = self.EndOfPrdvFunc_list[j](aGrid)
            EndOfPrdv = np.dot(self.MrkvArray[i, :], EndOfPrdv_all)

            # Calculate (normalized) value and marginal value at each gridpoint
            vNrmNow = self.u(cGrid) + EndOfPrdv
            vPnow = self.uP(cGrid)

            # Make a "decurved" value function with the inverse utility function
            vNvrs = self.uinv(vNrmNow)  # value transformed through inverse utility
            vNvrsP = vPnow * self.uinvP(vNrmNow)
            mNrm_temp = np.insert(mGrid, 0, mNrmMin)  # add the lower bound
            vNvrs = np.insert(vNvrs, 0, 0.0)
            vNvrsP = np.insert(
                vNvrsP, 0, self.MPCmaxEff[i] ** (-self.CRRA / (1.0 - self.CRRA))
            )
            MPCminNvrs = self.MPCminNow[i] ** (-self.CRRA / (1.0 - self.CRRA))
            vNvrsFunc_i = CubicInterp(
                mNrm_temp, vNvrs, vNvrsP, MPCminNvrs * self.hNrmNow[i], MPCminNvrs
            )

            # "Recurve" the decurved value function and add it to the list
            vFunc_i = ValueFunc(vNvrsFunc_i, self.CRRA)
            vFuncNow.append(vFunc_i)
        return vFuncNow


def _solveGL(
    solution_next,
    IncomeDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    MrkvArray,
    BoroCnstArt,
    aXtraGrid,
    vFuncBool,
    CubicBool,
    eta,
    nu,
    pssi,
    B,
):
    """
    Solves a single period consumption-saving problem with risky income and
    stochastic transitions between discrete states, in a Markov fashion.  Has
    identical inputs as solveConsIndShock, except for a discrete
    Markov transitionrule MrkvArray.  Markov states can differ in their interest
    factor, permanent growth factor, and income distribution, so the inputs Rfree,
    PermGroFac, and IncomeDstn are arrays or lists specifying those values in each
    (succeeding) Markov state.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncomeDstn_list : [[np.array]]
        A length N list of income distributions in each succeeding Markov
        state.  Each income distribution contains three arrays of floats,
        representing a discrete approximation to the income process at the
        beginning of the succeeding period. Order: event probabilities,
        permanent shocks, transitory shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree_list : np.array
        Risk free interest factor on end-of-period assets for each Markov
        state in the succeeding period.
    PermGroGac_list : float
        Expected permanent income growth factor at the end of this period
        for each Markov state in the succeeding period.
    MrkvArray : numpy.array
        An NxN array representing a Markov transition matrix between discrete
        states.  The i,j-th element of MrkvArray is the probability of
        moving from state i in period t to state j in period t+1.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.

    Returns
    -------
    solution : ConsumerSolution
        The solution to the single period consumption-saving problem. Includes
        a consumption function cFunc (using cubic or linear splines), a marg-
        inal value function vPfunc, a minimum acceptable level of normalized
        market resources mNrmMin, normalized human wealth hNrm, and bounding
        MPCs MPCmin and MPCmax.  It might also have a value function vFunc
        and marginal marginal value function vPPfunc.  All of these attributes
        are lists or arrays, with elements corresponding to the current
        Markov state.  E.g. solution.cFunc[0] is the consumption function
        when in the i=0 Markov state this period.
    """
    solver = GLSolver(
        solution_next,
        IncomeDstn,
        LivPrb,
        DiscFac,
        CRRA,
        Rfree,
        PermGroFac,
        MrkvArray,
        BoroCnstArt,
        aXtraGrid,
        vFuncBool,
        CubicBool,
        eta,
        nu,
        pssi,
        B,
    )
    solution_now = solver.solve()
    return solution_now

####################################################################################################
####################################################################################################


class GLConsumerType(IndShockConsumerType):
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
    # Is 'MrkvNow' a shock or a state?
    shock_vars_ = IndShockConsumerType.shock_vars_ + ["MrkvNow"]
    state_vars = IndShockConsumerType.state_vars + ["MrkvNow"]

    def __init__(self, cycles=0, **kwds):
        IndShockConsumerType.__init__(self, cycles=100, **kwds)
        self.solveOnePeriod = _solveGL


        if not hasattr(self, "global_markov"):
            self.global_markov = False
            
            
    def checkMarkovInputs(self):
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
        # Note IncomeDstn is (potentially) time-varying, so it is in time_vary.
        # Therefore it is a list, and each element of that list responds to the income distribution
        # at a particular point in time.  Each income distribution at a point in time should itself
        # be a list, with each element corresponding to the income distribution
        # conditional on a particular Markov state.
        # TODO: should this be a numpy array too?
        #for IncomeDstn_t in self.IncomeDstn:
          #  if not isinstance(IncomeDstn_t, list) or len(IncomeDstn_t) != StateCount:
            #    raise ValueError(
           #         "List in IncomeDstn is not the right length, it should be length equal to number of states"
           #     )

    def preSolve(self):
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
        AgentType.preSolve(self)
        self.checkMarkovInputs()

    def updateSolutionTerminal(self):
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
        print('This code should take 8 seconds ')

        
        IndShockConsumerType.updateSolutionTerminal(self)
        
        cmin= 1e-6  # lower bound on consumption


        Bgrid_uc = -2+((np.array(range(0,200))/200)**2)*52 # asset grid used in authors' code
        
        # Cut out asset levels below borrowing constraint
        self.Bgrid=[]
        for i in range(200):
            if  Bgrid_uc[i] > self.BoroCnstArt:
                self.Bgrid.append(Bgrid_uc[i])
        self.Bgrid = np.array(self.Bgrid).reshape(1,len(self.Bgrid))
  

        #initial Guess for Cpolicy
        Cguess = np.maximum((self.Rfree-1).reshape(13,1).dot(self.Bgrid),cmin)
        
        #Construct terminal consumption function and accompanying Marg Value Func
        self.CfuncGuess = LinearInterp(self.Bgrid,Cguess[0])
        self.vPFuncGuess=MargValueFunc(self.CfuncGuess,self.CRRA)

        # Make replicated terminal period solution: consume all resources, no human wealth, minimum m is 0
        StateCount = self.MrkvArray[0].shape[0]
        
        self.solution_terminal.cFunc = StateCount * [self.CfuncGuess]
        self.solution_terminal.vFunc = StateCount * [self.solution_terminal.vFunc]
        self.solution_terminal.vPfunc = StateCount * [ self.vPFuncGuess]
        self.solution_terminal.vPPfunc = StateCount * [self.solution_terminal.vPPfunc]
        self.solution_terminal.mNrmMin = np.ones(StateCount)*(-2)
        self.solution_terminal.hRto = np.zeros(StateCount)
        self.solution_terminal.MPCmax = np.ones(StateCount)
        self.solution_terminal.MPCmin = np.ones(StateCount)

    def initializeSim(self):
        self.shocks["MrkvNow"] = np.zeros(self.AgentCount, dtype=int)
        IndShockConsumerType.initializeSim(self)
        if (
            self.global_markov
        ):  # Need to initialize markov state to be the same for all agents
            base_draw = Uniform(seed=self.RNG.randint(0, 2 ** 31 - 1)).draw(1)
            Cutoffs = np.cumsum(np.array(self.MrkvPrbsInit))
            self.shocks["MrkvNow"] = np.ones(self.AgentCount) * np.searchsorted(
                Cutoffs, base_draw
            ).astype(int)
        self.shocks["MrkvNow"] = self.shocks["MrkvNow"].astype(int)

    def resetRNG(self):
        """
        Extended method that ensures random shocks are drawn from the same sequence
        on each simulation, which is important for structural estimation.  This
        method is called automatically by initializeSim().

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        PerfForesightConsumerType.resetRNG(self)

        # Reset IncomeDstn if it exists (it might not because resetRNG is called at init)
        if hasattr(self, "IncomeDstn"):
            T = len(self.IncomeDstn)
            for t in range(T):
                for dstn in self.IncomeDstn[t]:
                    dstn.reset()

    def simDeath(self):
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
            self.t_cycle - 1, self.shocks["MrkvNow"]
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

    def simBirth(self, which_agents):
        """
        Makes new Markov consumer by drawing initial normalized assets, permanent income levels, and
        discrete states. Calls IndShockConsumerType.simBirth, then draws from initial Markov distribution.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        IndShockConsumerType.simBirth(
            self, which_agents
        )  # Get initial assets and permanent income
        if (
            not self.global_markov
        ):  # Markov state is not changed if it is set at the global level
            N = np.sum(which_agents)
            base_draws = Uniform(seed=self.RNG.randint(0, 2 ** 31 - 1)).draw(N)
            Cutoffs = np.cumsum(np.array(self.MrkvPrbsInit))
            self.shocks["MrkvNow"][which_agents] = np.searchsorted(
                Cutoffs, base_draws
            ).astype(int)

    def getMarkovStates(self):
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
        # Draw random numbers that will be used to determine the next Markov state
        if self.global_markov:
            base_draws = np.ones(self.AgentCount) * Uniform(
                seed=self.RNG.randint(0, 2 ** 31 - 1)
            ).draw(1)
        else:
            base_draws = Uniform(seed=self.RNG.randint(0, 2 ** 31 - 1)).draw(
                self.AgentCount
            )
        dont_change = (
            self.t_age == 0
        )  # Don't change Markov state for those who were just born (unless global_markov)
        if self.t_sim == 0:  # Respect initial distribution of Markov states
            dont_change[:] = True

        # Determine which agents are in which states right now
        J = self.MrkvArray[0].shape[0]
        MrkvPrev = self.shocks["MrkvNow"]
        MrkvNow = np.zeros(self.AgentCount, dtype=int)
        MrkvBoolArray = np.zeros((J, self.AgentCount))

        for j in range(J):
            MrkvBoolArray[j, :] = MrkvPrev == j

        # Draw new Markov states for each agent
        for t in range(self.T_cycle):
            Cutoffs = np.cumsum(self.MrkvArray[t], axis=1)
            right_age = self.t_cycle == t
            for j in range(J):
                these = np.logical_and(right_age, MrkvBoolArray[j, :])
                MrkvNow[these] = np.searchsorted(
                    Cutoffs[j, :], base_draws[these]
                ).astype(int)
        if not self.global_markov:
            MrkvNow[dont_change] = MrkvPrev[dont_change]

        self.shocks["MrkvNow"] = MrkvNow.astype(int)

    def getShocks(self):
        """
        Gets new Markov states and permanent and transitory income shocks for this period.  Samples
        from IncomeDstn for each period-state in the cycle.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.getMarkovStates()
        MrkvNow = self.shocks["MrkvNow"]

        # Now get income shocks for each consumer, by cycle-time and discrete state
        PermShkNow = np.zeros(self.AgentCount)  # Initialize shock arrays
        TranShkNow = np.zeros(self.AgentCount)
        for t in range(self.T_cycle):
            for j in range(self.MrkvArray[t].shape[0]):
                these = np.logical_and(t == self.t_cycle, j == MrkvNow)
                N = np.sum(these)
                if N > 0:
                    IncomeDstnNow = self.IncomeDstn[t - 1][
                        j
                    ]  # set current income distribution
                    PermGroFacNow = self.PermGroFac[t - 1][
                        j
                    ]  # and permanent growth factor

                    # Get random draws of income shocks from the discrete distribution
                    EventDraws = IncomeDstnNow.draw_events(N)
                    PermShkNow[these] = (
                        IncomeDstnNow.X[0][EventDraws] * PermGroFacNow
                    )  # permanent "shock" includes expected growth
                    TranShkNow[these] = IncomeDstnNow.X[1][EventDraws]
        newborn = self.t_age == 0
        PermShkNow[newborn] = 1.0
        TranShkNow[newborn] = 1.0
        self.shocks["PermShkNow"] = PermShkNow
        self.shocks["TranShkNow"] = TranShkNow

    def readShocks(self):
        """
        A slight modification of AgentType.readShocks that makes sure that MrkvNow is int, not float.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        IndShockConsumerType.readShocks(self)
        self.shocks["MrkvNow"] = self.shocks["MrkvNow"].astype(int)

    def getRfree(self):
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
        RfreeNow = self.Rfree[self.shocks["MrkvNow"]]
        return RfreeNow

    def getControls(self):
        """
        Calculates consumption for each consumer of this type using the consumption functions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        J = self.MrkvArray[0].shape[0]

        MrkvBoolArray = np.zeros((J, self.AgentCount), dtype=bool)
        for j in range(J):
            MrkvBoolArray[j, :] = j == self.shocks["MrkvNow"]

        for t in range(self.T_cycle):
            right_t = t == self.t_cycle
            for j in range(J):
                these = np.logical_and(right_t, MrkvBoolArray[j, :])
                cNow[these], MPCnow[these] = (
                    self.solution[t].cFunc[j].eval_with_derivative(self.state_now['mNrmNow'][these])
                )
        self.controls["cNow"] = cNow
        self.MPCnow = MPCnow

    def calcBoundingValues(self):
        """
        Calculate human wealth plus minimum and maximum MPC in an infinite
        horizon model with only one period repeated indefinitely.  Store results
        as attributes of self.  Human wealth is the present discounted value of
        expected future income after receiving income this period, ignoring mort-
        ality.  The maximum MPC is the limit of the MPC as m --> mNrmMin.  The
        minimum MPC is the limit of the MPC as m --> infty.  Results are all
        np.array with elements corresponding to each Markov state.

        NOT YET IMPLEMENTED FOR THIS CLASS

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def makeEulerErrorFunc(self, mMax=100, approx_inc_dstn=True):
        """
        Creates a "normalized Euler error" function for this instance, mapping
        from market resources to "consumption error per dollar of consumption."
        Stores result in attribute eulerErrorFunc as an interpolated function.
        Has option to use approximate income distribution stored in self.IncomeDstn
        or to use a (temporary) very dense approximation.

        NOT YET IMPLEMENTED FOR THIS CLASS

        Parameters
        ----------
        mMax : float
            Maximum normalized market resources for the Euler error function.
        approx_inc_dstn : Boolean
            Indicator for whether to use the approximate discrete income distri-
            bution stored in self.IncomeDstn[0], or to use a very accurate
            discrete approximation instead.  When True, uses approximation in
            IncomeDstn; when False, makes and uses a very dense approximation.

        Returns
        -------
        None
        """
        raise NotImplementedError()
        
#------------------------------------------------------------------------------



