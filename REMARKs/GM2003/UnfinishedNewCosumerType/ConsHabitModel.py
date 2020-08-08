from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
from builtins import object
from copy import copy, deepcopy
import numpy as np
from scipy.optimize import newton
from HARK import AgentType, Solution, NullFunc, HARKobject, makeOnePeriodOOSolver
from HARK.utilities import warnings  # Because of "patch" to warnings modules
from HARK.interpolation import(
        LinearInterp,           # Piecewise linear interpolation
        CubicInterp,            # Piecewise cubic interpolation
        LinearInterpOnInterp1D, # Interpolator over 1D interpolations
        BilinearInterp,         # 2D interpolator
        ConstantFunction,       # Interpolator-like class that returns constant value
        IdentityFunction        # Interpolator-like class that returns one of its arguments
)
from HARK.distribution import Lognormal, MeanOneLogNormal, Uniform
from HARK.distribution import DiscreteDistribution, addDiscreteOutcomeConstantMean, combineIndepDstns 
from HARK.utilities import makeGridExpMult, CRRAutility, CRRAutilityP, \
                           CRRAutilityPP, CRRAutilityP_inv, CRRAutility_invP, CRRAutility_inv, \
                           CRRAutilityP_invP
from HARK import _log
from HARK import set_verbosity_level


__all__ = ['ConsumerSolution', 'ValueFunc', 'MargValueFunc', 'MargMargValueFunc',
'ConsPerfForesightSolver', 'ConsIndShockSetup', 'ConsIndShockSolverBasic', 'PerfForesightConsumerType',
           'init_perfect_foresight', 'init_lifecycle','init_cyclical']

utility       = CRRAutility
utilityP      = CRRAutilityP
utilityPP     = CRRAutilityPP
utilityP_inv  = CRRAutilityP_inv
utility_invP  = CRRAutility_invP
utility_inv   = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP

# =====================================================================
# === Classes that help solve consumption-saving models ===
# =====================================================================

class ConsumerSolution(Solution):
    '''
    A class representing the solution of a single period of a consumption-saving
    problem.  The solution must include a consumption function and marginal
    value function.

    Here and elsewhere in the code, Nrm indicates that variables are normalized
    by permanent income.
    '''
    distance_criteria = ['vPfunc']

    def __init__(self, cFunc=None, vFunc=None,
                       vPfunc=None, vPPfunc=None,
                       mNrmMin=None, hNrm=None, MPCmin=None, MPCmax=None, HNrm=None):
        '''
        The constructor for a new ConsumerSolution object.

        Parameters
        ----------
        cFunc : function
            The consumption function for this period, defined over market
            resources and habit stocks: c = cFunc(m, H).
        vFunc : function
            The beginning-of-period value function for this period, defined over
            market resources and habit stocks: v = vFunc(m,H).
        vPfunc : function
            The beginning-of-period marginal value function for this period,
            defined over market resources and habit stocks: vP = vPfunc(m,H).
        vPPfunc : function
            The beginning-of-period marginal marginal value function for this
            period, defined over market resources and habit stocks: vPP = vPPfunc(m,H).
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
        HNrm : float
            The habit stock for this period, follow a law of motion

        Returns
        -------
        None
        '''
        # Change any missing function inputs to NullFunc
        self.cFunc = cFunc if cFunc is not None else NullFunc()
        self.vFunc = vFunc if vFunc is not None else NullFunc()
        self.vPfunc = vPfunc if vPfunc is not None else NullFunc()
        # vPFunc = NullFunc() if vPfunc is None else vPfunc
        self.vPPfunc = vPPfunc if vPPfunc is not None else NullFunc()
        self.mNrmMin      = mNrmMin
        self.hNrm         = hNrm
        self.MPCmin       = MPCmin
        self.MPCmax       = MPCmax
        self.HNrm         = HNrm
        
    def appendSolution(self,new_solution):
        '''
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
        '''
        if type(self.cFunc)!=list:
            # Then we assume that self is an empty initialized solution instance.
            # Begin by checking this is so.
            assert NullFunc().distance(self.cFunc) == 0, 'appendSolution called incorrectly!'

            # We will need the attributes of the solution instance to be lists.  Do that here.
            self.cFunc       = [new_solution.cFunc]
            self.vFunc       = [new_solution.vFunc]
            self.vPfunc      = [new_solution.vPfunc]
            self.vPPfunc     = [new_solution.vPPfunc]
            self.mNrmMin     = [new_solution.mNrmMin]
            self.haNrm       = [new_solution.haNrm]
        else:
            self.cFunc.append(new_solution.cFunc)
            self.vFunc.append(new_solution.vFunc)
            self.vPfunc.append(new_solution.vPfunc)
            self.vPPfunc.append(new_solution.vPPfunc)
            self.mNrmMin.append(new_solution.mNrmMin)
            self.haNrm.append(new_solution.haNrm)


class ValueFunc(HARKobject):
    '''
    A class for representing a value function.  The underlying interpolation is
    in the space of (m,u_inv(v)); this class "re-curves" to the value function.
    '''
    distance_criteria = ['func','CRRA']

    def __init__(self,vFuncNvrs,CRRA):
        '''
        Constructor for a new value function object.

        Parameters
        ----------
        vFuncNvrs : function
            A real function representing the value function composed with the
            inverse utility function, defined on market resources: u_inv(vFunc(m))
        CRRA : float
            Coefficient of relative risk aversion.

        Returns
        -------
        None
        '''
        self.func = deepcopy(vFuncNvrs)
        self.CRRA = CRRA

    def __call__(self,m):
        '''
        Evaluate the value function at given levels of market resources m.

        Parameters
        ----------
        m : float or np.array
            Market resources (normalized by permanent income) whose value is to
            be found.

        Returns
        -------
        v : float or np.array
            Lifetime value of beginning this period with market resources m; has
            same size as input m.
        '''
        return utility(self.func(m),gam=self.CRRA)

class ValueFunc2D(HARKobject):
    '''
    A class for representing a value function in a model where habit stocks
    is included as the second state variable.  The underlying interpolation is
    in the space of (mNrm,haNrm) --> u_inv(v); this class "re-curves" to the value function.
    '''
    distance_criteria = ['func', 'CRRA']
    
    def __init__(self, vFuncNvrs, CRRA):
        '''
        Constructor for a new value function object.
        Parameters
        ----------
        vFuncNvrs : function
            A real function representing the value function composed with the
            inverse utility function, defined on market resources and habit
            stocks: u_inv(vFunc(m,H))
        CRRA : float
            Coefficient of relative risk aversion.
        Returns
        -------
        None
        '''
        self.func = deepcopy(vFuncNvrs)
        self.CRRA = CRRA
        
    def __call__(self, m, H):
        '''
        Evaluate the value function at given levels of normalized market resources mNrm
        and normalized habit stocks haNrm.

        Parameters
        ----------
        m : float or np.array
            Market resources (normalized by permanent income) whose value is to
            be found.
        H : float or np.array
            Habit stocks (normalized by permanent income) whose value is to
            be found.
        
        Returns
        -------
        v : float or np.array
            Lifetime value of beginning this period with normalized market resources
            m and normalized habit stock H; has same size as inputs m and H.
        '''
        return utility((self.func(m, H)/(H**self.Habitgamma)), gam=self.CRRA)

class MargValueFunc(HARKobject):
    '''
    A class for representing a marginal value function in models where the
    standard envelope condition of v'(m) = u'(c(m)) holds (with CRRA utility).
    '''
    distance_criteria = ['cFunc','CRRA']

    def __init__(self,cFunc,CRRA):
        '''
        Constructor for a new marginal value function object.

        Parameters
        ----------
        cFunc : function
            A real function representing the marginal value function composed
            with the inverse marginal utility function, defined on market
            resources: uP_inv(vPfunc(m)).  Called cFunc because when standard
            envelope condition applies, uP_inv(vPfunc(m)) = cFunc(m).
        CRRA : float
            Coefficient of relative risk aversion.

        Returns
        -------
        None
        '''
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA

    def __call__(self,m):
        '''
        Evaluate the marginal value function at given levels of market resources m.

        Parameters
        ----------
        m : float or np.array
            Market resources (normalized by permanent income) whose marginal
            value is to be found.

        Returns
        -------
        vP : float or np.array
            Marginal lifetime value of beginning this period with market
            resources m; has same size as input m.
        '''
        return utilityP(self.cFunc(m),gam=self.CRRA)

    def derivative(self,m):
        '''
        Evaluate the derivative of the marginal value function at given levels
        of market resources m; this is the marginal marginal value function.

        Parameters
        ----------
        m : float or np.array
            Market resources (normalized by permanent income) whose marginal
            marginal value is to be found.

        Returns
        -------
        vPP : float or np.array
            Marginal marginal lifetime value of beginning this period with market
            resources m; has same size as input m.
        '''
        c, MPC = self.cFunc.eval_with_derivative(m)
        return MPC*utilityPP(c,gam=self.CRRA)

class MargValueFunc2D(HARKobject):
    '''
    A class for representing a marginal value function in models where the
    standard envelope condition of dvdm(m,H) = u'(c(m,H)) holds (with CRRA utility).
    '''
    distance_criteria = ['cFunc', 'CRRA']

    def __init__(self, cFunc, CRRA):
        '''
        Constructor for a new marginal value function object.
        
        Parameters
        ----------
        cFunc : function
            A real function representing the marginal value function composed
            with the inverse marginal utility function, defined on normalized individual market
            resources and normalized habit stocks:  uP_inv(vPfunc(m,H)).
            Called cFunc because when standard envelope condition applies,
            uP_inv(vPfunc(m,H)) = cFunc(m,H).
        CRRA : float
            Coefficient of relative risk aversion.
        Returns
        -------
        new instance of MargValueFunc
        '''
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA

    def __call__(self, m, H):
        return utilityP(self.cFunc(m, H), gam=self.CRRA)/((H**Hgamma)**(1-self.CRRA))

class MargMargValueFunc(HARKobject):
    '''
    A class for representing a marginal marginal value function in models where
    the standard envelope condition of v'(m) = u'(c(m)) holds (with CRRA utility).
    '''
    distance_criteria = ['cFunc','CRRA']

    def __init__(self,cFunc,CRRA):
        '''
        Constructor for a new marginal marginal value function object.

        Parameters
        ----------
        cFunc : function
            A real function representing the marginal value function composed
            with the inverse marginal utility function, defined on market
            resources: uP_inv(vPfunc(m)).  Called cFunc because when standard
            envelope condition applies, uP_inv(vPfunc(m)) = cFunc(m).
        CRRA : float
            Coefficient of relative risk aversion.

        Returns
        -------
        None
        '''
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA

    def __call__(self,m):
        '''
        Evaluate the marginal marginal value function at given levels of market
        resources m.

        Parameters
        ----------
        m : float or np.array
            Market resources (normalized by permanent income) whose marginal
            marginal value is to be found.

        Returns
        -------
        vPP : float or np.array
            Marginal marginal lifetime value of beginning this period with market
            resources m; has same size as input m.
        '''
        c, MPC = self.cFunc.eval_with_derivative(m)
        return MPC*utilityPP(c,gam=self.CRRA)

class MargMargValueFunc2D(HARKobject):
    '''
    A class for representing a marginal marginal value function in models where the
    standard envelope condition of v'(m,H) = u'(c(m,H)) holds (with CRRA utility).
    '''
    distance_criteria = ['cFunc', 'CRRA']

    def __init__(self, cFunc, CRRA):
        '''
        Constructor for a new marginal marginal value function object.
        Parameters
        ----------
        cFunc : function
            A real function representing the marginal value function composed
            with the inverse marginal utility function, defined on market
            resources and the level of habit stock: uP_inv(vPfunc(m,H)).
            Called cFunc because when standard envelope condition applies,
            uP_inv(vPfunc(m,H)) = cFunc(m,H).
        CRRA : float
            Coefficient of relative risk aversion.
        Returns
        -------
        None
        '''
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA

    def __call__(self, m, H):
        '''
        Evaluate the marginal marginal value function at given levels of market
        resources m and habit stock H.
        Parameters
        ----------
        m : float or np.array
            Market resources whose marginal marginal value is to be calculated.
        H : float or np.array
            Habit stock levels whose marginal marginal value is to be calculated.
        Returns
        -------
        vPP : float or np.array
            Marginal marginal value of beginning this period with market
            resources m and habit stock H; has same size as inputs.
        '''
        c = self.cFunc(m, H)
        MPC = self.cFunc.derivativeX(m, H)
        # see interpolation.py for derevativeX
        return MPC*utilityPP(c, gam=self.CRRA)/((H**Hgamma)**(1-self.CRRA))


# =====================================================================
# === Classes and functions that solve consumption-saving models ===
# =====================================================================

class ConsPerfForesightHabitSolver(object):
    '''
    A class for solving a one period perfect foresight consumption-saving problem with habit formation.
    An instance of this class is created by the function solvePerfForesightHabit in each period.
    '''
    def __init__(self,solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac,BoroCnstArt,MaxKinks, Hgamma, Hlambda):
        '''
        Constructor for a new ConsPerfForesightSolver.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one-period problem.
        DiscFac : float
            Intertemporal discount factor for future utility.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the next period.
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree : float
            Risk free interest factor on end-of-period assets.
        PermGroFac : float
            Expected permanent income growth factor at the end of this period.
        BoroCnstArt : float or None
            Artificial borrowing constraint, as a multiple of permanent income.
            Can be None, indicating no artificial constraint.
        MaxKinks : int
            Maximum number of kink points to allow in the consumption function;
            additional points will be thrown out.  Only relevant in infinite
            horizon model with artificial borrowing constraint.
        Hgamma : float
            Importance of habits
        Hlambda : float
            Speed with which habits ‘catch up’ to consumption

        Returns:
        ----------
        None
        '''
        # We ask that HARK users define single-letter variables they use in a dictionary
        # attribute called notation. Do that first.
        self.notation = {'a': 'assets after all actions',
                         'm': 'market resources at decision time',
                         'c': 'consumption',
                         'H': 'habit stocks'}
        self.assignParameters(solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac,BoroCnstArt,MaxKinks,Hgamma, Hlambda)

    def assignParameters(self,solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac,BoroCnstArt,MaxKinks,Hgamma, Hlambda):
        '''
        Saves necessary parameters as attributes of self for use by other methods.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        DiscFac : float
            Intertemporal discount factor for future utility.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree : float
            Risk free interest factor on end-of-period assets.
        PermGroFac : float
            Expected permanent income growth factor at the end of this period.
        BoroCnstArt : float or None
            Artificial borrowing constraint, as a multiple of permanent income.
            Can be None, indicating no artificial constraint.
        MaxKinks : int
            Maximum number of kink points to allow in the consumption function;
            additional points will be thrown out.
        Hgamma : float
            Importance of habits
        Hlambda : float
            Speed with which habits ‘catch up’ to consumption
            
        Returns
        -------
        None
        '''
        self.solution_next  = solution_next
        self.DiscFac        = DiscFac
        self.LivPrb         = LivPrb
        self.CRRA           = CRRA
        self.Rfree          = Rfree
        self.PermGroFac     = PermGroFac
        self.MaxKinks       = MaxKinks
        self.BoroCnstArt    = BoroCnstArt
        self.Hgamma         = Hgamma
        self.Hlambda        = Hlambda

    def defUtilityFuncs(self):
        '''
        Defines CRRA utility function for this period (and its derivatives),
        saving them as attributes of self for other methods to use.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.u   = lambda c : utility((c/(H**Hgamma)),gam=self.CRRA)  # utility function
        self.uP  = lambda c : utilityP(c,gam=self.CRRA)/((H**Hgamma)**(1-self.CRRA)) # marginal utility function
        self.uPP = lambda c : utilityPP(c,gam=self.CRRA)/((H**Hgamma)**(1-self.CRRA))# marginal marginal utility function

    def defValueFuncs(self):
        '''
        Defines the value and marginal value functions for this period.
        Uses the fact that for a perfect foresight CRRA utility problem,
        if the MPC in period t is :math:`\kappa_{t}`, and relative risk 
        aversion :math:`\rho`, then the inverse value vFuncNvrs has a 
        constant slope of :math:`\kappa_{t}^{-\rho/(1-\rho)}` and 
        vFuncNvrs has value of zero at the lower bound of market resources 
        mNrmMin.  See PerfForesightConsumerType.ipynb documentation notebook
        for a brief explanation and the links below for a fuller treatment.
            
        https://econ.jhu.edu/people/ccarroll/public/lecturenotes/consumption/PerfForesightCRRA/#vFuncAnalytical
        https://econ.jhu.edu/people/ccarroll/SolvingMicroDSOPs/#vFuncPF

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # See the PerfForesightConsumerType.ipynb documentation notebook for the derivations
        vFuncNvrsSlope = self.MPCmin**(-self.CRRA/(1.0-self.CRRA)) 
        vFuncNvrs      = BilinearInterp(np.array([self.mNrmMinNow, self.mNrmMinNow + 1.0]), np.arrat([self.HNrmNow, self.HNrmNow + 1.0]), np.array([0.0, vFuncNvrsSlope]))
        self.vFunc     = ValueFunc2D(vFuncNvrs,self.CRRA)
        self.vPfunc    = MargValueFunc2D(self.cFunc,self.CRRA)

    def makePFHcFunc(self):
        '''
        Makes the (linear) consumption function for this period.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # Use a local value of BoroCnstArt to prevent comparing None and float below.
        if self.BoroCnstArt is None:
            BoroCnstArt = -np.inf
        else:
            BoroCnstArt = self.BoroCnstArt
        
        # Calculate human wealth this period
        self.hNrmNow = (self.PermGroFac/self.Rfree)*(self.solution_next.hNrm + 1.0)
        # h_t = Gamma_{t+1}/R * (h_{t+1}+1)
        
        # Calculate the lower bound of the marginal propensity to consume
        PatFacH      = ((self.Rfree*self.DiscFacEff)**(1.0/self.CRRA))/self.Rfree*(self.solution_next.HNrm/self.solution.HNrm)**(self.Hgamma*(1-1/self.CRRA))
        self.MPCmin  = 1.0/(1.0 + PatFacH/self.solution_next.MPCmin)
        
        # Extract the discrete kink points in next period's consumption function;
        # don't take the last one, as it only defines the extrapolation and is not a kink.
        mNrmNext = self.solution_next.cFunc.x_list[:-1]
        cNrmNext = self.solution_next.cFunc.y_list[:-1]
        
        # Calculate the end-of-period asset values that would reach those kink points
        # next period, then invert the first order condition to get consumption. Then
        # find the endogenous gridpoint (kink point) today that corresponds to each kink
        aNrmNow = (self.PermGroFac/self.Rfree)*(mNrmNext-1.0)
        cNrmNow = (self.DiscFacEff*self.Rfree)**(-1./self.CRRA)*(self.PermGroFac*cNrmNext)
        mNrmNow = aNrmNow + cNrmNow
         # Calculate habits
        HNrmNow = (HNrmNext-self.Hlambda*cNrmNow)/(1-self.Hlambda)
        
        # Add an additional point to the list of gridpoints for the extrapolation,
        # using the new value of the lower bound of the MPC.
        mNrmNow = np.append(mNrmNow, mNrmNow[-1] + 1.0)
        cNrmNow = np.append(cNrmNow, cNrmNow[-1] + self.MPCmin)
        HNrmNow = np.append(HNrmNow, HNrmNow[-1] + 1.0)
        # to be checked
        
        # If the artificial borrowing constraint binds, combine the constrained and
        # unconstrained consumption functions.
        if BoroCnstArt > mNrmNow[0]:
            # Find the highest index where constraint binds
            cNrmCnst = mNrmNow - BoroCnstArt
            CnstBinds = cNrmCnst < cNrmNow
            idx = np.where(CnstBinds)[0][-1]
            
            if idx < (mNrmNow.size-1):
                # If it is not the *very last* index, find the critical level
                # of mNrm and HNrm where the artificial borrowing contraint begins to bind.
                d0 = cNrmNow[idx] - cNrmCnst[idx]
                d1 = cNrmCnst[idx+1] - cNrmNow[idx+1]
                m0 = mNrmNow[idx]
                m1 = mNrmNow[idx+1]
                H0 = HNrmNow[idx]
                H1 = HNrmNow[idx+1]
                alpha = d0/(d0 + d1)
                mCrit = m0 + alpha*(m1 - m0)
                HCrit = H0 + alpha*(H1 - H0)
                # Adjust the grids of mNrm and cNrm to account for the borrowing constraint.
                cCrit = mCrit - BoroCnstArt
                mNrmNow = np.concatenate(([BoroCnstArt, mCrit], mNrmNow[(idx+1):]))
                HNrmNow = np.concatenate(([BoroCnstArt, HCrit], HNrmNow[(idx+1):]))
                cNrmNow = np.concatenate(([0., cCrit], cNrmNow[(idx+1):]))
                
            else:
                # If it *is* the very last index, then there are only three points
                # that characterize the consumption function: the artificial borrowing
                # constraint, the constraint kink, and the extrapolation point.
                mXtra = (cNrmNow[-1] - cNrmCnst[-1])/(1.0 - self.MPCmin)
                HXtra = (cNrmNow[-1] - cNrmCnst[-1])/(1.0 - self.MPCmin)
                mCrit = mNrmNow[-1] + mXtra
                HCrit = HNrmNow[-1] + HXtra
                cCrit = mCrit - BoroCnstArt
                mNrmNow = np.array([BoroCnstArt, mCrit, mCrit + 1.0])
                HNrmNow = np.arrat([BoroCnstArt, HCrit, HCrit + 1.0])
                cNrmNow = np.array([0., cCrit, cCrit + self.MPCmin])
                
        # If the mNrm and cNrm grids have become too large, throw out the last
        # kink point, being sure to adjust the extrapolation.
        if mNrmNow.size > self.MaxKinks:
            mNrmNow = np.concatenate((mNrmNow[:-2], [mNrmNow[-3] + 1.0]))
            HNrmNow = np.concatenate((HNrmNow[:-2], [HNrmNow[-3] + 1.0]))
            cNrmNow = np.concatenate((cNrmNow[:-2], [cNrmNow[-3] + self.MPCmin]))
        
        # Construct the consumption function
        
        cFunc_by_Habit = []
        for j in range(Habit_N):
            cNrm_temp = EndOfPrddvdaNvrs[:,j]
            mNrm_temp = aNrmGrid + cNrm_temp
            cFunc_by_Habit.append(LinearInterp(np.insert(mNrm_temp, 0, 0.0), np.insert(cNrm_temp, 0, 0.0)))  
        cFunc_now = LinearInterpOnInterp1D(cFunc_by_Habit, HabitGrid)
        self.cFunc = LinearInterpOnInterp1D(cFunc_by_Habit, HabitGrid)  
        #self.cFunc = BilinearInterp(mNrmNow, HNrmNow, cNrmNow)
        
        # Construct the marginal value of mNrm function
        dvdmFunc_now = MargValueFunc2D(cFunc_now, CRRA)
        
        # If the value function has been requested, construct it now
        if vFuncBool:
            # First, make an end-of-period value function over aNrm and Habit
            EndOfPrdvNvrsFunc = BilinearInterp(EndOfPrdvNvrs, aNrmGrid, HabitGrid)
            # EndOfPrdvNvrs not defined yet
            EndOfPrdvFunc = ValueFunc2D(EndOfPrdvNvrsFunc, CRRA)
        
            # Construct the value function when the agent can adjust his portfolio
            mNrm_temp  = aXtraGrid # Just use aXtraGrid as our grid of mNrm values
            cNrm_temp  = cFuncAdj_now(mNrm_temp)
            aNrm_temp  = mNrm_temp - cNrm_temp
            Share_temp = ShareFuncAdj_now(mNrm_temp)
            v_temp     = u(cNrm_temp) + EndOfPrdvFunc(aNrm_temp, Share_temp)
            vNvrs_temp = n(v_temp)
            vNvrsP_temp= uP(cNrm_temp)*nP(v_temp)
            vNvrsFuncAdj = CubicInterp(
                    np.insert(mNrm_temp,0,0.0),  # x_list
                    np.insert(vNvrs_temp,0,0.0), # f_list
                    np.insert(vNvrsP_temp,0,vNvrsP_temp[0])) # dfdx_list
            vFuncAdj_now = ValueFunc(vNvrsFuncAdj, CRRA) # Re-curve the pseudo-inverse value function
            
            # Construct the value function when the agent *can't* adjust his portfolio
            mNrm_temp  = np.tile(np.reshape(aXtraGrid, (aXtraGrid.size, 1)), (1, Habit_N))
            # Habit_N is habitgridsize, to be added
            HNrm_temp  = np.tile(np.reshape(HabitGrid, (1, Habit_N)), (aXtraGrid.size, 1))
            cNrm_temp  = cFunc_now(mNrm_temp, Habit_temp)
            aNrm_temp  = mNrm_temp - cNrm_temp
            v_temp     = u(cNrm_temp) + EndOfPrdvFunc(aNrm_temp, Habit_temp)
            vNvrs_temp = n(v_temp)
            # i guess n is u^-1
            vNvrsP_temp= (uP(cNrm_temp))/(HNrm_temp**(Hgamma*(1-self.CRRA)))*nP(v_temp)
            # nP may need to be adjusted
            vNvrsFunc_by_Habit = []
            for j in range(Share_N):
                vNvrsFunc_by_Habit.append(CubicInterp(
                        np.insert(mNrm_temp[:,0],0,0.0),  # x_list
                        np.insert(vNvrs_temp[:,j],0,0.0), # f_list
                        np.insert(vNvrsP_temp[:,j],0,vNvrsP_temp[j,0]))) #dfdx_list
            vNvrsFunc = LinearInterpOnInterp1D(vNvrsFunc_by_Habit, HabitGrid)
            vFunc_now = ValueFunc2D(vNvrsFunc, CRRA)
        
        else: # If vFuncBool is False, fill in dummy values
            vFunc_now = None
            
        
        
        
        # Calculate the upper bound of the MPC as the slope of the bottom segment.
        self.MPCmax = (cNrmNow[1] - cNrmNow[0]) / (mNrmNow[1] - mNrmNow[0])
        # not sure used where, and may need to modify as gradient.
        
        # Add two attributes to enable calculation of steady state market resources.
        self.ExIncNext = 1.0 # Perfect foresight income of 1
        self.mNrmMinNow = mNrmNow[0] # Relabeling for compatibility with addSSmNrm
        
        
    def addSSmNrm(self,solution):
        '''
        Finds steady state (normalized) market resources and adds it to the
        solution.  This is the level of market resources such that the expectation
        of market resources in the next period is unchanged.  This value doesn't
        necessarily exist.

        Parameters
        ----------
        solution : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.
        Returns
        -------
        solution : ConsumerSolution
            Same solution that was passed, but now with the attribute mNrmSS.
        '''
        # Make a linear function of all combinations of c and m that yield mNext = mNow
        mZeroChangeFunc = lambda m : (1.0-self.PermGroFac/self.Rfree)*m + (self.PermGroFac/self.Rfree)*self.ExIncNext

        # Find the steady state level of market resources
        searchSSfunc = lambda m : solution.cFunc(m) - mZeroChangeFunc(m) # A zero of this is SS market resources
        m_init_guess = self.mNrmMinNow + self.ExIncNext # Minimum market resources plus next income is okay starting guess
        try:
            mNrmSS = newton(searchSSfunc,m_init_guess)
        except:
            mNrmSS = None

        # Add mNrmSS to the solution and return it
        solution.mNrmSS = mNrmSS
        return solution

    def solve(self):
        '''
        Solves the one period perfect foresight consumption-saving problem.

        Parameters
        ----------
        None

        Returns
        -------
        solution : ConsumerSolution
            The solution to this period's problem.
        '''
        self.defUtilityFuncs()
        self.DiscFacEff = self.DiscFac*self.LivPrb
        self.makePFcFunc()
        self.defValueFuncs()
        #solution = ConsumerSolution(cFunc=self.cFunc, vFunc=self.vFunc, vPfunc=self.vPfunc,
                                    #mNrmMin=self.mNrmMinNow, hNrm=self.hNrmNow,
                                    #MPCmin=self.MPCmin, MPCmax=self.MPCmax, HNrm=self.HNrm)
        
        
        solution = ConsumerSolution(
                    cFunc = cFunc_now,
                    vPfunc = vPfunc_now,
                    vFunc = vFunc_now,
                    dvdmFunc = dvdmFunc_now,
                    #dvdHFunc = dvdHFunc_now, to be added
                    vFunc = vFunc_now
    )
        solution = self.addSSmNrm(solution)
        return solution
    
# =====================================================================
# === Classes and functions that solve consumption-saving models ===
# =====================================================================

class PerfForesightConsumerHabitType(AgentType):
    '''
    A perfect foresight consumer type with habits who has no uncertainty other than mortality.
    His problem is defined by a coefficient of relative risk aversion, intertemporal
    discount factor, interest factor, an artificial borrowing constraint (maybe)
    and time sequences of the permanent income growth rate and survival probability.
    '''
    # Define some universal values for all consumer types
    # Consume all market resources: c_T = m_T
    cFunc_terminal = IdentityFunction(i=dim=0, n_dims=2) #0 is the first dimension, which is market resources
    # Value function is the utility from consuming market resources comparing to multiplicative habits
    vFunc_terminal = ValueFunc2D((cFunc/(HNrm**Hgamma), self.CRRA)    
    HNrmNow = (HNrmNext-self.Hlambda*cNrmNow)/(1-self.Hlambda)
    # What is HNrm_terminal?
    solution_terminal_   = ConsumerSolution(cFunc = cFunc_terminal_,
                                            vFunc = vFunc_terminal_, mNrmMin=0.0, hNrm=0.0,
                                            MPCmin=1.0, MPCmax=1.0)
    time_vary_ = ['LivPrb','PermGroFac']
    time_inv_  = ['CRRA','Rfree','DiscFac','MaxKinks','BoroCnstArt']
    poststate_vars_ = ['aNrmNow','pLvlNow']
    shock_vars_ = []

    def __init__(self,
                 cycles=1,
                 verbose=1,
                 quiet=False,
                 **kwds):
        '''
        Instantiate a new consumer type with given data.
        See init_perfect_foresight for a dictionary of
        the keywords that should be passed to the constructor.
        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.
        Returns
        -------
        None
        '''

        params = init_perfect_foresight.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic AgentType
        AgentType.__init__(self,solution_terminal=deepcopy(self.solution_terminal_),
                           cycles=cycles,
                           pseudo_terminal=False,**kwds)

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary      = deepcopy(self.time_vary_)
        self.time_inv       = deepcopy(self.time_inv_)
        self.poststate_vars = deepcopy(self.poststate_vars_)
        self.shock_vars     = deepcopy(self.shock_vars_)
        self.verbose        = verbose
        self.quiet          = quiet
        self.solveOnePeriod = makeOnePeriodOOSolver(ConsPerfForesightSolver) 
        set_verbosity_level((4-verbose)*10)

    def preSolve(self):
        self.updateSolutionTerminal() # Solve the terminal period problem
        
        # Fill in BoroCnstArt and MaxKinks if they're not specified or are irrelevant.
        if not hasattr(self,'BoroCnstArt'): # If no borrowing constraint specified...
            self.BoroCnstArt = None       # ...assume the user wanted none
        if not hasattr(self,'MaxKinks'):
            if self.cycles > 0: # If it's not an infinite horizon model...
                self.MaxKinks = np.inf  # ...there's no need to set MaxKinks
            elif self.BoroCnstArt is None: # If there's no borrowing constraint...
                self.MaxKinks = np.inf # ...there's no need to set MaxKinks
            else:
                raise(AttributeError('PerfForesightConsumerType requires the attribute MaxKinks to be specified when BoroCnstArt is not None and cycles == 0.'))

            
    def checkRestrictions(self):
        """
        A method to check that various restrictions are met for the model class.
        """
        if self.DiscFac < 0:
            raise Exception('DiscFac is below zero with value: ' + str(self.DiscFac))

        return

    def updateSolutionTerminal(self):
        '''
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.
        Parameters
        ----------
        none
        Returns
        -------
        none
        '''
        self.solution_terminal.vFunc   = ValueFunc(self.cFunc_terminal_,self.CRRA)
        self.solution_terminal.vPfunc  = MargValueFunc(self.cFunc_terminal_,self.CRRA)
        self.solution_terminal.vPPfunc = MargMargValueFunc(self.cFunc_terminal_,self.CRRA)

    def unpackcFunc(self):
        '''
        "Unpacks" the consumption functions into their own field for easier access.
        After the model has been solved, the consumption functions reside in the
        attribute cFunc of each element of ConsumerType.solution.  This method
        creates a (time varying) attribute cFunc that contains a list of consumption
        functions.
        Parameters
        ----------
        none
        Returns
        -------
        none
        '''
        self.cFunc = []
        for solution_t in self.solution:
            self.cFunc.append(solution_t.cFunc)
        self.addToTimeVary('cFunc')

    def initializeSim(self):
        self.PlvlAggNow = 1.0
        self.PermShkAggNow = self.PermGroFacAgg # This never changes during simulation
        AgentType.initializeSim(self)


    def simBirth(self,which_agents):
        '''
        Makes new consumers for the given indices.  Initialized variables include aNrm and pLvl, as
        well as time variables t_age and t_cycle.  Normalized assets and permanent income levels
        are drawn from lognormal distributions given by aNrmInitMean and aNrmInitStd (etc).
        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".
        Returns
        -------
        None
        '''
        # Get and store states for newly born agents
        N = np.sum(which_agents) # Number of new consumers to make
        self.aNrmNow[which_agents] = Lognormal(
            mu=self.aNrmInitMean,
            sigma=self.aNrmInitStd,
            seed=self.RNG.randint(0,2**31-1)).draw(N)
        pLvlInitMeanNow = self.pLvlInitMean + np.log(self.PlvlAggNow) # Account for newer cohorts having higher permanent income
        self.pLvlNow[which_agents] = Lognormal(
            pLvlInitMeanNow,
            self.pLvlInitStd,
            seed=self.RNG.randint(0,2**31-1)).draw(N)
        self.t_age[which_agents]   = 0 # How many periods since each agent was born
        self.t_cycle[which_agents] = 0 # Which period of the cycle each agent is currently in
        return None

    def simDeath(self):
        '''
        Determines which agents die this period and must be replaced.  Uses the sequence in LivPrb
        to determine survival probabilities for each agent.
        Parameters
        ----------
        None
        Returns
        -------
        which_agents : np.array(bool)
            Boolean array of size AgentCount indicating which agents die.
        '''
        # Determine who dies
        DiePrb_by_t_cycle = 1.0 - np.asarray(self.LivPrb)
        DiePrb = DiePrb_by_t_cycle[self.t_cycle-1] # Time has already advanced, so look back one
        DeathShks = Uniform(
            seed=self.RNG.randint(0,2**31-1)).draw(N=self.AgentCount)
        which_agents = DeathShks < DiePrb
        if self.T_age is not None: # Kill agents that have lived for too many periods
            too_old = self.t_age >= self.T_age
            which_agents = np.logical_or(which_agents,too_old)
        return which_agents

    def getShocks(self):
        '''
        Finds permanent and transitory income "shocks" for each agent this period.  As this is a
        perfect foresight model, there are no stochastic shocks: PermShkNow = PermGroFac for each
        agent (according to their t_cycle) and TranShkNow = 1.0 for all agents.
        Parameters
        ----------
        None
        Returns
        -------
        None
        '''
        PermGroFac = np.array(self.PermGroFac)
        self.PermShkNow = PermGroFac[self.t_cycle-1] # cycle time has already been advanced
        self.TranShkNow = np.ones(self.AgentCount)

    def getRfree(self):
        '''
        Returns an array of size self.AgentCount with self.Rfree in every entry.
        Parameters
        ----------
        None
        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        '''
        RfreeNow = self.Rfree*np.ones(self.AgentCount)
        return RfreeNow

    def getStates(self):
        '''
        Calculates updated values of normalized market resources and permanent income level for each
        agent.  Uses pLvlNow, aNrmNow, PermShkNow, TranShkNow.
        Parameters
        ----------
        None
        Returns
        -------
        None
        '''
        pLvlPrev = self.pLvlNow
        aNrmPrev = self.aNrmNow
        RfreeNow = self.getRfree()

        # Calculate new states: normalized market resources and permanent income level
        self.pLvlNow = pLvlPrev*self.PermShkNow # Updated permanent income level
        self.PlvlAggNow = self.PlvlAggNow*self.PermShkAggNow # Updated aggregate permanent productivity level
        ReffNow      = RfreeNow/self.PermShkNow # "Effective" interest factor on normalized assets
        self.bNrmNow = ReffNow*aNrmPrev         # Bank balances before labor income
        self.mNrmNow = self.bNrmNow + self.TranShkNow # Market resources after income
        return None

    def getControls(self):
        '''
        Calculates consumption for each consumer of this type using the consumption functions.
        Parameters
        ----------
        None
        Returns
        -------
        None
        '''
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow  = np.zeros(self.AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cNrmNow[these], MPCnow[these] = self.solution[t].cFunc.eval_with_derivative(self.mNrmNow[these])
        self.cNrmNow = cNrmNow
        self.MPCnow = MPCnow
        return None

    def getPostStates(self):
        '''
        Calculates end-of-period assets for each consumer of this type.
        Parameters
        ----------
        None
        Returns
        -------
        None
        '''
        self.aNrmNow = self.mNrmNow - self.cNrmNow
        self.aLvlNow = self.aNrmNow*self.pLvlNow   # Useful in some cases to precalculate asset level
        return None

    def checkCondition(self,
                       name,
                       test,
                       messages,
                       verbose,
                       verbose_messages=None):
        """
        Checks one condition.
        Parameters
        ----------
        name : string
             Name for the condition.
        test : function(self -> boolean)
             A function (of self) which tests the condition
        messages : dict{boolean : string}
            A dictiomary with boolean keys containing values
            for messages to print if the condition is
            true or false.
        verbose_messages : dict{boolean : string}
            (Optional) A dictiomary with boolean keys containing values
            for messages to print if the condition is
            true or false under verbose printing.
        """
        self.conditions[name] = test(self)
        set_verbosity_level((4-verbose)*10)
        _log.info(messages[self.conditions[name]].format(self))
        if verbose_messages:
            _log.debug(verbose_messages[self.conditions[name]].format(self))


    def checkAIC(self, verbose=None):
        '''
        Evaluate and report on the Absolute Impatience Condition
        '''
        name = "AIC"
        test = lambda agent : agent.thorn < 1

        messages = {
            True:  "The value of the absolute impatience factor (APF) for the supplied parameter values satisfies the Absolute Impatience Condition.",
            False: "The given type violates the Absolute Impatience Condition with the supplied parameter values; the APF is {0.thorn}"}
        verbose_messages = {
            True :  "  Because the APF < 1, the absolute amount of consumption is expected to fall over time.",
            False : "  Because the APF > 1, the absolute amount of consumption is expected to grow over time."
        }
        verbose = self.verbose if verbose is None else verbose
        self.checkCondition(name, test, messages, verbose, verbose_messages)

    def checkGICPF(self, verbose=None):
        '''
        Evaluate and report on the Growth Impatience Condition for the Perfect Foresight model
        '''
        name = "GICPF"

        self.GPFPF = self.thorn/self.PermGroFac[0]

        test = lambda agent : agent.GPFPF < 1

        messages = {
            True :  'The value of the Growth Patience Factor for the supplied parameter values satisfies the Perfect Foresight Growth Impatience Condition.',
            False : 'The value of the Growth Patience Factor for the supplied parameter values fails the Perfect Foresight Growth Impatience Condition; the GPFPF is: {0.GPFPF}',
        }

        verbose_messages = {
            True:   '  Therefore, for a perfect foresight consumer, the ratio of individual wealth to permanent income will fall indefinitely.',
            False:  '  Therefore, for a perfect foresight consumer, the ratio of individual wealth to permanent income is expected to grow toward infinity.',
        }
        verbose = self.verbose if verbose is None else verbose
        self.checkCondition(name, test, messages, verbose, verbose_messages)

    def checkRIC(self, verbose=None):
        '''
        Evaluate and report on the Return Impatience Condition
        '''

        self.RPF = self.thorn/self.Rfree

        name = "RIC"
        test = lambda agent: self.RPF < 1
        
        messages = {
            True :  'The value of the Return Patience Factor for the supplied parameter values satisfies the Return Impatience Condition.',
            False : 'The value of the Return Patience Factor for the supplied parameter values fails the Return Impatience Condition; the factor is {0.RIF}'
        }

        verbose_messages = {
            True :  '  Therefore, the limiting consumption function is not c(m)=0 for all m',
            False : '  Therefore, the limiting consumption function is c(m)=0 for all m'
        }
        verbose = self.verbose if verbose is None else verbose
        self.checkCondition(name, test, messages, verbose,verbose_messages)

    def checkFHWC(self, verbose=None):
        '''
        Evaluate and report on the Finite Human Wealth Condition
        '''

        self.FHWF = self.PermGroFac[0]/self.Rfree
        self.cNrmPDV = 1.0/(1.0-self.thorn/self.Rfree)

        name = "FHWC"
        test = lambda agent: self.FHWF < 1
        
        messages = {
            True :  'The Finite Human wealth factor value for the supplied parameter values satisfies the Finite Human Wealth Condition.',
            False : 'The given type violates the Finite Human Wealth Condition; the Finite Human wealth factor value {0.FHWF}',
        }

        verbose_messages = {
            True :  '  Therefore, the limiting consumption function is not c(m)=Infinity\nand human wealth normalized by permanent income is {0.hNrm}\nand the PDV of future consumption growth is {0.cNrmPDV}',
            False : '  Therefore, the limiting consumption function is c(m)=Infinity for all m'
        }
        verbose = self.verbose if verbose is None else verbose
        self.checkCondition(name, test, messages, verbose)

    def checkConditions(self, verbose=None):
        '''
        This method checks whether the instance's type satisfies the
        Absolute Impatience Condition (AIC), 
        the Return Impatience Condition (RIC),
        the Finite Human Wealth Condition (FHWC) and the perfect foresight 
        model's version of the Finite Value of the Growth Impatience Condition (GICPF) and 
        Autarky Condition (FVACPF). Depending on the configuration of parameter values, some 
        combination of these conditions must be satisfied in order for the problem to have 
        a nondegenerate solution. To check which conditions are required, in the verbose mode
        a reference to the relevant theoretical literature is made.
        Parameters
        ----------
        verbose : boolean
            Specifies different levels of verbosity of feedback. When False, it only reports whether the
            instance's type fails to satisfy a particular condition. When True, it reports all results, i.e.
            the factor values for all conditions.
        Returns
        -------
        None
        '''
        self.conditions = {}

        self.violated = False

        # This method only checks for the conditions for infinite horizon models
        # with a 1 period cycle. If these conditions are not met, we exit early.
        if self.cycles!=0 or self.T_cycle > 1:
            return

        self.thorn = (self.Rfree*self.DiscFac*self.LivPrb[0])**(1/self.CRRA)

        verbose = self.verbose if verbose is None else verbose
        self.checkAIC(verbose)
        self.checkGICPF(verbose)
        self.checkRIC(verbose)
        self.checkFHWC(verbose)

        if hasattr(self,'BoroCnstArt') and self.BoroCnstArt is not None:
            self.violated = not self.conditions['RIC']
        else:
            self.violated = not self.conditions['RIC'] or not self.conditions['FHWC'] 
        

# Make a dictionary to specify an idiosyncratic income shocks consumer
init_idiosyncratic_shocks = dict(init_perfect_foresight,
                                 **{
    # assets above grid parameters
    'aXtraMin': 0.001,      # Minimum end-of-period "assets above minimum" value
    'aXtraMax': 20,         # Maximum end-of-period "assets above minimum" value
    'aXtraNestFac': 3,      # Exponential nesting factor when constructing "assets above minimum" grid
    'aXtraCount': 48,       # Number of points in the grid of "assets above minimum"
    'aXtraExtra': [None],   # Some other value of "assets above minimum" to add to the grid, not used
    # Income process variables
    'PermShkStd': [0.1],    # Standard deviation of log permanent income shocks
    'PermShkCount': 7,      # Number of points in discrete approximation to permanent income shocks
    'TranShkStd': [0.1],    # Standard deviation of log transitory income shocks
    'TranShkCount': 7,      # Number of points in discrete approximation to transitory income shocks
    'UnempPrb': 0.05,       # Probability of unemployment while working
    'UnempPrbRet': 0.005,   # Probability of "unemployment" while retired
    'IncUnemp': 0.3,        # Unemployment benefits replacement rate
    'IncUnempRet': 0.0,     # "Unemployment" benefits when retired
    'BoroCnstArt': 0.0,     # Artificial borrowing constraint; imposed minimum level of end-of period assets
    'tax_rate': 0.0,        # Flat income tax rate
    'T_retire': 0, # Period of retirement (0 --> no retirement)
    'vFuncBool': False,     # Whether to calculate the value function during solution
    'CubicBool': False,     # Use cubic spline interpolation when True, linear interpolation when False
})