"""
TODO : documment me

Pure Python version of CG ( basically the scipy implementation)
"""


__docformat__ = 'restructedtext en'
__authors__ = ( "Razvan Pascanu " )
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


import numpy, numpy as np
import theano.tensor as TT, theano
import time
import scipy.optimize.linesearch as ls

from numpy import atleast_1d, eye, mgrid, argmin, zeros, shape, \
     squeeze, vectorize, asarray, absolute, sqrt, Inf, asfarray, isinf


abs = absolute
import __builtin__
pymin = __builtin__.min
pymax = __builtin__.max
__version__="0.7"
_epsilon = sqrt(numpy.finfo(float).eps)


def vecnorm(x, ord=2):
    if ord == Inf:
        return numpy.amax(abs(x))
    elif ord == -Inf:
        return numpy.amin(abs(x))
    else:
        return numpy.sum(abs(x)**ord,axis=0)**(1.0/ord)


class wrap_function_cls(object):
    def __init__(self,fn, args,ncalls):
        self.fn = fn
        self.ncalls = ncalls
        self.args = args
        self.idx = 0
    def set_idx(self, idx):
        self.idx = idx

    def __call__(self,x):
        self.ncalls[self.idx] += 1
        return self.fn(x, *self.args)


def wrap_function(function, args):
    ncalls = [0,0,0,0,0]
    return ncalls, wrap_function_cls(function, args, ncalls)

def py_cubicmin(a,fa,fpa,b,fb,c,fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.

    If no minimizer can be found return None

    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

    C = fpa
    D = fa
    db = b-a
    dc = c-a
    if (db == 0) or (dc == 0) or (b==c): return None
    denom = (db*dc)**2 * (db-dc)
    d1 = np.empty((2,2))
    d1[0,0] = dc**2
    d1[0,1] = -db**2
    d1[1,0] = -dc**3
    d1[1,1] = db**3
    [A,B] = np.dot(d1, np.asarray([fb-fa-C*db,fc-fa-C*dc]).flatten())
    A /= denom
    B /= denom
    radical = B*B-3*A*C
    if radical < 0:  return None
    if (A == 0): return None
    xmin = a + (-B + np.sqrt(radical))/(3*A)
    return xmin

def py_quadmin(a,fa,fpa,b,fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa,

    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    D = fa
    C = fpa
    db = b-a*1.0
    if (db==0): return None
    B = (fb-D-C*db)/(db*db)
    if (B <= 0): return None
    xmin = a  - C / (2.0*B)
    return xmin

def py_zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
          phi, derphi, phi0, derphi0, c1, c2):
    """
    Part of the optimization algorithm in `scalar_search_wolfe2`.
    """

    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0
    while 1:

        # interpolate to find a trial step length between a_lo and
        # a_hi Need to choose interpolation here.  Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside of the interval bounded by a_lo or a_hi
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection

        dalpha = a_hi-a_lo;
        if dalpha < 0: a,b = a_hi,a_lo
        else: a,b = a_lo, a_hi

        # minimizer of cubic interpolant
        # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
        #
        # if the result is too close to the end points (or out of the
        # interval) then use quadratic interpolation with phi_lo,
        # derphi_lo and phi_hi if the result is stil too close to the
        # end points (or out of the interval) then use bisection

        if (i > 0):
            cchk = delta1*dalpha
            a_j = py_cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)
        if (i==0) or (a_j is None) or (a_j > b-cchk) or (a_j < a+cchk):
            qchk = delta2*dalpha
            a_j = py_quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                a_j = a_lo + 0.5*dalpha
        # Check new value of a_j

        phi_aj = phi(a_j)
        if (phi_aj > phi0 + c1*a_j*derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j,3)
            if abs(derphi_aj) <= -c2*derphi0:
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj*(a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j

            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if (i > maxiter):
            a_star = a_j
            val_star = phi_aj
            valprime_star = None
            break


    return a_star, val_star, valprime_star

def py_scalar_search_wolfe2(phi, derphi=None, phi0=None,
                         old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9, amax=50):
    """Find alpha that satisfies strong Wolfe conditions.

    alpha > 0 is assumed to be a descent direction.

    Parameters
    ----------
    phi : callable f(x,*args)
        Objective scalar function.

    derphi : callable f'(x,*args), optional
        Objective function derivative (can be None)
    phi0 : float, optional
        Value of phi at s=0
    old_phi0 : float, optional
        Value of phi at previous point
    derphi0 : float, optional
        Value of derphi at s=0
    args : tuple
        Additional arguments passed to objective function.
    c1 : float
        Parameter for Armijo condition rule.
    c2 : float
        Parameter for curvature condition rule.

    Returns
    -------
    alpha_star : float
        Best alpha
    phi_star
        phi at alpha_star
    phi0
        phi at 0
    derphi_star
        derphi at alpha_star

    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions.  See Wright and Nocedal, 'Numerical Optimization',
    1999, pg. 59-60.

    For the zoom phase it uses an algorithm by [...].

    """

    if phi0 is None:
        phi0 = phi(0.,1)

    if derphi0 is None and derphi is not None:
        derphi0 = derphi(0.,1)

    alpha0 = 0
    if old_phi0 is not None:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
    else:
        alpha1 = 1.0

    if alpha1 < 0:
        alpha1 = 1.0

    if alpha1 == 0:
        # This shouldn't happen. Perhaps the increment has slipped below
        # machine precision?  For now, set the return variables skip the
        # useless while loop, and raise warnflag=2 due to possible imprecision.
        alpha_star = None
        phi_star = phi0
        phi0 = old_phi0
        derphi_star = None

    phi_a1 = phi(alpha1,1)
    #derphi_a1 = derphi(alpha1)  evaluated below

    phi_a0 = phi0
    derphi_a0 = derphi0

    i = 1
    maxiter = 10
    while 1:         # bracketing phase
        if alpha1 == 0:
            break
        if (phi_a1 > phi0 + c1*alpha1*derphi0) or \
           ((phi_a1 >= phi_a0) and (i > 1)):
            alpha_star, phi_star, derphi_star = \
                        py_zoom(alpha0, alpha1, phi_a0,
                              phi_a1, derphi_a0, phi, derphi,
                              phi0, derphi0, c1, c2)

            break

        derphi_a1 = derphi(alpha1,2)
        if (abs(derphi_a1) <= -c2*derphi0):
            alpha_star = alpha1
            phi_star = phi_a1
            derphi_star = derphi_a1
            break

        if (derphi_a1 >= 0):
            alpha_star, phi_star, derphi_star = \
                        py_zoom(alpha1, alpha0, phi_a1,
                              phi_a0, derphi_a1, phi, derphi,
                              phi0, derphi0, c1, c2)
            break

        alpha2 = 2 * alpha1   # increase by factor of two on each iteration
        i = i + 1
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1,2)
        derphi_a0 = derphi_a1

        # stopping test if lower function not found
        if i > maxiter:
            alpha_star = alpha1
            phi_star = phi_a1
            derphi_star = None
            break



    return alpha_star, phi_star, phi0, derphi_star

def py_line_search_wolfe2(f, myfprime, xk, pk, gfk=None, old_fval=None,
                        old_old_fval=None, args=(), c1=1e-4, c2=0.9, amax=50):
    """Find alpha that satisfies strong Wolfe conditions.

    Parameters
    ----------
    f : callable f(x,*args)
        Objective function.
    myfprime : callable f'(x,*args)
        Objective function gradient (can be None).
    xk : ndarray
        Starting point.
    pk : ndarray
        Search direction.
    gfk : ndarray, optional
        Gradient value for x=xk (xk being the current parameter
        estimate). Will be recomputed if omitted.
    old_fval : float, optional
        Function value for x=xk. Will be recomputed if omitted.
    old_old_fval : float, optional
        Function value for the point preceding x=xk
    args : tuple, optional
        Additional arguments passed to objective function.
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.

    Returns
    -------
    alpha0 : float
        Alpha for which ``x_new = x0 + alpha * pk``.
    fc : int
        Number of function evaluations made.
    gc : int
        Number of gradient evaluations made.

    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions.  See Wright and Nocedal, 'Numerical Optimization',
    1999, pg. 59-60.

    For the zoom phase it uses an algorithm by [...].

    """
    fc = [0]
    gc = [0]
    gval = [None]

    def phi(alpha, idx):
        fc[0] += 1
        f.set_idx(idx)
        return f(xk + alpha * pk, *args)

    if isinstance(myfprime, tuple):
        def derphi(alpha):
            fc[0] += len(xk)+1
            eps = myfprime[1]
            fprime = myfprime[0]
            newargs = (f,eps) + args
            gval[0] = fprime(xk+alpha*pk, *newargs)  # store for later use
            return np.dot(gval[0], pk)
    else:
        fprime = myfprime
        def derphi(alpha, idx):
            gc[0] += 1
            fprime.set_idx(idx)
            gval[0] = fprime(xk+alpha*pk, *args)  # store for later use
            return np.dot(gval[0], pk)

    derphi0 = np.dot(gfk, pk)
    #print '>>>> derphi0', derphi0
    alpha_star, phi_star, old_fval, derphi_star = \
                py_scalar_search_wolfe2(phi, derphi, old_fval, old_old_fval,
                                     derphi0, c1, c2, amax)

    if derphi_star is not None:
        # derphi_star is a number (derphi) -- so use the most recently
        # calculated gradient used in computing it derphi = gfk*pk
        # this is the gradient at the next step no need to compute it
        # again in the outer loop.
        derphi_star = gval[0]

    return alpha_star, fc[0], gc[0], phi_star, old_fval, derphi_star

def py_fmin_cg(f, x0, fprime=None, args=(), gtol=1e-5, norm=Inf, epsilon=_epsilon,
              maxiter=None, full_output=0, disp=1, retall=0, callback=None):
    """Minimize a function using a nonlinear conjugate gradient algorithm.

    Parameters
    ----------
    f : callable f(x,*args)
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    fprime : callable f'(x,*args)
        Function which computes the gradient of f.
    args : tuple
        Extra arguments passed to f and fprime.
    gtol : float
        Stop when norm of gradient is less than gtol.
    norm : float
        Order of vector norm to use.  -Inf is min, Inf is max.
    epsilon : float or ndarray
        If fprime is approximated, use this value for the step
        size (can be scalar or vector).
    callback : callable
        An optional user-supplied function, called after each
        iteration.  Called as callback(xk), where xk is the
        current parameter vector.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e. f(xopt) == fopt.
    fopt : float
        Minimum value found, f(xopt).
    func_calls : int
        The number of function_calls made.
    grad_calls : int
        The number of gradient calls made.
    warnflag : int
        1 : Maximum number of iterations exceeded.
        2 : Gradient and/or function calls not changing.
    allvecs : ndarray
        If retall is True (see other parameters below), then this
        vector containing the result at each iteration is returned.

    Other Parameters
    ----------------
    maxiter : int
        Maximum number of iterations to perform.
    full_output : bool
        If True then return fopt, func_calls, grad_calls, and
        warnflag in addition to xopt.
    disp : bool
        Print convergence message if True.
    retall : bool
        Return a list of results at each iteration if True.

    Notes
    -----
    Optimize the function, f, whose gradient is given by fprime
    using the nonlinear conjugate gradient algorithm of Polak and
    Ribiere. See Wright & Nocedal, 'Numerical Optimization',
    1999, pg. 120-122.

    """
    x0 = asarray(x0).flatten()
    if maxiter is None:
        maxiter = len(x0)*200
    func_calls, f = wrap_function(f, args)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)

    f.set_idx(0)
    myfprime.set_idx(0)

    gfk = myfprime(x0)
    k = 0
    N = len(x0)
    xk = x0
    old_fval = f(xk)
    old_old_fval = old_fval + 5000

    if retall:
        allvecs = [xk]
    sk = [2*gtol]
    warnflag = 0
    pk = -gfk
    gnorm = vecnorm(gfk,ord=norm)
    while (gnorm > gtol) and (k < maxiter):
        f.set_idx(1)
        myfprime.set_idx(1)
        deltak = numpy.dot(gfk,gfk)
        # These values are modified by the line search, even if it fails
        old_fval_backup = old_fval
        old_old_fval_backup = old_old_fval

        alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                 py_line_search_wolfe2(f,myfprime,xk,pk,gfk,
                                    old_fval_backup,old_old_fval_backup)
        f.set_idx(1)
        myfprime.set_idx(1)
        if alpha_k is None or alpha_k == 0:
            # This line search also failed to find a better solution.
            warnflag = 2
            break
        xk = xk + alpha_k*pk
        if retall:
            allvecs.append(xk)
        if gfkp1 is None:
            gfkp1 = myfprime(xk)
        yk = gfkp1 - gfk
        beta_k = pymax(0,numpy.dot(yk,gfkp1)/deltak)
        pk = -gfkp1 + beta_k * pk
        gfk = gfkp1
        gnorm = vecnorm(gfk,ord=norm)

        if callback is not None:
            callback(xk)
        k += 1


    if disp or full_output:
        fval = old_fval
    if warnflag == 2:
        if disp:
            print "Warning: Desired error not necessarily achieved due to precision loss"
            print "         Current function value: %f" % fval
            print "         Iterations: %d" % k
            print "         Function evaluations: ", func_calls
            print "         Gradient evaluations: ", grad_calls

    elif k >= maxiter:
        warnflag = 1
        if disp:
            print "Warning: Maximum number of iterations has been exceeded"
            print "         Current function value: %f" % fval
            print "         Iterations: %d" % k
            print "         Function evaluations: ", func_calls
            print "         Gradient evaluations: ", grad_calls
    else:
        if disp:
            print "Optimization terminated successfully."
            print "         Current function value: %f" % fval
            print "         Iterations: %d" % k
            print "         Function evaluations: ",  func_calls
            print "         Gradient evaluations: ",  grad_calls


    if full_output:
        retlist = xk, fval, func_calls[0], grad_calls[0], warnflag
        if retall:
            retlist += (allvecs,)
    else:
        retlist = xk
        if retall:
            retlist = (xk, allvecs)

    return retlist



