# marlev - a Levenberg-Marquardt fitting library
# Copyright (C) 2024 Martin Teichmann
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, see <https://www.gnu.org/licenses/>.
"""
=====================================
Least-Squares Fitting (:mod:`marlev`)
=====================================

This module contains a class-based Levenberg-Marquardt fitter
:class:`Fit` and a function :func:`leastsq` which is a function-based
interface to the same functionality.  """
from functools import wraps
from numpy import (
    arange, array, diagonal, empty, finfo, linspace, maximum, sqrt, zeros,
    zeros_like)
from itertools import count
from scipy.linalg import solve_triangular, qr, qr_multiply, norm
import functools

__all__ = ["Fit", "FitError", "InvalidParameter", "leastsq",
    "Function", "fitfunction"]

try:
    from ._qrsolv import qrsolv
except ImportError:
    def qrsolv(s, diag):
        for j in range(len(diag)):
            if diag[j] == 0:
                continue
            ta = zeros_like(s[0, j:])
            ta[0] = diag[j]
            for k in range(j, len(diag)):
                if ta[0] == 0:
                    ta = ta[1:]
                    continue
                if abs(s[k, k]) > abs(ta[0]):
                    tan = ta[0] / s[k, k]
                    cos = 1 / sqrt(1 + tan * tan)
                    sin = cos * tan
                else:
                    cotan = s[k, k] / ta[0]
                    sin = 1 / sqrt(1 + cotan * cotan)
                    cos = sin * cotan
                tmp = -sin * s[k, k:] + cos * ta
                s[k, k:] = cos * s[k, k:] + sin * ta
                ta = tmp[1:]


class FitError(Exception):
    """ The error raised while fitting

    Parameters
    ----------
    exitcode : int
        The same as in the class `Fit`
    message : str
        A description of the error
    """
    def __init__(self, exitcode, message):
        Exception.__init__(self, message)
        self.exitcode = exitcode

class InvalidParameter(Exception):
    """ Indicate a parameter is invalid

    Parameters
    ----------
    number : int
        number is the number of the parameter, or 
        a list of numbers of parameters, which are erroneous.
        Set to ``None`` (the default) if no specific parameter is a problem.
    """
    def __init__(self, number=None):
        Exception.__init__(self)
        self.number = number


class Status:
    """Cache the calculated results for a set of parameters

    During fitting, we calculate a lot of data for a given set of parameters.
    We store them in objects of this class for later use.
    """
    def __init__(self, x):
        self.parameters = x


def calculator(getter):
    @wraps(getter)
    def wrapper(self):
        try:
            return getattr(self.status, wrapper.__name__)
        except AttributeError:
            ret = getter(self)
            setattr(self.status, wrapper.__name__, ret)
            return ret
    return wrapper


class Fit:
    """ Least-Squares Fitting using Levenberg-Marquardt

    This class contains methods to minimize the sum of the squares 
    of `M` nonlinear functions with `N` parameters by a modification of the
    Levenberg-Marquardt algorithm. The user must overwrite the method
    :meth:`func` that calculates the functions, and, optionally, the method
    :meth:`jacobian` to calculate its partial derivatives.

    Parameters
    ----------

    params : vector, length `N`
        contains the last successful set of parameters during fit.
        Useful if fit failed.
    running : boolean
        if set to False, fit will stop at next iteration. Use this to
        stop the fit from another thread.
    eps : float
        the stepsize used in the forward-difference approximation
    iterations : int
        number of iterations needed in last fit
    nfev : int
        the number of times the function has been evaluated
    fvec : vector, length `M`
        the function value at the last successful parameter optimization
    scale : vector, length `N`
        the scale of the changes of the parameters in each iteration 
    exitcode : int
        the reason the fitting has stopped. Possible values:
        (see also the parameter names of the method :meth:`fit`)

        :0:
            an unknown (typically user) error has occurred
        :1:
            both actual and predicted relative reductions
            in the sum of squares are at most `ftol`.

        :2:
            relative error between two consecutive iterates
            is at most `xtol`.

        :3:
            conditions for 1 and 2 both hold.

        :4:
            the cosine of the angle between function value and any
            column of the Jacobian is at most `gtol` in
            absolute value.

        :5:
            number of function evaluations has reached its limit

        :6:
            `ftol` is too small. no further reduction in
            the sum of squares is possible.

        :7:
            `xtol` is too small. No further improvement in
            the approximate solution `x` is possible.

        :8:
            `gtol` is too small. The function value is orthogonal to the
            columns of the jacobian to machine precision.

        :9:
            number of iterations has reached its limit
    """

    def fjac(self, j):
        r""" Calculate the derivative of the function by one parameter.

        This method, which is used by :meth:`jacobian`, calculates the
        derivative

        .. math::
           \frac{\partial f}{\partial x_j}

        by a forward-difference approximation.

        The user should overwrite this method (or :meth:`jacobian`) if another
        way to calculate the derivative of the function is appropriate.

        Parameters
        ----------

        j : int
            The number of the function's parameter to derive by.

        Returns
        -------
        out : vector, length `M`
            The derivative of the function at `x` by `x[j]`.
        """
        x = self.parameters()
        f0 = self.values()
        tmp = x[j]
        h = sqrt(self.eps) * abs(tmp)
        if h == 0:
            h = sqrt(self.eps)
        for c in [1, -1, 0.1, -0.1]:
            x[j] = tmp + c * h
            try:
                self.nfev += 1
                f1 = self.func(x)
            except InvalidParameter:
                continue
            x[j] = tmp
            return (f1 - f0) / (c * h)
        return zeros_like(f0)

    @calculator
    def parameters(self):
        raise RuntimeError('this should never happen')

    @calculator
    def jacobian(self):
        r""" Calculate the Jacobian matrix

        The Jacobian matrix is a[i, j] with

        .. math::
            a[i, j] = \left.\frac{\partial f_i}{\partial x_j}\right|_x

        This default implementation uses :meth:`fjac` to calculate the
        derivatives. The user should overwrite this method (or
        :meth:`fjac`) if another method to calculate the Jacobian is
        more appropriate.

        Returns
        -------
        ret : matrix, `M` by `N` 
            the Jacobian at `x` """
        fvec = self.values()
        x = self.parameters()
        ret = empty((len(x), len(fvec)), dtype=fvec.dtype)
        for j in range(len(x)):
            ret[j, :] = self.fjac(j)
        return ret.T

    @calculator
    def values(self):
        return self.func(self.parameters())

    @calculator
    def fnorm(self):
        return norm(self.values())

    @calculator
    def qtf(self):
        return self.decompose()[0]

    @calculator
    def r(self):
        return self.decompose()[1]

    @calculator
    def ipvt(self):
        return self.decompose()[2]

    def decompose(self):
        try:
            return self.status.qtf, self.status.r, self.status.ipvt
        except AttributeError:
            ret = qr_multiply(self.jacobian(), self.values(), pivoting=True)
            self.status.qtf, self.status.r, self.status.ipvt = ret
            return ret

    def func(self, x):
        """ The function to be fitted to

        This method is only a placeholder, it is to be overwritten by
        the user with the function to be fitted. Therefore, the following
        is a description of the function that the user has to supply,
        not the actual function which does nothing.

        Parameters
        ----------
        x : vector, length `N`
            the parameters where to calculate the function at

        Returns
        -------
        fvec : vector, length `M`
            the value of the function at `x`. 

        Raises
        ------
        InvalidParameter
            This method may raise the exception `InvalidParameter`, indicating
            that one or more of the values in `x` are not appropriate.
            The user may pass the number of the offending parameter as a 
            parameter to `InvalidParameter`, or a list of numbers in the
            case of multiple offending parameters.
        """
        raise NotImplemented("I don't know which function to fit!")

    def _qrsolv(self, r, ipvt, diag, qtb):
        r""" Solve the Levenberg-Marquart problem. Internal method.

        Given an `M` by `N` matrix `A`, an `N` by `N` diagonal matrix `D`,
        and an `M`-vector `b`, the problem is to determine an `x` which
        solves the system ::

            A x = b,    D x = 0 

        in the least squares sense.

        This subroutine completes the solution of the problem
        if it is provided with the necessary information from the
        QR factorization, with column pivoting, of `A`. That is, if
        `AP = QR`, where `P` is a permutation matrix, `Q` has orthogonal
        columns, and `R` is an upper triangular matrix with diagonal
        elements of nonincreasing magnitude, then `_qrsolv` expects
        the full upper triangle of `R`, the permutation matrix `P`,
        and the first n components of ``Q.T b``. The system
        ``Ax = b, Dx = 0``, is then equivalent to ::

              R z = Q.T b,  P.T D P z = 0,

        where ``x = Pz``. If this system does not have full rank,
        then a least squares solution is obtained. On output `_qrsolv`
        also provides an upper triangular matrix S such that ::

              P.T (A.T A + D D) P = S.T S.

        Parameters
        ----------
        r : ndarray, `N` by `N`
            On input the full upper triangle must contain the full upper
            triangle of the matrix `R`. 

        ipvt : integer array
            defines the permutation matrix `P` such that `A P = Q R`.
            Column `j` of `P` is column `ipvt[j]` of the identity matrix.

        diag : vector, length `N`
            contains the diagonal elements of the matrix `D`.

        qtb : vector, length `N`
            contains the first `N` elements of the vector ``Q.T b``.

        Returns
        -------
        x : vector, length `N`
            contains the least squares solution of the system
            ``A x = b, D x = 0``.

        s : matrix, `N` by `N`
            contains the aforementioned matrix `S`. """
        s = empty((r.shape[0], r.shape[1] + 1), dtype=r.dtype)
        s[:, :-1] = r
        s[:, -1] = qtb
        # Eliminate the diagonal matrix D using a givens rotation.
        qrsolv(s, diag[ipvt])
        nsing = s.diagonal().nonzero()[0][-1] + 1
        x = zeros_like(qtb)
        x[ipvt[:nsing]] = solve_triangular(s[:nsing, :nsing], s[:, -1])
        return x, s[:, :-1]
            
    def _lmpar(self, r, qtb, ipvt, diag, delta, par):
        r""" Calculate the Levenberg-Marquardt parameter. Internal method.

        Given an `M` by `N` matrix `A`, an `N` by `N` nonsingular diagonal
        matrix `D`, an `M`-vector `b`, and a positive number `delta`,
        the problem is to determine a value for the parameter
        `par` such that if `x` solves the system ::

              A x = b,    sqrt(par) D x = 0,

        in the least squares sense, and `dxnorm` is the Euclidean
        norm of `D x`, then either `par` is zero and ::

              10 (dxnorm - delta) <= delta,

        or `par` is positive and

              10 |dxnorm - delta| <= delta.

        This method completes the solution of the problem
        if it is provided with the necessary information from the
        QR factorization, with column pivoting, of `A`. That is, if
        `AP = QR`, where `P` is a permutation matrix, `Q` has orthogonal
        columns, and `R` is an upper triangular matrix with diagonal
        elements of nonincreasing magnitude, then `_lmpar` expects
        the full upper triangle of `R`, the permutation matrix `P`,
        and the first n components of ``Q.T b``. On output
        `_lmpar` also provides an upper triangular matrix `S` such that ::

              P.T (A.T A + par D D) P = S.T S.

        `S` is employed within `_lmpar` and may be of separate interest.

        Only a few iterations are generally needed for convergence
        of the algorithm. If, however, the limit of 10 iterations
        is reached, then the output par will contain the best
        value obtained so far.

        Parameters
        ----------
        r : matrix, `N` by `N`
            On input the full upper triangle must contain the full upper
            triangle of the matrix `R`.

        ipvt : integer array
            defines the permutation matrix `P` such that ``A P = Q R``.
            Column `j` of `P` is column ``ipvt[j]`` of the identity matrix.

        diag : vector, length `N`
            contains the diagonal elements of the matrix `D`.

        qtb : vector, length `N`
            contains the first `N` elements of the vector ``q.T b``.

        delta : float, > 0
            specifies an upper bound on the Euclidean norm of ``D x``.

        par : float, > 0
            contains an initial estimate of the Levenberg-Marquardt
            parameter.

        Returns
        -------
        par : float
            The optimal parameter found
        x : vector, length `N`
            contains the least squares solution of the system 
            ``A x = b,    sqrt(par) D x = 0``,
            for the output `par`. """
        dwarf = finfo(r.dtype).tiny

        # Compute and store in x the Gauss-Newton direction. If the
        # jacobian is rank-deficient, obtain a least squares solution.
        nsing = r.diagonal().nonzero()[0][-1] + 1
        x = zeros_like(qtb)
        x[ipvt[:nsing]] = solve_triangular(r[:nsing, :nsing], qtb[:nsing])

        # Evaluate the function at the origin, and test
        # for acceptance of the Gauss-Newton direction.
        dxnorm = norm(diag * x)
        fp = dxnorm - delta
        if fp < 0.1 * delta:
            return 0, x

        # If the Jacobian is not rank deficient, the Newton
        # step provides a lower bound, parl, for the zero of
        # the function. Otherwise set this bound to zero.
        parl = 0
        if nsing == r.shape[0]:
            parl = fp / delta / (solve_triangular(r, 
                (diag ** 2 * x)[ipvt] / dxnorm, trans=1) ** 2).sum()

        # Calculate an upper bound, paru, for the zero of the function.
        gnorm = norm((qtb @ r) / diag[ipvt])
        paru = gnorm / delta
        if paru == 0:
            paru = dwarf / min(delta, 0.1)

        # If the input par lies outside of the interval (parl, paru),
        # set par to the closer endpoint.
        par = min(max(par, parl), paru)
        if par == 0:
            par = gnorm / dxnorm 

        for iter in count():
            # Evaluate the function at the current value of par.
            if par == 0:
                par = max(dwarf, 0.001 * paru)
            x, s = self._qrsolv(r, ipvt, sqrt(par) * diag, qtb)
            dxnorm = norm(diag * x)
            temp = fp
            fp = dxnorm - delta

            # If the function is small enough, accept the current value
            # of par. Also test for the exceptional cases where parl
            # is zero or the number of iterations has reached 10.
            if (abs(fp) <= 0.1 * delta or (parl == 0 and fp <= temp < 0)
                    or iter == 10):
                return par, x

            # Compute the Newton correction.
            parc = fp / delta / (
                solve_triangular(s, (diag ** 2 * x)[ipvt] / dxnorm, trans=1
                        ) ** 2).sum()
            if fp > 0:
                parl = max(parl, par)
            else:
                paru = min(paru, par)
            par = max(parl, par + parc)

    def plot_intermediate(self, good):
        """ plot intermediate results

        This method, which does nothing in its default implemenation,
        is called by the fitting algorithm for every new parameters tested,
        and may be overwritten to plot the function while fitting.

        Parameters
        ----------
        good : boolean
            Whether this estimation is better (True) or worse (False) than
            the last estimate. """
        pass

    def fit(self, x, ftol=1.49012e-8, xtol=1.49012e-8, gtol=0.0, maxfev=None,
                epsfcn=None, factor=100, diagin=None, iterations=100):
        """ The Levenberg-Marquardt algorithm

        This method performs the actual Levenberg-Marquardt minimization
        this class is written for.

        Parameters
        ----------
        x : vector, length N
            an initial estimate of the solution vector

        ftol : float, >= 0, optional
            Termination occurs when both the actual and predicted
            relative reductions in the sum of squares are at most
            `ftol`.  Therefore, `ftol` measures the relative error
            desired in the sum of squares.
     
        xtol : float, >= 0, optional
            Termination occurs when the relative error between two
            consecutive iterates is at most `xtol`. Therefore, `xtol`
            measures the relative error desired in the approximate
            solution.
     
        gtol : float, >= 0, optional
            Termination occurs when the cosine of the angle between
            the function and any column of the jacobian is at most
            `gtol` in absolute value. Therefore, `gtol` measures the
            orthogonality desired between the function vector and the
            columns of the Jacobian.
     
        iterations : int, optional
            is the maximum number of iterations.
     
        factor : float, > 0, optional
            used in determining the initial step bound. this bound is
            set to the product of factor and the Euclidean norm of
            `diag * x` if nonzero, or else to factor itself. In most
            cases factor should lie in the interval 0.1 < x < 100.
            100 is a generally recommended value.

        diagin : vector, length N, optional
            contains positive entries that serve as multiplicative
            scale factors for the variables.  

        epsfcn : float, > 0, optional
            is an variable used in determining a suitable step length
            for the forward-difference approximation. This approximation
            assumes that the relative errors in the functions are of the
            order of epsfcn. """
        laststatus = self.status = Status(1. * array(x))
        self.nfev = 1
        par = 0
        epsmch = finfo(self.status.parameters.dtype).eps
        if epsfcn is None:
            epsfcn = epsmch
        self.eps = epsfcn
        self.running = True
        self.exitcode = 0
        self.message = "an unknown (typically user) error has occurred"

        for self.iterations in range(iterations):
            fjac = self.jacobian()
            
            fjacnorm = sqrt((fjac ** 2).sum(0))
            qtf, r, ipvt = self.decompose()

            if self.iterations == 0:
                if diagin is None:
                    self.scale = fjacnorm.copy()
                    self.scale[self.scale == 0] = 1
                else:
                    self.scale = diagin
                xnorm = norm(self.scale * x)
                delta = factor * xnorm
                if delta == 0:
                    delta = factor 
            fjacnorm[fjacnorm == 0] = -1

            gnorm = (abs((r.T @ qtf) / self.fnorm()) / fjacnorm[ipvt]).max()
            if gnorm <= gtol:
                self.message = "the cosine of the angle between function " \
                    "value and any column of the Jacobian is at most `gtol` " \
                    "in absolute value."
                self.exitcode = 4
                return self.status.parameters

            if diagin is None: 
                self.scale = maximum(self.scale, fjacnorm) 
            ratio = 0
            while ratio < 0.0001:
                # determine the LM parameter, and calculate function
                # at new position. If that fails, reduce step bound
                par, p = self._lmpar(r, qtf, ipvt, self.scale, delta, par)
                self.status = Status(laststatus.parameters - p)
                pnorm = norm(self.scale * p)
                try:
                    self.nfev += 1
                    testf = self.values()
                except InvalidParameter as error:
                    if error.number is not None:
                        self.scale[error.number] *= 2
                    else:
                        delta = 0.5 * min(delta, 10 * pnorm)
                        par *= 2
                    if self.running:
                        continue
                    else:
                        return self.status.parameters

                # adjust the initial step bound
                if self.iterations == 0:
                    delta = min(delta, pnorm)

                # calculate the metrics for convergence
                actual_reduction = -1
                if 0.1 * self.fnorm() < laststatus.fnorm:
                    actual_reduction = \
                        1 - (self.fnorm() / laststatus.fnorm) ** 2
                temp1 = norm(r @ p[ipvt]) / laststatus.fnorm
                temp2 = sqrt(par) * pnorm / laststatus.fnorm
                predicted_reduction = temp1 ** 2 + 2 * temp2 ** 2
                directional_deriv = -(temp1 ** 2 + temp2 ** 2)
                ratio = (actual_reduction / predicted_reduction
                    if predicted_reduction != 0 else 0)

                # update the step bound
                if ratio <= 0.25:
                    if actual_reduction > 0:
                        temp = 0.5
                    else:
                        temp = 0.5 * directional_deriv / (
                            directional_deriv + 0.5 * actual_reduction)
                    if 0.1 * self.fnorm() >= laststatus.fnorm or temp < 0.1:
                        temp = 0.1
                    delta = temp * min(delta, 10 * pnorm)
                    par /= temp
                elif par == 0 or ratio > 0.75:
                    delta = 2 * pnorm
                    par = 0.5 * par

                # if iteration is successful, update parameters
                if ratio >= 0.0001:
                    xnorm = norm(self.scale * self.status.parameters)
                    self.plot_intermediate(True)
                    laststatus = self.status
                else:
                    self.plot_intermediate(False)
                    self.status = laststatus

                # test for convergence
                c1 = (abs(actual_reduction) <= ftol and
                      predicted_reduction <= ftol and ratio <= 2)
                c2 = delta <= (xtol * xnorm)
                if c1 and c2:
                    self.message = "both actual and predicted relative " \
                        "reductions in the sum of squares are at most " \
                        "`ftol`, and the relative error between two " \
                        "consecutive iterates is at most `xtol`"
                elif c1:
                    self.message = "both actual and predicted relative " \
                        "reductions in the sum of squares are at most `ftol`."
                elif c2:
                    self.message = "relative error between two consecutive " \
                        "iterates is at most `xtol`"
                self.exitcode = 2 * c2 + c1
                if c1 or c2 or not self.running:
                    return self.status.parameters

                # tests for termination and stringent tolerances.
                if gnorm < epsmch:
                    self.exitcode = 8
                    self.message = f"gtol={gtol} is too small, func(x) is " \
                        "orthogonal to the columns of the Jacobian to " \
                        "machine precision."
                    raise FitError(8, self.message)
                if delta < epsmch * xnorm:
                    self.exitcode = 7
                    self.message = f"xtol={xtol} is too small, no further " \
                        "improvement in the approximate solution is possible."
                    raise FitError(7, self.message)
                if (abs(actual_reduction) < epsmch and
                        predicted_reduction < epsmch and 0.5 * ratio < 1):
                    self.exitcode = 6
                    self.message = f"ftol={ftol} is too small, no further " \
                        "reduction in the sum of squares is possible."
                    raise FitError(6, self.message)
                if maxfev is not None and self.nfev > maxfev:
                    self.exitcode = 5
                    self.message = "Number of function evaluations has " \
                                   f"reached {self.nfev}."
                    raise FitError(5, self.message)
        self.exitcode = 9
        self.message = f"Number of iterations has reached {iterations}."
        raise FitError(9, self.message)

    def covar(self, sigma=None, eps=0):
        """ calculate the covariance matrix of the solution

        This method gives an estimation of the error of the fitting
        parameters.

        Parameters
        ----------
        x : vector, length N
            the value to calculate the errors at
        eps : float, optional
            the threshold below which a parameter is considered
            linear-dependent (ie, has no influence on a fitting result).
        f : vector, length M
            the function value at `p`, calculated internally if None """
        if sigma is None:
            sigma = self.fnorm() / sqrt(len(self.values())
                                        - len(self.parameters()))
        r = self.r() / sigma
        jpvt = self.ipvt()
        condition = abs(r.diagonal()) > abs(r[0, 0]) * eps
        nsing = condition.nonzero()[0][-1] + 1
        r = r[:nsing, :nsing]
        ri = zeros((len(jpvt), nsing), dtype=r.dtype)
        ri[jpvt[:nsing], arange(nsing)] = 1
        ri = solve_triangular(r.T, ri.T, lower=True, overwrite_b=True)
        return ri.T @ ri

    def error(self, sigma=None, eps=0):
        """ calculate the error of the fit parameters

        Give an estimation on the error of the fit result.

        Returns
        -------
        err : vector, length `N`
            the estimated error for each parameter
        """
        return sqrt(diagonal(self.covar(sigma, eps)))

def minimize(fun, x0, args=(), method="maquardt-levenberg", jac=None,
             options=dict(), full_output=False, callback=None, retall=False):
    class MyFit(Fit):
        def func(self, x):
            return fun(x, *args)

        if jac is not None:
            def jacobian(self, x, fvec, ret):
                return jac(x)

        def plot_intermediate(self, x, f, good):
            if callback is not None:
                callback(x)
            if retall:
                self.allvecs.append(x.copy())
    fit = MyFit()
    try:
        fit.allvecs = [ ]
        ret = fit.fit(x0, **options)
        success = True
    except FitError as e:
        mesg = e.message
        success = False
        ret = None
    else:
        mesg = fit.message
    info = {
            "solution": ret,
            "success": success,
            "status": fit.exitcode,
            "message": mesg,
            "fun": fit.fvec,
            "nfev": fit.nfev,
            "nit": fit.iterations
        }
    if retall:
        info["allvecs"] = fit.allvecs
    return ret, info

class Function(Fit, dict):
    """Convenience class to fit to Python functions

    This class is a specialization of the :class:`Fit` class,
    if one wants to fit to a simple Python function. A possibly
    even simpler way to do so is the decorator `fitfunction`.

    An example::

        # define the function we want to fit to:
        def gaussian(x, width, height, position):
            return height * np.exp(-((x - position) / width) ** 2 / 2)

        # make an object and define the functions parameters and their
        # default values (they must coincide in name!)
        fit = Function(gaussian, width=1, height=1, position=1)

        x = np.arange(10)
        y = np.array([0.1, 1.5, 2.7, 3.2, 4.8, 5.3, 4.4, 3.6, 2.9, 1.0])

        # fit to the data. Give the starting values for width and
        # height. As position is not mentioned, don't fit it, keep it
        # constant.
        fit.fit(x, y, width=1, height=1)

        plt.plot(x, y, ".")
        plt.plot(x, fit(x)) # one can simply use fit as a function!

    Parameters
    ----------
    function : function
        The function to fit to. See the attribute `function`.
    additional parameters : must coincide with parameters to function
        Gives the default values for the function's parameters.

    Attributes
    ----------
    function : function
        this is the function to call from this object. The first
        parameter to this function is the variable to which we fit,
        the others are the actual parameters.
    parameters : list of strings
        lists the names of the parameters of the fitting functions,
        as they are declared. They are used both as attributes in the
        object, as well as parameters to the function.
    fitparameters : list of strings
        while fitting, this is the list of parameters which are actually
        fitted. All other parameters stay constant.  """

    def __init__(self, function=None, **parameters):
        self.function = function
        dict.__init__(self, parameters)

    def __call__(self, x):
        """Call the user-supplied function

        Here we call the user-supplied function with its variable `x`
        and all parameters as currently stored in the object. """

        try:
            return self.function(x, **self)
        except InvalidParameter as e:
            if not isinstance(e.number, (list, tuple)):
                e.number = e.number,
            e.number = [self.fitparameters.index(n) for n in e.number]
            raise

    def func(self, x):
        """The fitting function

        This function sets the currently tried values of the 
        fit to the local parameters, and then calls the user supplied
        function. """

        for p, xx in zip(self.fitparameters, x):
            self[p] = xx
        return self(self.datax) - self.datay

    def fit(self, x, y, **parameters):
        """Fit to data `x` and `y`.

        This function performs the actual fit, by minimizing
        :math:`|f(x) - y|`, varying the parameters to the user-supplied
        function `f`. """

        self.datax = x
        self.datay = y
        self.fitparameters, start = zip(*parameters.items())
        ret = Fit.fit(self, start)
        return dict(zip(self.fitparameters, ret))

    def plot(self, **kwargs):
        """Plot the last data together with its fit

        Keyword arguments are handed over to the underlying
        :func:`~matplotlib.pyplot.plot` function. Arguments prefixed with
        ``fit_`` are only handed over to the plotted fit, vice versa for a
        ``data_`` prefix.

        as an example, a Gaussian fit through data `x` and `y` may be done
        like ::

            gaussian.fit(x, y, width=3, height=4, position=3)
            gaussian.plot(data_color='g', fit_color='r')

        which would result in a plot similar to the following:

        .. plot::

            from numpy import arange, exp, random
            from marlev.marlev import fitfunction

            @fitfunction(width=1, height=1, position=0)
            def gaussian(x, width, height, position):
                return height * exp(-((x - position) / width) ** 2 / 2)

            x = arange(30) - 10
            y = 5 * exp(-0.1 * (x - 3) ** 2) + random.standard_normal(30)
            gaussian.fit(x, y, width=3, height=4, position=3)
            gaussian.plot(data_color='g', fit_color='r')
        """
        from matplotlib.pyplot import plot

        n = max(101, 2 * len(self.datax))
        fx = linspace(min(self.datax), max(self.datax), 101)

        fargs = {k[4:] if k.startswith("fit_") else k: v
                 for k, v in kwargs.items() if not k.startswith("data_")}
        plot(fx, self(fx), '-', **fargs)
        fargs = {k[5:] if k.startswith("data_") else k: v
                 for k, v in kwargs.items() if not k.startswith("fit_")}
        plot(self.datax, self.datay, '.', **fargs)


def fitfunction(**defaults):
    """Decorator to turn a function into a fitting function

    This decorator turns a normal function into a fittable function,
    which is an object of the class :class:`Function`. As parameters to this
    decorator you give the default values for the fitted parameters.
    An example::

        @fitfunction(width=1, height=1, position=0)
        def gaussian(x, width, height, position):
            return height * exp(-((x - position) / width) ** 2 / 2)

    gives you a function in one variable (`x`) that represents a Gaussian
    with the parameters already set. So, by typing ::

        >>> x = np.linspace(-20, 20, 101)
        >>> plt.plot(x, gaussian(x))

    you get a plot of this Gaussian. You may change the default values::

        >>> gaussian["width"] = 2 

    The whole point of this function then is to be able to fit to
    data, e.g. by ::

        >>> x = np.arange(10)
        >>> y = np.array([0.1, 1.5, 2.7, 3.2, 4.8, 5.3, 4.4, 3.6, 2.9, 1.0])
        >>> gaussian.fit(x, y, width=1, height=2)

    where the additional named parameters are the default values for
    the parameters to fit. All other parameters are kept constant
    at the previously defined value. """

    def wrapper(f):
        ret = Function(f, **defaults)
        return functools.wraps(f)(ret)
    return wrapper

def leastsq(func, x0, args=(), Dfun=None, full_output=0, col_deriv=0, **kwargs):
    """ Function equivalent to the class :class:`Fit`

    This function is supposed to be a drop-in replacement for 
    :func:`scipy.optimize.leastsq`. See the documentation there.
    """
    class MyFit(Fit):
        def func(self, x):
            return func(x, *args)

        if Dfun is not None:
            def jacobian(self, x, fvec, ret):
                if col_deriv:
                    return Dfun(x)
                else:
                    return Dfun(x).T
    fit = MyFit()
    try:
        ret = fit.fit(x0, **kwargs)
    except FitError as e:
        ret = None
        mesg = e.message
    else:
        mesg = fit.message
    if full_output:
        return ret, fit.covar(ret), {"nfev": fit.nfev,
                "fvec": fit.fvec}, mesg, fit.exitcode
    else:
        return ret, fit.exitcode
