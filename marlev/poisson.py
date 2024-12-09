from numpy import empty_like, log, sqrt, dot
from scipy.linalg import qr_multiply, solve_triangular

from .marlev import calculator, Fit


class PoissonFit(Fit):
    """ Fitting to Poissonian distributed data

    This follows `Laurence, Ted A. and Chromy, Brett A., "Efficient maximum
    likelihood estimator fitting of histograms", Nature Methods 7, 338 (2010)
    <https://doi.org/10.1038/nmeth0510-338>`_

    One major difference to the class Fit is that experimental data to
    fit to has to be supplied directly, as the errors directly depend
    on them, and not just on the difference between the calculated and
    measured value.

    As this class is for Poissonian distributed data, the data has to be
    positive or zero, while the fitting function must be strictly zero. """

    def do_decompose(self):
        j = self.jacobian()
        j1 = j[self.positive, :]
        f = self.values()[self.positive]
        da = (j1.T * (self.sqrt_data / f)).T
        bDd = f / self.sqrt_data - self.sqrt_data
        qtf, r, ipvt = qr_multiply(da, bDd, pivoting=True)
        qtf += solve_triangular(r.T, (j.T @ ~self.positive)[ipvt], lower=True,
                                overwrite_b=True)
        return qtf, r, ipvt

    @calculator
    def fnorm(self):
        r = self.values().sum() - self.sum_data
        f = self.values()[self.positive]
        r -= (self.positive_data * log(f / self.positive_data)).sum()
        if r > 0:
            return sqrt(2 * r)
        else:
            return 0 # should not happen except numerical error

    def fit(self, x, data=None, **kwargs):
        if data is not None:
            self.data = data
        self.positive = self.data > 0
        self.positive_data = self.data[self.positive]
        self.sqrt_data = sqrt(self.positive_data)
        self.sum_data = self.data.sum()
        return Fit.fit(self, x, **kwargs)
