from numpy import exp, linspace
from numpy.random import normal

from pytest import approx

from .marlev import leastsq


def test_gaussian():
    x = linspace(0, 10, 101)
    y = 20 * exp(-0.5 * ((x - 7) / 5) ** 2) + normal(scale=0.1, size=101)

    def gaussian(height, center, width):
        return height * exp(-0.5 * ((x - center) / width) ** 2)

    args = (21, 6.5, 4.8)
    args, _ = leastsq(lambda a: gaussian(*a) - y, args)
    assert args == approx((20, 7, 5), abs=0.1)
