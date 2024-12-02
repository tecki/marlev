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
This is an internal module only, containing the inner loops
for marlev.
"""
cdef extern from "alloca.h":
    void *alloca(int) nogil

cdef extern from "string.h":
    void *memset(void *, int, int) nogil

cdef extern from "math.h":
    double sqrt(double) nogil
    double fabs(double) nogil

cimport cython

@cython.boundscheck(False)
@cython.cdivision(True)
def qrsolv(cython.floating[:, :] s, cython.floating[:] diag):
    cdef unsigned int N
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l
    cdef cython.floating sin
    cdef cython.floating cos
    cdef cython.floating tan
    cdef cython.floating cotan
    cdef cython.floating tmp
    cdef cython.floating *ta

    with cython.nogil:
        N = diag.shape[0]
        ta = <cython.floating *> alloca((N + 1) * sizeof(cython.floating))

        for 0 <= j < N:
            if diag[j] == 0:
                continue
            memset(ta, 0, (N + 1) * sizeof(cython.floating))
            ta[j] = diag[j]
            for j <= k < N:
                if ta[k] == 0:
                    continue
                if fabs(s[k, k]) > fabs(ta[k]):
                    tan = ta[k] / s[k, k]
                    cos = 1 / sqrt(1 + tan * tan)
                    sin = cos * tan
                else:
                    cotan = s[k, k] / ta[k]
                    sin = 1 / sqrt(1 + cotan * cotan)
                    cos = sin * cotan
                for k <= l <= N:
                    tmp = s[k, l]
                    s[k, l] = cos * tmp + sin * ta[l]
                    ta[l] = -sin * tmp + cos * ta[l]
