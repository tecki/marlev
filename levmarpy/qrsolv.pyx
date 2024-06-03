"""
This is an internal module only, containing the inner loops
for levmarpy.
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
            if diag[j] != 0:
                memset(ta, 0, (N + 1) * sizeof(cython.floating))
                ta[j] = diag[j]
                for j <= k < N:
                    if ta[k] != 0:
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
