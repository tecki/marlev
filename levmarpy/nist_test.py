from numpy import array, errstate
import pytest
from pathlib import Path

from .levmar import Fit, InvalidParameter

p = Path(__file__).parent / 'nist'

class NistFit(Fit):
    def func(self, x):
        p = dict(zip(self.parameters.keys(), x))
        p['x'] = self.datax
        try:
            with errstate(all='raise'):
                return eval(self.cfunc, globs, p) - self.datay
        except FloatingPointError:
            raise InvalidParameter

globs = {'e': 0}
exec('from numpy import exp, arctan, cos, sin, pi', globs)


@pytest.mark.parametrize('path', p.glob('*.dat'), ids=lambda p: p.name)
def test_fit(path):
    fit = NistFit()

    with path.open() as fin:
        lines = list(fin)
        infunc = False
        data = False
        func = ''
        ld = []
        for ll in lines:
            ls = ll.strip()

            if data:
                *d, = ls.split()
                ld.append([float(x) for x in d])
            if ls.startswith("Certified"):
                _, start, _, stop = ls.rsplit(maxsplit=3)
                start = int(start) - 1
                stop = int(stop[:-1])
            elif ls.startswith("Data: ") and ls.endswith(" x"):
                data = True
            elif ls.startswith("y =") or ls.startswith("y  ="):
                func = ls[4:].strip()
                infunc = True
            elif infunc:
                func += ls
            if infunc and func.endswith(' e'):
                func = func.replace('[', '(').replace(']', ')')
                fit.cfunc = compile(func, str(p), mode='eval')
                infunc = False

        params = {}

        for no in [0, 1]:
            for ll in lines[start:stop-5]:
                name, _, *args= ll.split()
                params[name] = tuple(float(a) for a in args)
            ad = array(ld)
            fit.parameters = params
            fit.datax = ad[:, 1]
            fit.datay = ad[:, 0]
            fit.fit(array([p[no] for p in params.values()]), iterations=1000)

            for res, (_, _, ref, dev) in zip(fit.params, params.values()):
                assert res == pytest.approx(ref, abs=dev)
