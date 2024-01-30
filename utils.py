import sys
import logging
from functools import reduce


def GetLogger(logpath):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(logpath, mode='w')
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    log.addHandler(sh)

    return log


def sec_to_str(t):
    '''Convert seconds to days:hours:minutes:seconds'''
    [d, h, m, s, n] = reduce(lambda ll, b : divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24])
    f = ''
    if d > 0:
        f += '{D}d:'.format(D=d)
    if h > 0:
        f += '{H}h:'.format(H=h)
    if m > 0:
        f += '{M}m:'.format(M=m)

    f += '{S}s'.format(S=s)
    return f