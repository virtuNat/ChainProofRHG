"Chain-Proof Random Hit Generator module. Contains the ChainProofRHG class and methods."
from math import ceil, log10
from random import random
from functools import partialmethod
import operator

BASE_ERROR = 1e-6


def getavgprob(bchance):
    """Calculate the effective probability from a given base hit chance."""
    if not (0 <= bchance <= 1):
        raise ValueError('Probability values lie between 0 and 1 inclusive.')
    elif bchance >= 0.5:
        return 1 / (2 - bchance)
    hitchance = bchance
    chancesum = bchance
    hitscount = bchance
    # Sum the individual pass chances to get the cumulative number of chances.
    for i in range(2, int(ceil(1 / bchance)) + 1):
        hitchance = min(1, bchance * i) * (1 - chancesum)
        chancesum += hitchance
        hitscount += hitchance * i
    # Take the reciprocal to convert from 1 in N times happening to a probability.
    return 1 / hitscount


def getbchance(avgprob, error=BASE_ERROR):
    """Uses the bisection method to find the base chance and return it."""
    if not (0 <= avgprob <= 1):
        raise ValueError('Probability values lie between 0 and 1 inclusive.')
    elif avgprob >= 2/3:
        return 2 - (1 / avgprob)
    lower = 0
    upper = avgprob
    while True:
        # Get the midpoint.
        midpoint = (lower + upper) / 2
        midvalue = getavgprob(midpoint)
        if abs(midvalue - avgprob) < error:
            # Return if the error is sufficiently small.
            break
        # Replace the point such that the two points still bracket the true value.
        elif midvalue < avgprob:
            lower = midpoint
        else:
            upper = midpoint
    return midpoint


class ChainProofRHG(object):
    """Chain-Proof Random Hit Generator for more consistent RNG.

    ChainProofRHG() objects exist as a tool to emulate the style of random hit generation used in
    Warcraft 3, and subsequently, DOTA 2.

    Documentation incomplete.
    """
    __slots__ = (
        '_error',
        '_failcount',
        '_lastcount',
        '_lock',
        '_prob',
        '_proc',
        '_procnow',
        )

    def __init__(self, avgprob, error=BASE_ERROR):
        if error > 1e-4:
            raise ValueError('Expected error value too large')
        self._error = error # Minimum accuracy of iteration.
        self._failcount = 0 # Number of missed hits so far.
        self._lastcount = 0 # The number of times needed to hit the last time.
        self._lock = False  # Used to lock __next__ into returning StopIteration.
        self._prob = round(avgprob, self.errordigits) # Initialize the average probability value.
        self._proc = getbchance(avgprob, error) # Initialize the base probability value.
        self._procnow = self._proc

    def __getattr__(self, name):
        if name == 'prob':
            # The average probability for a test to hit.
            return self._prob
        elif name == 'proc':
            # The base probability of each test.
            return self._proc
        elif name == 'procnow':
            # The probability of the next test to hit.
            return self._procnow
        elif name == 'error':
            # The error used when determining self.proc
            return self._error
        elif name == 'errordigits':
            # Number of accurate digits past the decimal point + 1.
            return -int(ceil(log10(self._error)))
        elif name == 'lastcount':
            # The number of times it took to hit the last time.
            return self._lastcount
        elif name == 'maxfails':
            # The maximum number of times it can fail in a row.
            return int(ceil(1 / float(self._proc)))

    def getavgprob(self):
        """Calculate the effective probability from the current hit chance for comparison purposes."""
        return getavgprob(self._proc)

    def reset(self):
        """Reset iteration values."""
        self._failcount = 0
        self._lock = False

    def test_nhits(self, n):
        """Evaluate n hits."""
        return (bool(self) for _ in range(n))

    def __repr__(self):
        """Nicely-formatted expression that can be used in eval()."""
        return '{}({}, {!r})'.format(
            self.__class__.__name__,
            self._prob, self._error,
            )

    # Note: The rich comparison methods take the minimum allowable error into account
    # when comparing against another ChainProofRHG object.
    def _cmp(self, other, op):
        """Allow for direct comparison of the probability value as a number."""
        if isinstance(other, ChainProofRHG):
            return op(
                (self._prob, self.errordigits),
                (other.prob, other.errordigits)
                )
        else:
            return op(self._prob, other)

    __eq__ = partialmethod(_cmp, op=operator.eq)
    __ne__ = partialmethod(_cmp, op=operator.ne)
    __lt__ = partialmethod(_cmp, op=operator.lt)
    __le__ = partialmethod(_cmp, op=operator.le)
    __gt__ = partialmethod(_cmp, op=operator.gt)
    __ge__ = partialmethod(_cmp, op=operator.ge)

    def __hash__(self):
        """Make the object hashable."""
        return hash((self._prob, self._error))

    def __bool__(self):
        """Evaluate the next hit, returning True if it does, and False otherwise."""
        hit = random() < self._procnow
        if hit:
            self._lastcount = self._failcount + 1
            self._failcount = 0
        else:
            # If the hit fails, increase the probability for the next hit.
            self._failcount += 1
        return hit

    def __int__(self):
        """Evaluate the next hit as an integer."""
        return int(bool(self))

    def __float__(self):
        """Returns the probability value."""
        return self._prob

    def __iter__(self):
        """Allows the probability to be used as an iterator, iterating until a hit."""
        self._lock = False
        return self

    def __next__(self):
        """Attempt to roll for a hit until one happens, then raise StopIteration."""
        if self._lock:
            raise StopIteration
        hit = random() < self._procnow
        if hit:
            self._lastcount = self._failcount + 1
            self._failcount = 0
            self._procnow = self._proc
            self._lock = True
            raise StopIteration
        self._procnow = min(1, self._procnow + self._proc)
        self._failcount += 1
        return self._failcount

    def _operate(self, other, op):
        """Generic mathematical operator function."""
        # Allows usage of common arithmetic operators on ChainProofRHG objects directly for convenience.
        if isinstance(other, ChainProofRHG):
            return self.__class__(op(self._prob, other.prob), max(self._error, other.error))
        return self.__class__(op(self._prob, other), self._error)

    def _roperate(self, other, op):
        """Allows for operations the other way to work."""
        return self.__class__(op(other, self._prob), self._error)

    # Addition
    __add__ = partialmethod(_operate, op=operator.add)
    __radd__ = partialmethod(_roperate, op=operator.add)
    # Subtraction
    __sub__ = partialmethod(_operate, op=operator.sub)
    __rsub__ = partialmethod(_roperate, op=operator.sub)
    # Multiplication
    __mul__ = partialmethod(_operate, op=operator.mul)
    __rmul__ = partialmethod(_roperate, op=operator.mul)
    # True Division
    __truediv__ = partialmethod(_operate, op=operator.truediv)
    __rtruediv__ = partialmethod(_roperate, op=operator.truediv)
    # Exponentiation
    __pow__ = partialmethod(_operate, op=operator.pow)
    __rpow__ = partialmethod(_roperate, op=operator.pow)

    def _setoperate(self, other, op):
        """Returns a ChainProofRHG object using the operator as a probability set logic operator."""
        if isinstance(other, ChainProofRHG):
            return self.__class__(op(self._prob, other.prob), max(self._error, other.error))
        elif 0 <= other <= 1:
            return self.__class__(op(self._prob, other), self._error)
        else:
            raise TypeError("Incompatible operand type between probability and non-probability")

    # P(A) & P(B) = P(A) * P(B)
    __and__ = partialmethod(_setoperate, op=operator.mul)
    __rand__ = partialmethod(_setoperate, op=operator.mul)
    # P(A) ^ P(B) = P(A) + P(B) - 2 * P(A) * P(B)
    __xor__ = partialmethod(_setoperate, op=lambda l, r: l + r - 2*l*r)
    __rxor__ = partialmethod(_setoperate, op=lambda l, r: l + r - 2*l*r)
    # P(A) | P(B) = P(A) + P(B) - P(A) * P(B)
    __or__ = partialmethod(_setoperate, op=lambda l, r: l + r - l*r)
    __ror__ = partialmethod(_setoperate, op=lambda l, r: l + r - l*r)

    # ~P(A) = 1 - P(A)
    def __invert__(self):
        """Return a ChainProofRHG object with the probability that this will not hit."""
        return self.__class__(1 - self._prob)

    def __round__(self, n=0):
        """Allows the use of round() to truncate probability value."""
        return round(self._prob, min(n, self.errordigits))


if __name__ == '__main__':
    # Some simple tests.
    cprhg = ChainProofRHG(0.25)
    assert cprhg != ChainProofRHG(0.25, 1e-5)
    print(cprhg)
    assert cprhg.prob == 0.25
    assert abs(cprhg.getavgprob() - 0.25) < cprhg.error
    print(cprhg.getavgprob() - 0.25)
    cprhg = ChainProofRHG(0.17)
    print(cprhg)
    assert cprhg.prob == 0.17
    assert cprhg.procnow == cprhg.proc
    print(cprhg.procnow)
    for i in cprhg:
        print(i, end=' ')
    print("|", cprhg.lastcount)
    a = ChainProofRHG(0.1)
    assert a == 0.1 == ChainProofRHG(0.1)
    assert 0 < a < 1
    assert 0.1 <= a <= 0.1
    b = ChainProofRHG(0.15)
    print(a + b)
    print((a + b).proc)
    assert a + b == 0.25
    assert a + 0.1 == 0.2
    assert 0.1 + a == 0.2
    print(0.1 + a)
    a = a + 0.1
    assert a == ChainProofRHG(0.2)
    assert round(a - b, 2) == 0.05
    assert round(a - 0.05, 2) == 0.15
    assert round(0.05 - float(a), 2) == -0.15
    assert a * 5 == 1.
    assert 5 * a == 1.
    assert a * b == 0.03
    b = a * b
    assert b == ChainProofRHG(0.03)
    b = b / a
    assert b == 0.15
    print(a | b)
    print((a | b).proc)
    assert a | b == a + b - (a * b)
    print(a & b)
    print((a & b).proc)
    assert a & b == a * b
    print(a ^ b)
    print((a ^ b).proc)
    assert a ^ b == a + b - (2 * a * b)
    print(~a)
    print((~a).proc)
    assert ~~a == a
    cprhg = ChainProofRHG(0.15)
    print(cprhg)
    hitlist = [len([i for i in cprhg]) + 1 for _ in range(25)]
    print(hitlist)
    print(len(hitlist) / sum(hitlist))
    for prob in range(5, 51, 5):
        print('{:02}%: {}'.format(prob, getbchance(prob/100)))
