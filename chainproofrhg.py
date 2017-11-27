"Chain-Proof Random Hit Generator module. Contains the ChainProofRHG class and methods."
from math import fabs, ceil, log10
from random import random
from functools import partialmethod
import operator

BASE_ERROR = 1e-8


def getbasechance(prob, error=BASE_ERROR):
        """Uses the bisection method to find the base chance and return it."""
        lower = 0.
        upper = prob
        while True:
            # Get the midpoint.
            midpoint = 0.5 * (lower + upper)
            midvalue = getprobability(midpoint, error)
            if fabs(midvalue - prob) < error:
                # Return if the error is sufficiently small.
                break
            # Replace the point such that the two points still bracket the true value.
            elif midvalue < prob:
                lower = midpoint
            elif midvalue > prob:
                upper = midpoint
        return midpoint


def getprobability(chance, error=BASE_ERROR):
        """Calculate the effective probability from a given base hit chance."""
        # Get the maximum number of times the chance can fail until it is guaranteed to pass.
        max_fails = int(ceil(1 / float(chance)))
        # Initialize calculation values.
        chance_at_i = 0.
        chance_upto_i = 0.
        total = 0.
        # Sum the individual pass chances to get the cumulative number of chances.
        for i in range(1, max_fails + 1):
            chance_at_i = min(1., i * chance) * (1 - chance_upto_i)
            chance_upto_i += chance_at_i
            total += i * chance_at_i
        # Take the reciprocal to convert from 1 in N times happening to a probability.
        return 1. / total


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
        'do_reset'
        )

    def __init__(self, p, error=BASE_ERROR, do_reset=True):
        self._error = error  # Minimum accuracy of iteration.
        self._failcount = 0  # Number of missed hits so far.
        self._lastcount = 0  # The number of times needed to hit the last time.
        self._lock = False  # Used to lock __next__ into returning StopIteration.
        self.prob = p  # Initialize the average probability value.
        self.do_reset = do_reset # Reset chain counter when modifying probability.

    def __setattr__(self, name, value):
        if name == '_prob':
            super().__setattr__('_prob', round(value, -int(log10(self._error))))
        elif name == 'prob':
            if isinstance(value, ChainProofRHG):
                # Copy the probability value if attempting to set using another ChainProofRHG object.
                self._prob = value._prob
            else:
                # Behavior is not defined outside of the range 0 < p < 1.
                if not 0 < value < 1:
                    raise ValueError(
                        "Probability values may not exceed the range between 0 and 1"
                    )
                self._prob = value
            self._proc = getbasechance(self._prob, self._error)
            if self.do_reset:
                self._lastcount = 0
        elif name == 'proc':
            raise AttributeError(
                "The base hit chance is a read-only attribute:\n"
                "Change the <prob> attribute or perform an "
                "in-place operation to the object instead"
            )
        elif name == 'error':
            self._error = value
            self._proc = getbasechance(self._prob, self._error)
            if self.do_reset:
                self._lastcount = 0
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name == 'prob':
            # The average probability for a test to hit.
            return self._prob
        elif name == 'proc':
            # The base probability of each test.
            return self._proc
        elif name == 'procn':
            # The probability of the next test to hit.
            return (self._failcount + 1) * self._proc
        elif name == 'error':
            # The error used when determining self.proc
            return self._error
        elif name == 'lastcount':
            # The number of times it took to hit the last time.
            return self._lastcount

    def getprobability(self):
        """Calculate the effective probability from the current hit chance for comparison purposes."""
        return getprobability(self._proc, self._error)

    def reset(self):
        """Reset iteration values."""
        self._failcount = 0
        self._lock = False

    def test_nhits(self, n):
        """Evaluate n hits."""
        return (bool(self) for _ in range(n))

    def __repr__(self):
        """Nicely-formatted expression that can be used in eval()."""
        return '{}({}, {!r}, {})'.format(
            self.__class__.__name__,
            self._prob, self._error, self.do_reset
            )

    # Note: The rich comparison methods take the minimum allowable error into account
    # when comparing against another ChainProofRHG object.
    def _cmp(self, other, op):
        """Allow for direct comparison of the probability value as a number."""
        if isinstance(other, ChainProofRHG):
            return op(
                (self._prob, log10(self._error)),
                (other._prob, log10(other._error))
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
        return hash((self._prob, self._error, self.do_reset))

    def __bool__(self):
        """Evaluate the next hit, returning True if it does, and False otherwise."""
        hit = random() < self.procn
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

    def __length_hint__(self):
        """Returns the maximum number of times it can fail in a row."""
        return int(ceil(1 / float(self._proc)))

    def __iter__(self):
        """Allows the probability to be used as an iterator, iterating until a hit."""
        self._lock = False
        return self

    def __next__(self):
        """Attempt to roll for a hit until one happens, then raise StopIteration."""
        if self._lock:
            raise StopIteration
        else:
            if self:
                self._lock = True
                raise StopIteration
            else:
                return self._failcount

    def _operate(self, other, op):
        """Generic mathematical operator function."""
        # Allows usage of common arithmetic operators on ChainProofRHG objects directly for convenience.
        if isinstance(other, ChainProofRHG):
            return self.__class__(
                op(self._prob, other._prob),
                max(self._error, other._error)
                )
        else:
            return op(self._prob, other)

    def _roperate(self, other, op):
        """Allows for operations the other way to work."""
        return op(other, self._prob)

    def _ioperate(self, other, op):
        """Allows for in-place operations."""
        if isinstance(other, ChainProofRHG):
            self.prob = op(self._prob, other._prob)
        else:
            self.prob = op(self._prob, other)
        return self

    # Addition
    __add__ = partialmethod(_operate, op=operator.add)
    __radd__ = partialmethod(_roperate, op=operator.add)
    __iadd__ = partialmethod(_ioperate, op=operator.add)
    # Subtraction
    __sub__ = partialmethod(_operate, op=operator.sub)
    __rsub__ = partialmethod(_roperate, op=operator.sub)
    __isub__ = partialmethod(_ioperate, op=operator.sub)
    # Multiplication
    __mul__ = partialmethod(_operate, op=operator.mul)
    __rmul__ = partialmethod(_roperate, op=operator.mul)
    __imul__ = partialmethod(_ioperate, op=operator.mul)
    # Floor Division
    __floordiv__ = partialmethod(_operate, op=operator.floordiv)
    __rfloordiv__ = partialmethod(_roperate, op=operator.floordiv)
    __ifloordiv__ = partialmethod(_ioperate, op=operator.floordiv) # Must always fail
    # True Division
    __truediv__ = partialmethod(_operate, op=operator.truediv)
    __rtruediv__ = partialmethod(_roperate, op=operator.truediv)
    __itruediv__ = partialmethod(_ioperate, op=operator.truediv)
    # Exponentiation
    __pow__ = partialmethod(_operate, op=operator.pow)
    __rpow__ = partialmethod(_roperate, op=operator.pow)
    __ipow__ = partialmethod(_ioperate, op=operator.pow)

    def _setoperate(self, other, op):
        """Returns a ChainProofRHG object using the operator as a probability set logic operator."""
        if isinstance(other, ChainProofRHG):
            return self.__class__(
                op(self._prob, other._prob),
                max(self._error, other._error)
                )
        elif 0 < other < 1:
            return self.__class__(op(self._prob, other))
        else:
            raise TypeError("Incompatible operand type between probability and non-probability")

    def _rsetoperate(self, other, op):
        """Commutativity support."""
        return _setoperate(other, self, op)

    def _isetoperate(self, other, op):
        """In-place opertion support."""
        self.prob = _setoperate(self, other, op)
        return self

    # P(A) & P(B) = P(A) * P(B)
    __and__ = partialmethod(_setoperate, op=operator.mul)
    __rand__ = partialmethod(_rsetoperate, op=operator.mul)
    __iand__ = partialmethod(_isetoperate, op=operator.mul)
    # P(A) ^ P(B) = P(A) + P(B) - 2 * P(A) * P(B)
    __xor__ = partialmethod(_setoperate, op=lambda l, r: l + r - 2*l*r)
    __rxor__ = partialmethod(_rsetoperate, op=lambda l, r: l + r - 2*l*r)
    __ixor__ = partialmethod(_isetoperate, op=lambda l, r: l + r - 2*l*r)
    # P(A) | P(B) = P(A) + P(B) - P(A) * P(B)
    __or__ = partialmethod(_setoperate, op=lambda l, r: l + r - l*r)
    __ror__ = partialmethod(_rsetoperate, op=lambda l, r: l + r - l*r)
    __ior__ = partialmethod(_isetoperate, op=lambda l, r: l + r - l*r)

    # ~P(A) = 1 - P(A)
    def __invert__(self):
        """Return a ChainProofRHG object with the probability that this will not hit."""
        return self.__class__(1 - self._prob)

    def __round__(self, n=0):
        """Allows the use of round() to truncate probability value."""
        return round(self._prob, n)

    def round_ip(self, n=0):
        """Performs rounding in-place."""
        self.prob = round(self._prob, n)


if __name__ == '__main__':
    # Some simple tests.
    cprhg = ChainProofRHG(0.25)
    assert cprhg != ChainProofRHG(0.25, 1e-2)
    print(cprhg)
    assert cprhg.getprobability() - 0.25 < cprhg.error
    print(cprhg.getprobability() - 0.25)
    cprhg.prob = 0.17
    print(cprhg)
    assert cprhg.getprobability() - 0.17 < cprhg.error
    assert cprhg.procn == cprhg.proc
    print(cprhg.procn)
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
    a += 0.1
    assert a == ChainProofRHG(0.2)
    assert round(a - b, 2) == 0.05
    assert round(a - 0.05, 2) == 0.15
    assert round(0.05 - a, 2) == -0.15
    assert a * 5 == 1.
    assert 5 * a == 1.
    assert a * b == 0.03
    b *= a
    assert b == ChainProofRHG(0.03)
    b /= a
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
    cprhg.prob = 0.15
    print(cprhg)
    hitlist = [len([i for i in cprhg]) + 1 for i in range(operator.length_hint(cprhg))]
    print(hitlist)
    print(operator.length_hint(cprhg) / sum(hitlist))
    for prob in range(5, 40, 5):
        print('{}%: {}'.format(prob, ChainProofRHG(prob/100).proc))
