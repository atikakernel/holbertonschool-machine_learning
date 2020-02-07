#!/usr/bin/env python3
"""Poisson distribution calculations"""


class Poisson:
    """Poisson distribution stats class"""
    def __init__(self, data=None, lambtha=1.):
        """Initialize poisson distribution stats"""
        if data:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            elif not isinstance(data, list):
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = float(sum(data) / len(data))
        else:
            self.lambtha = float(lambtha)


    def pmf(self, k):
        """PMF at k number of events"""
        e = 2.7182818285
        k = int(k)
        if k < 0:
            return 0

        fact = 1
        for n in range(1, k + 1):
            fact = fact * n

        return ((pow(self.lambtha, k) / fact) *
                pow(e, -self.lambtha))

    def cdf(self, k):
        """CDF at k number of events"""
        k = int(k)
        if k < 0:
            return 0
        count = 0.0
        while k >= 0:
            count += self.pmf(k)
            k -= 1
        return count
