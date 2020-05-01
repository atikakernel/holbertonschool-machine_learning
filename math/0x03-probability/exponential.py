#!/usr/bin/env python3
"""Poisson distribution calculations"""


class Exponential:
    """Poisson distribution stats class"""
    def __init__(self, data=None, lambtha=1.):
        """Initialize poisson distribution stats"""
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)

        else:
            if type(data) != list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """PMF at k number of events"""
        e = 2.7182818285
        x = int(x)
        if x < 0:
            return 0
        return self.lambtha * e**(-self.lambtha * x)

    def cdf(self, x):
        """CDF at k number of events"""
        x = int(x)
        e = 2.7182818285
        if x < 0:
            return 0
        return (1 - (e**(-self.lambtha*x)))
