#!/usr/bin/env python3
"""Poisson distribution calculations"""
e = 2.7182818285


def factorial(n):
    """ return fctt"""
    if n == 0:
        return 1
    total = 1
    for i in range(1, n + 1):
        total = total * i
    return total

class Binomial:
    """Poisson distribution stats class"""
    def __init__(self, data=None, n=1, p=0.5):
        """Initialize poisson distribution stats"""
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            var = []
            for i in range(len(data)):
                var.append((data[i] - mean) ** 2)
            variance = sum(var) / len(data)

            self.p = 1 - (variance / mean)
            self.n = int(round(mean/self.p))
            self.p = mean / self.n

    def pmf(self, k):
        """Calculates the value of the PMF"""
        if type(k) != int:
            k = int(k)
        if k < 0:
            return 0

        PDF = factorial(self.n) / (factorial(k) * factorial(self.n - k))
        return PDF * self.p**k * (1 - self.p)**(self.n - k)

    def cdf(self, k):
        """Calculates the value of the CDF"""
        if type(k) != int:
            k = int(k)
            if k < 0:
                return 0
            CDF = 0
            for i in range(k + 1):
                CDF += self.pmf(i)

            return CDF
