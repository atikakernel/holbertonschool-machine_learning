#!/usr/bin/env python3
"""Poisson distribution calculations"""


class Normal:
    """Poisson distribution stats class"""
    def __init__(self, data=None, mean=0, stddev=1.):
        """Initialize poisson distribution stats"""
        self.mean = float(mean)
        self.stddev = float(stddev)

        if data is None:
            if self.stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if isinstance(data, list):
                if len(data) > 1:
                    self.mean = sum(data) / len(data)
                    n = len(data)
                    variance = sum([(n - self.mean) ** 2 for n in data]) / n
                    self.stddev = variance ** 0.5
                else:
                    raise ValueError("data must contain multiple values")

            else:
                raise TypeError("data must be a list")

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""

        return self.stddev * z + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        pi = 3.1415926536
        e = 2.7182818285
        aux = ((x - self.mean) / self.stddev)**2

        return (1 / (self.stddev * (2 * pi)**(1/2))) * e**((-1/2) * aux)

    def cdf(self, x):
        """Cumulative Distribution function"""
        e = 2.7182818285
        pi = 3.1415926536
        b = ((x - self.mean)/(self.stddev * (2 ** (1/2))))
        erf1 = (2/(Normal.pi ** (1/2)))
        erf2 = b - (b ** 3)/3 + (b ** 5)/10 - (b ** 7)/42 + (b ** 9) / 216
        erf = erf1 * erf2
        return 0.5 * (1 + erf)
