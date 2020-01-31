#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

xticks = np.arange(0, 110, step=10)
plt.hist(student_grades, bins=xticks, align='mid', edgecolor='k')
plt.xticks(xticks)
plt.title("Project A")
plt.xlabel('Grades')
plt.ylabel('Fraction Remaining')
plt.axis([0, 100, 0, 30])
plt.show()
