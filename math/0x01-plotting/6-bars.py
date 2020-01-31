#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

people = ['Farrah', 'Fred', 'Felicia']
width = 0.50
p1 = plt.bar(people, fruit[0], width=width, color='red')
p2 = plt.bar(people, fruit[1], width=width,
             color='yellow', bottom=fruit[0])
p3 = plt.bar(people, fruit[2], width=width,
             color='#ff8000', bottom=fruit[0]+fruit[1])
p4 = plt.bar(people, fruit[3], width=width,
             color='#ffe5b4', bottom=fruit[0]+fruit[1]+fruit[2])
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.xticks(people)
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0], p3[0], p4[0]),
           ('apples', 'bananas', 'oranges', 'peaches'))
plt.show()
