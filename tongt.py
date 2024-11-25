# import numpy as np
# import matplotlib.pyplot as plt
# from simplicial import SimplicialComplex
# from mogutda import SimplicialComplex, AlphaComplex
#
# n = 30 #number of points to generate
#
# #generate space of parameter
# theta = np.linspace(0, 2.0*np.pi, n)
# a, b, r = 0.0, 0.0, 5.0
# x = a + r*np.cos(theta)
# y = b + r*np.sin(theta)
#
# x2 = np.random.uniform(-0.75,0.75,n) + x #add some "jitteriness" to the points
# y2 = np.random.uniform(-0.75,0.75,n) + y
# fig, ax = plt.subplots()
# ax.scatter(x2,y2)
#
# data = np.array([[1,4],[1,1],[6,1],[6,4]])
# #for example... this is with a small epsilon, to illustrate the presence of a 1-dimensional cycle
# graph = SimplicialComplex.generate_graph(raw_data=data, epsilon=5.1)
# ripsComplex = SimplicialComplex.rips(nodes=graph[0], edges=graph[1], k=3)
# SimplicialComplex.draw(origData=data, ripsComplex=ripsComplex, axes=[0,7,0,5])
######################################
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x = [2, 4, 6, 8, 10]
y1 = [93.65, 49.3, 32.78, 24.75, 19.34]
y2 = [91.7, 72.85, 64.35, 50.9, 56.6]
y3 = [90.95, 56.8, 37.27, 31.08, 25.83]
curve1, = ax.plot(x, y1, label='Curve 1')
curve2, = ax.plot(x, y2, label='Curve 2')
curve3, = ax.plot(x, y3, label='Curve 3')

# 将图例分成两部分，其中一部分放置在左下角，另一部分放置在右上角
legend1 = ax.legend(handles=[curve1, curve3], loc='lower left', borderaxespad=0)
legend2 = ax.legend(handles=[curve2], loc='upper right', borderaxespad=0)

# 将两个图例框添加到Axes对象中
ax.add_artist(legend1)
ax.add_artist(legend2)

plt.show()