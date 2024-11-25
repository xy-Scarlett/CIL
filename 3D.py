# import matplotlib.pyplot as plt
#
# # 生成数据
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]
#
# # 绘制折线图
# fig, ax = plt.subplots()
# ax.plot(x, y)
#
# # 设置x轴和y轴标签的字体大小
# ax.set_xlabel('X Label', fontsize=12)
# ax.set_ylabel('Y Label', fontsize=12)
#
# # 设置x轴和y轴刻度的字体大小
# ax.tick_params(axis='x', labelsize=10)
# ax.tick_params(axis='y', labelsize=10)
#
# # 设置图内的标题和图例的字体大小
# ax.set_title('Line Plot', fontsize=14)
# ax.legend(['Line'], fontsize=10)
#
# plt.show()

##################################################
#三维柱状图
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # 生成数据
# x = np.arange(1, 6)
# y = np.arange(1, 6)
# z = np.array([[3, 4, 1, 7, 5], [5, 3, 6, 1, 2], [2, 5, 3, 2, 6], [6, 2, 5, 4, 1], [1, 6, 2, 5, 3]])
#
# # 创建画布和坐标轴
# fig = plt.figure()
# ax = Axes3D(fig)
#
# # 绘制三维柱状图
# dx = dy = 0.5
# dz = z.ravel()%将多维数组转换为一维数组
# xpos, ypos = np.meshgrid(x, y)
# xpos = xpos.flatten()
# ypos = ypos.flatten()
# ax.bar3d(xpos, ypos, np.zeros(len(dz)), dx, dy, dz)
#
# # 设置坐标轴标签和标题
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_title('3D Bar Chart')
#
# # 显示图形
# plt.show()

##########################################设置字体大小
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # 生成数据
# x = [1, 2, 3]
# y = [4, 5, 6]
# z = [7, 8, 9]
#
# # 创建3D坐标系
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制三维柱状图
# ax.bar(x, y, zs=z, zdir='z', alpha=0.8)
#
# # 设置轴标签和字体大小
# ax.set_xlabel('X Label', fontsize=14)
# ax.set_ylabel('Y Label', fontsize=14)
# ax.set_zlabel('Z Label', fontsize=14)
#
# # 设置轴刻度字体大小
# ax.tick_params(axis='x', labelsize=12)
# ax.tick_params(axis='y', labelsize=12)
# ax.tick_params(axis='z', labelsize=12)
#
# # 显示图形
# plt.show()

######################################步长和范围
# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
#
# # 创建画布和3D坐标系
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 随机生成数据
# x = np.random.randint(0, 5, size=10)
# y = np.random.randint(0, 10, size=10)
# z = np.random.randint(0, 20, size=10)
#
# # 画三维柱状图
# ax.bar3d(x, y, np.zeros_like(z), 1, 2, z)
#
# # 设置x、y、z轴的取值范围和步长
# ax.set_xlim(0, 5)
# ax.set_ylim(0, 10)
# ax.set_zlim(0, 20)
#
# ax.set_xticks(range(6))
# ax.set_yticks(range(0, 11, 2))
# ax.set_zticks(range(0, 21, 4))
#
# ax.set_xinterval(1)
# ax.set_yinterval(2)
# ax.set_zinterval(4)
#
# # 显示图形
# plt.show()
#######################################三维可实现
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# #构造需要显示的值
# X=np.arange(0, 5, step=1)#X轴的坐标
# Y=np.arange(0, 9, step=1)#Y轴的坐标
# #设置每一个（X，Y）坐标所对应的Z轴的值，在这边Z（X，Y）=X+Y
# Z=np.zeros(shape=(5, 9))
# for i in range(5):
#   for j in range(9):
#     Z[i, j]=i+j
#
# xx, yy=np.meshgrid(X, Y)#网格化坐标
# X, Y=xx.ravel(), yy.ravel()#矩阵扁平化
# bottom=np.zeros_like(X)#设置柱状图的底端位值
# Z=Z.ravel()#扁平化矩阵
#
# width=height=1#每一个柱子的长和宽
#
# #绘图设置
# fig=plt.figure()
# ax=fig.gca(projection='3d')#三维坐标轴
# ax.bar3d(X, Y, bottom, width, height, Z, shade=True)#
# #坐标轴设置
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z(value)')
# # ax.set_zlim(5, 12)
# plt.show()

##################################
# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 自定义x, y, z轴的取值范围和步长
# x_step = 0.5
# y_step = 0.1
# z_step = 0.2
# x = np.arange(0, 5, x_step)
# y = np.arange(0, 1, y_step)
# z = np.arange(0, 2, z_step)
#
# # 根据x, y, z轴的尺度生成三维网格
# X, Y, Z = np.meshgrid(x, y, z)
#
# # 生成随机的三维数据
# data = np.random.randint(low=0, high=10, size=(len(x), len(y), len(z)))
#
# # 绘制三维柱状图
# ax.bar3d(X.ravel(), Y.ravel(), Z.ravel(), x_step, y_step, data.ravel(), alpha=0.8)
#
# # 设置坐标轴标签和字体大小
# ax.set_xlabel('X Label', fontsize=12)
# ax.set_ylabel('Y Label', fontsize=12)
# ax.set_zlabel('Z Label', fontsize=12)
#
# # 设置坐标轴刻度标签和字体大小
# ax.tick_params(axis='x', labelsize=10)
# ax.tick_params(axis='y', labelsize=10)
# ax.tick_params(axis='z', labelsize=10)
#
# plt.show()
###########################################调整刻度范围
# import matplotlib.pyplot as plt
# import numpy as np
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 生成一些测试数据
# x = np.random.randn(10)
# y = np.random.randn(10)
# z = np.random.randn(10)
#
# # 绘制三维柱状图
# ax.bar3d(x, y, 0, 0.5, 0.5, z)
#
# # 设置每个轴的刻度范围
# ax.set_xlim([-2, 2])
# ax.set_ylim([-3, 3])
# ax.set_zlim([0, 10])
#
# # 显示图形
# plt.show()

#############################颜色区分
# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
#
# # 生成数据
# x = np.array([0, 1, 2, 3])
# y = np.array([0, 1, 2])
# z = np.random.rand(3, 4)
#
# # 设置颜色映射
# colors = ['r', 'g', 'b']
# cmap = plt.cm.colors.ListedColormap(colors)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制三维柱状图
# for i in range(len(y)):
#     xs = x
#     ys = np.ones(len(x)) * y[i]
#     zs = z[i]
#     ax.bar(xs, zs, zs=ys, zdir='y', color=cmap(i))
#
# # 设置坐标轴标签
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# # 显示图像
# plt.show()
##################################3D曲面图
#
# # -*- coding: utf-8 -*-
# """
# Created on Thu Sep 24 16:17:13 2015
#
# @author: Eddy_zheng
# """
#
# from matplotlib import pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure()
# ax = Axes3D(fig)
# X = np.arange(-4, 4, 0.25)
# Y = np.arange(-4, 4, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
#
# # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
#
# plt.show()
###############################################3D离散图
# # -*- coding: utf-8 -*-
# """
# Created on Thu Sep 24 16:37:21 2015
#
# @author: Eddy_zheng
# """
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# data = np.random.randint(0, 255, size=[40, 40, 40])
#
# x, y, z = data[0], data[1], data[2]
# ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
# #  将数据点分成三部分画，在颜色上有区分度
# ax.scatter(x[:10], y[:10], z[:10], c='y')  # 绘制数据点
# ax.scatter(x[10:20], y[10:20], z[10:20], c='r')
# ax.scatter(x[30:40], y[30:40], z[30:40], c='g')
#
# ax.set_zlabel('Z')  # 坐标轴
# ax.set_ylabel('Y')
# ax.set_xlabel('X')
# plt.show()
#################################################
# import numpy as np
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x = np.arange(1,4+1,1)
# y = np.arange(1,4+1,1)
# hist = (np.random.randint(0, 1000, 16)).reshape((4,4)) # 生成16个随机整数
#
# zpos = 0
# color = ('r','g','b','y')
#
# # Construct arrays with the dimensions for the 16 bars.
# dx = dy = 0.5
# for i in range(4):
#     c = color[i]
#     ax.bar3d(range(4), [i] * 4, [0] * 4,
#              dx, dy, hist[i, :],
#              color=c)
#
# # 设置坐标轴的刻度
# ax.set_xticks(x)
# ax.set_xlabel('X')
# ax.set_yticks(y)
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# ax.view_init(elev=30,azim=-60)
# # 将三维的灰色背诵面换成白色
# # ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# # ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# # ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# plt.show()
####################################
# import numpy as np
# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d
#
# x = np.random.randint(0,40,10)
# y = np.random.randint(0,40,10)
# z = 80 * abs(np.sin(x+y))
# ax = plt.subplot(projection='3d')  # 三维图形
#
# for xx, yy, zz in zip(x,y,z):
#     color = np.random.random(3)   # 随机颜色元祖
#     ax.bar3d(
#         xx,            # 每个柱的x坐标
#         yy,            # 每个柱的y坐标
#         0,             # 每个柱的起始坐标
#         dx=5,          # x方向的宽度
#         dy=5,          # y方向的厚度
#         dz=zz,         # z方向的高度
#         color=color)   #每个柱的颜色
#
# plt.show()
##############################
# import matplotlib.pyplot as plt
# import numpy as np
#
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # 定义数据
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # 生成随机数据
# x, y = np.meshgrid(np.arange(0, 8, 1), np.arange(0, 8, 1))
# z = np.random.randint(0, 10, size=(8, 8))
# color = ['#4048f0', '#6878f0', '#8d9af1', '#b3bdf3', '#d9e1f5', '#f3c6ba', '#ee9070', '#e85d2c']
#
# # 绘制图形
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(x)):
#     for j in range(len(y)):
#         xs = x[i, j]
#         ys = y[i, j]
#         zs = z[i, j]
#         c = color[xs]
#         ax.bar3d(xs, ys, 0, 0.8, 0.8, zs, color=c)
#
# # 设置坐标轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# # 显示图形
# plt.show()

##############################################
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
#
# # 创建一个 3D 坐标系
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 三个给定的数组
# x_data = np.array([1, 2, 3, 4, 5])
# y_data = np.array([1, 2, 3, 4, 5])
# z1_data = np.array([1, 2, 3, 4, 5])
# z2_data = np.array([5, 4, 3, 2, 1])
# z3_data = np.array([1, 3, 2, 5, 4])
#
# # 为不同的柱体设置不同的颜色
# colors = ['b', 'g', 'r']
#
# # 绘制三个柱状图
# for i, z_data in enumerate([z1_data, z2_data]):
#     ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 0.5, 0.5, z_data,
#              color=colors[i % len(colors)], alpha=0.8)
#
# # 设置坐标轴标签和范围
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_xlim3d(0, 6)
# ax.set_ylim3d(0, 6)
# ax.set_zlim3d(0, 6)
#
# plt.show()
####################################
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
#
# # 构造数据
# x_data = np.array([1, 2, 3, 4, 5])
# y_data = np.array([1, 2, 3])
# z_data = np.array([[20, 30, 10, 15, 25], [15, 25, 10, 20, 30], [10, 20, 30, 25, 15]])
#
# # 计算柱子的位置
# xpos, ypos = np.meshgrid(x_data, y_data)
# xpos = xpos.flatten() - 0.5
# ypos = ypos.flatten() - 0.5
# zpos = np.zeros_like(xpos)
#
# # 设置柱子的大小
# # dx = 1.0 * np.ones_like(zpos)
# # dy = dx.copy()
# dx = 0.8
# dy = 0.4
# dz = z_data.flatten()
#
# # 设置颜色映射
# color_map = plt.get_cmap('coolwarm')
# #colors = ['#4048f0', '#6878f0', '#8d9af1', '#b3bdf3', '#d9e1f5', '#f3c6ba', '#ee9070', '#e85d2c', 'r']
# # 绘制3D柱状图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color_map(dx))
#
# # 设置坐标轴标签
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()
######################################
# import matplotlib.pyplot as plt
# import numpy as np
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 三个数组的数据
# x = np.array([0, 1, 2, 3])
# y = np.array([0, 1, 2])
# z = np.array([[10, 5, 3], [8, 10, 2], [6, 2, 9], [5, 6, 7]])
#
# # 生成颜色映射表
# cmap = plt.get_cmap('coolwarm')
#
# # 绘制三维柱状图
# dx, dy = 0.5, 0.5
# for i, j in enumerate(x):
#     xs = np.arange(len(y))
#     ys = z[i]
#     cs = [cmap((j-x[0])/(x[-1]-x[0]))]*len(ys) # 根据x轴刻度不同，柱体颜色不同
#     ax.bar3d(j*dx, xs, 0, dx, dy, ys, color=cs)
#
# # 设置图形参数
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()
#######################################################
# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
#
# # 创建 3D 图形对象
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 创建数据
# x_data = np.array([1, 2, 3, 4, 5])
# y_data = np.array([1, 2, 3])
# z_data = np.array([[10, 5, 1], [8, 3, 2], [7, 4, 3], [9, 3, 1], [5, 2, 2]])
#
# # 计算每个柱体的高度和位置
# xpos, ypos = np.meshgrid(x_data, y_data)
# xpos = xpos.flatten()
# ypos = ypos.flatten()
# zpos = np.zeros_like(xpos)
# dz = z_data.flatten()
#
# # 设置每个柱体的颜色
# colors = ['r', 'b', 'g', 'y', 'c'] * len(y_data)  # 重复颜色列表，保证和数据对应
#
# # 绘制柱状图
# ax.bar3d(xpos, ypos, zpos, 1, 1, dz, color=colors, edgecolor='k')
#
# # 设置图形参数
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()
####################################
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # 生成数据
# x = np.array([1, 2, 3, 4, 5, 6, 7])
# y = np.array([1, 2, 3, 4])
# z = np.array([[1, 2, 3, 4, 5, 6, 7],[1, 2, 3, 4, 5, 6, 7],[1, 2, 3, 4, 5, 6, 7],[1, 2, 3, 4, 5, 6, 7]])
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 画图
# dx = 0.8
# dy = 0.5
# dz = 0.1
# x_offset = -dx / 2
# y_offset = -dy / 2
# colors = ['#4048f0', '#6878f0', '#8d9af1', '#b3bdf3', '#d9e1f5', '#f3c6ba', '#ee9070', '#e85d2c']
# for i, xi in enumerate(x):
#     for j, yj in enumerate(y):
#         color = plt.cm.viridis(z[j][i])
#         ax.bar3d(xi + x_offset, yj + y_offset, 0, dx, dy, z[j][i], color=colors[i], linewidth=0.5, edgecolor='black')
#
# # 调整坐标轴范围
# ax.set_xlim([min(x) - dx, max(x) + dx])
# ax.set_ylim([min(y) - dy, max(y) + dy])
# ax.set_zlim([0, max(np.ravel(z)) + dz])
#
# # 设置坐标轴标签
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()
######################################
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import LinearLocator
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# x = [1, 2, 3, 4, 5]
# y = [1, 2, 3]
# z = np.array([[10, 20, 30, 40, 50],
#               [60, 70, 80, 90, 100],
#               [10, 20, 30, 40, 50]])
#
# z_scale = 2 # 缩放比例
# z *= z_scale
#
# xpos, ypos = np.meshgrid(x, y)
#
# xpos = xpos.flatten()
# ypos = ypos.flatten()
# zpos = np.zeros_like(xpos)
#
# dx = 0.5 * np.ones_like(zpos)
# dy = dx.copy()
# dz = z.flatten()
#
# colors = ['b', 'g', 'r', 'c', 'm']
#
# for i, (xp, yp, zp, dc) in enumerate(zip(xpos, ypos, dz, colors)):
#     ax.bar3d(xp, yp, zp, dx[i], dy[i], dz[i], color=dc, edgecolor='k', lw=0.5)
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_zlim(0, np.max(z))  # 设置 z 轴刻度范围
# ax.zaxis.set_major_locator(LinearLocator(10))  # 设置 z 轴刻度数量
#
# plt.show()
####################################
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 创建数据
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([1, 2, 3])
z_data = np.array([[20, 30, 10, 15, 20],
                   [30, 20, 15, 10, 25],
                   [10, 15, 25, 30, 5]])

# 设置颜色列表
colors = []
for i in range(len(x_data)):
    for j in range(len(y_data)):
        color = (z_data[j][i] - np.min(z_data)) / (np.max(z_data) - np.min(z_data))
        colors.append((color, 1-color, 0.2))

# 绘制图形
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
xpos, ypos = np.meshgrid(x_data, y_data, indexing='xy')
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = z_data.flatten()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, edgecolor='black', linewidth=0.5)

# 设置坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
