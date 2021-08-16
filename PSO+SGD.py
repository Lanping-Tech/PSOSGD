import numpy as np
from sklearn import datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt

# 波士顿房价数据集，取前500条，并进行标准化
# housing_data = datasets.fetch_california_housing()
# input = preprocessing.scale(np.array(housing_data.data)[:500,:])
# output = preprocessing.scale(np.array(housing_data.target)[:500])

# 糖尿病数据集
input= datasets.load_diabetes()['data']
print()
output = preprocessing.scale(np.array(datasets.load_diabetes()['target']))

dim = input.shape[1]
# print(dim)
# print(input[:,0])
# print(output)

# 参数定义
number_iteration = 200
number_particle = 150
number_dimension = dim
weight_momentum = 0.5
weight_particle_optmized_location = 0.33
weight_global_optmized_location = 0.33
weight_gradient = 0.01

# --------------粒子群算法--------------------#
xlimit_max = 10
xlimit_min = -10
vlimit_max = 0.5
vlimit_min = -0.5


# 计算适应度
def fit(pop, input, output):
    fitness = np.zeros(number_particle)
    pre = np.matmul(pop, input.T)
    for i in range(number_particle):
        mse = pre[i, :] - output[:]
        fitness[i] = np.matmul(mse, mse.T)
    return fitness


# 更新速度和位置
def Update_V_P(v, pop, PopBest, GlobalBest):
    new_v = np.zeros(v.shape, dtype=float)
    new_pop = np.zeros(pop.shape, dtype=float)
    # print(GlobalBest.shape)
    for i in range(new_v.shape[0]):
        new_v[i, :] = weight_momentum * v[i, :] + weight_particle_optmized_location * np.random.rand(1) * (
                    PopBest[i, :] - pop[i, :]) + weight_global_optmized_location * np.random.rand(1) * (
                                  GlobalBest - pop[i, :])
        new_pop[i, :] = pop[i, :] + new_v[i, :]

    new_v[np.where(new_v > vlimit_max)] = vlimit_max
    new_v[np.where(new_v < vlimit_min)] = vlimit_min
    new_pop[np.where(new_pop > xlimit_max)] = xlimit_max
    new_pop[np.where(new_pop < xlimit_min)] = xlimit_min
    return new_v, new_pop


# 更新局部最优和个体最优
def Update_Best(PopBest, GlobalBest, fitness, new_fitness, pop):
    for i in range(pop.shape[0]):
        if new_fitness[i] < fitness[i]:
            PopBest[i, :] = pop[i, :]
    if np.min(new_fitness) < np.min(fitness):
        GlobalBest = pop[np.argmin(new_fitness)]
    return PopBest, GlobalBest


# 初始种群
pop_init = xlimit_min + (xlimit_max - xlimit_min) * np.random.rand(number_particle, number_dimension)
v_init = vlimit_min + (vlimit_max - vlimit_min) * np.random.rand(number_particle, number_dimension)
fitness = fit(pop_init, input, output)
# 局部最优和个体最优
PopBest = pop_init
Bestindex = np.argmin(fitness)
GlobalBest = pop_init[Bestindex, :]
v = v_init
pop = pop_init

# 迭代训练
fitnessBest_PSO = []
for iter in range(number_iteration):
    # 更新个体和种群，并处理边界问题
    v, pop = Update_V_P(v, pop, PopBest, GlobalBest)
    # 计算适应度
    new_fitness = fit(pop, input, output)
    # 更新局部最优和个体最优
    PopBest, GlobalBest = Update_Best(PopBest, GlobalBest, fitness, new_fitness, pop)
    fitness = new_fitness
    fitnessBest_PSO.append(np.min(fitness))
#print(GlobalBest)

# ------------------梯度下降法（结合动量）-------------------#
w = xlimit_min + (xlimit_max - xlimit_min) * np.random.rand(1, number_dimension)
v = None


# 计算梯度
def Update_w_v(w):
    # gw = -2*(output.reshape(500,1)-input.dot(w.T)).T.dot(input)
    gw = -2 * (output.reshape(442, 1) - input.dot(w.T)).T.dot(input)
    return gw


# 计算适应度
def fit_SGD(w):
    # mse = w.dot(input.T) - (output.reshape(1, 500))
    mse = w.dot(input.T) - (output.reshape(1, 442))
    return mse.dot(mse.T)

flag = True

# 迭代训练
fitnessBest_SGD = []
for iter in range(number_iteration):
    gw = Update_w_v(w)
    if flag:
        v = gw
        flag = False
    else:
        v = weight_momentum * v + (1 - weight_momentum) * gw  # 计算动量
    w = w - 0.2 * v  # 更新梯度
    print('iteration:', + iter, 'loss:', + fit_SGD(w)[0])
    fitnessBest_SGD.append(fit_SGD(w)[0])

#print(w)
plt.figure(1)
plt.plot(fitnessBest_PSO, label='PSO')
plt.plot(fitnessBest_SGD, label='SGD')

plt.figure(2)
plt.scatter(range(442), output)
plt.scatter(range(442), np.matmul(input, GlobalBest))
plt.title("PSO")

plt.figure(3)
plt.scatter(range(442), output)
plt.scatter(range(442), np.matmul(input, w.T))
plt.title("SGD")


# ---------------粒子群+梯度下降法（结合动量）--------------------#
# 利用梯度下降和粒子群更新速度和位置
def Update_V_P_gw(v, pop, PopBest, GlobalBest):
    new_v = np.zeros(v.shape, dtype=float)
    new_pop = np.zeros(pop.shape, dtype=float)
    # print(GlobalBest.shape)
    for i in range(new_v.shape[0]):
        # gw = Update_w_v(pop[i, :].reshape(1, 8))
        gw = Update_w_v(pop[i, :].reshape(1, 10))
        # print(gw)
        new_v[i, :] = weight_momentum * v[i, :] + weight_particle_optmized_location * np.random.rand(1) * (
                    PopBest[i, :] - pop[i, :]) + weight_global_optmized_location * np.random.rand(1) * (
                                  GlobalBest - pop[i, :]) - 0.00005 * gw
        new_pop[i, :] = pop[i, :] + new_v[i, :]

    new_v[np.where(new_v > vlimit_max)] = vlimit_max
    new_v[np.where(new_v < vlimit_min)] = vlimit_min
    new_pop[np.where(new_pop > xlimit_max)] = xlimit_max
    new_pop[np.where(new_pop < xlimit_min)] = xlimit_min
    return new_v, new_pop


# 初始种群
pop_init = xlimit_min + (xlimit_max - xlimit_min) * np.random.rand(number_particle, number_dimension)
v_init = vlimit_min + (vlimit_max - vlimit_min) * np.random.rand(number_particle, number_dimension)
# w = xlimit_min + (xlimit_max - xlimit_min) * np.random.rand(1, number_dimension)
fitness = fit(pop_init, input, output)
# 局部最优和个体最优
PopBest_PSO_SGD = pop_init
Bestindex = np.argmin(fitness)
GlobalBest_PSO_SGD = pop_init[Bestindex, :]
v = v_init
pop = pop_init

# 迭代训练
fitnessBest_PSO_SGD = []
for iter in range(number_iteration):
    v, pop = Update_V_P_gw(v, pop, PopBest, GlobalBest)
    # 计算适应度
    new_fitness = fit(pop, input, output)
    # 更新局部最优和个体最优
    PopBest_PSO_SGD, GlobalBest_PSO_SGD = Update_Best(PopBest, GlobalBest, fitness, new_fitness, pop)
    fitness = new_fitness
    print('iteration:', + iter, 'loss:', + np.min(fitness))
    fitnessBest_PSO_SGD.append(np.min(fitness))

# -----------------画图------------------#
plt.figure(1)
plt.plot(fitnessBest_PSO_SGD, label='PSO-SGD')
plt.xlabel("iter")
plt.ylabel("fitness")
plt.legend()

plt.figure(4)
plt.scatter(range(442), output)
plt.scatter(range(442), np.matmul(input, GlobalBest_PSO_SGD))
plt.title("PSO-SGD")

plt.show()
