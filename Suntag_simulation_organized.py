"""
@author: ndemirpehlivan
"""

import numpy as np
import random as r
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numba import njit
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,mark_inset)

N = 1200
a = 0.1  # initiation rate
p = 6 # elongation rate
b = p # termination rate
io=1
d=20
interval=0.1
tau_c = N/p
dt = int(tau_c/interval)

t = 20000000

# Function to simulate TASEP

def TASEP(N, t, sample_interval=1):
    sample = t//sample_interval
    dt_gillespie=np.zeros(t)
    gillespie_time=np.zeros(t)
    light_intensity = np.zeros(sample)
    light_intensity2 = np.zeros(sample)
    light_intensity3 = np.zeros(sample)
    sampled_time = np.zeros(sample)
    l = np.zeros(N, dtype=bool)
    rhos = np.zeros(sample)
    for i in range(1,t):
        forced = False
        S = a * (1 - l[0]) + b * l[-1] + p * np.sum(l[:-1] * (1 - l[1:]))
        k = np.random.uniform(0, S)
        Sp = 0
        dt_gillespie[i] = (-(np.log(1-(r.uniform(0,1)))/S))
        gillespie_time[i] = gillespie_time[i - 1] + dt_gillespie[i]
        
        if not l[0]:
            Sp += a
            if k <= Sp and not forced:
                l[0] = True
                forced = True
                
                
        if l[0] and not l[1]:
            Sp += p
            if k <= Sp and not forced:
                l[1] = True
                l[0] = False
                forced = True

        for j in range(1, N - 2):
            if l[j] and not l[j + 1]:
                Sp += p
                if k <= Sp and not forced:
                    l[j] = False
                    l[j + 1] = True
                    forced = True
                    break

        if l[N - 2] and not l[N - 1]:
            Sp += p
            if k <= Sp and not forced:
                l[N - 1] = True
                l[N - 2] = False
                forced = True

        if l[N - 1]:
            Sp += b
            if k <= Sp and not forced:
                l[N - 1] = False
                forced = True
                
                
        if forced:
            positions = np.where(l)[0]+1
            gfp = np.where(positions<=600, positions//d, 30)
            gfp2 = np.where(positions<=900, positions//d, 45)
            gfp3 = positions//d
            
            if i % sample_interval ==0:
                
                index=i//sample_interval
                light_intensity[index] = sum(gfp)
                light_intensity2[index] = sum(gfp2)
                light_intensity3[index] = sum(gfp3)
                sampled_time[index] = gillespie_time[i]

                rhos[index] = np.mean(l)
        
        if i % 500000 == 0:
            print(f"Iteration {i}")
        
    return rhos, light_intensity, light_intensity2, light_intensity3, sampled_time
   

# # Perform simulations
density, light600, light900, light1200, time = TASEP(N, t)

# theorical = [a/p] * len(density)
# plt.plot(time, density, 'b', label = 'The Density')
# plt.plot(time, theorical, color='orange', linewidth = 0.8, label='Mean field Result')
# plt.plot(time, theorical)
# plt.xticks(fontsize = 12)
# plt.yticks(fontsize = 12)
# plt.legend()
# plt.show()

# moyenne = [np.mean(light600)]*t
# plt.plot(time, light600, label = '600/1200')
# plt.plot(time, light740, label = '740/1200')
# plt.plot(time, light900, label = '900/1200')
# plt.plot(time, light1040, label = '1040/1200')
# plt.plot(time, light1200, label = '1200/1200')
# plt.xticks(fontsize = 12)
# plt.yticks(fontsize = 12)
# plt.legend()
# plt.show()

simulation_time = time
simulation_light = light600
simulation_light2 = light900
simulation_light3 = light1200

def resampling_time(time, light, interval):
    new_time_points = np.arange(time[0], time[-1] + interval, interval)
    interpolator = interp1d(time, light, kind='linear', fill_value='extrapolate')
    return new_time_points, interpolator(new_time_points)

def analytical_average(c, v, M):
    return c * M * ((N / v) - (M * d / (2 * v)))

arranged_time, arranged_signal = resampling_time(simulation_time, simulation_light, interval)
arranged_time, arranged_signal2 = resampling_time(simulation_time, simulation_light2, interval)
arranged_time, arranged_signal3 = resampling_time(simulation_time, simulation_light3, interval)

# plt.plot(simulation_time, simulation_light)
# plt.scatter(arranged_time, arranged_signal, facecolor='none', edgecolor='r')
# plt.show()

simulation_lights = [arranged_signal, arranged_signal2, arranged_signal3]

@njit
def correlation_function(light, dt):
    correlation_light = np.zeros(dt)
    delta_light = light - np.mean(light)
    for o in range(dt):
            correlation_light[o] = np.mean(delta_light[:len(light)-o] * delta_light[o:])
    return correlation_light

@njit
def model_function(dt, c, v, M):
    T = N / v
    tf = M * d / N
    K = c * T * (M ** 2) / (6 * (tf ** 2))

    C_an = []
    for i in range(dt):
        tauf = i / T
        if 0 <= tauf <= 1 - tf:
            C_an.append(K * ((tauf ** 3) - (3 * tf * (tauf ** 2)) - (3 * tauf * (tf ** 2)) + (6 - 4 * tf) * (tf ** 2)))
        elif 1 - tf <= tauf <= tf:
            C_an.append(K * ((tauf ** 3) + (3 * tf * (tf - 2) * tauf) + (3 - tf ** 2) * tf))
        elif tf <= tauf <= 1:
            C_an.append(K * 3 * tf * (1 - tauf) ** 2)
        else:
            C_an.append(0)

    return np.array(C_an)

@njit
def model_function_normalized(dt, c, v, M):
    T = N / v
    tf = M * d / N
    c0 = (6 - 4 * tf) * (tf ** 2)
    K = c * T * (M ** 2) / (6 * (tf ** 2))

    C_an = []
    for i in range(dt):
        tauf = i / T
        if 0 <= tauf <= 1 - tf:
            C_an.append(((tauf ** 3) - (3 * tf * (tauf ** 2)) - (3 * tauf * (tf ** 2)) + (6 - 4 * tf) * (tf ** 2))/c0)
        elif 1 - tf <= tauf <= tf:
            C_an.append(((tauf ** 3) + (3 * tf * (tf - 2) * tauf) + (3 - tf ** 2) * tf)/c0)
        elif tf <= tauf <= 1:
            C_an.append((3 * tf * (1 - tauf) ** 2)/c0)
        else:
            C_an.append(0)

    return np.array(C_an)

@njit
def analytical_average(c, v, M):
    return c * M * ((N / v) - (M * d / (2 * v)))

@njit
def analytical_var(c, v, M):
    return c * M ** 2 * ((N / v) - (M * d * 2 / (3 * v)))

@njit
def analytical_std(c, v, M):
    return np.sqrt(c * M ** 2 * ((N / v) - (M * d * 2 / (3 * v))))

correlation_results = []
correlation_results_normalized = []
M_values_analytical = np.arange(30,61,1)
M_values_simulation = np.arange(30,61,15)
sample_number = 3

for i in range(sample_number):
    correlation_result = correlation_function(simulation_lights[i], dt)
    correlation_results.append(correlation_result)
    correlation_results_normalized.append(correlation_result/correlation_result[i])
    print(i)

x = np.linspace(0, tau_c, dt)
x1 = np.linspace(0, tau_c, int(tau_c))
ana_results = [model_function(int(tau_c), a, p, M) for M in M_values_analytical]
ana_results_normalized = [model_function_normalized(int(tau_c), a, p, M) for M in M_values_analytical]
      
plt.rcParams['text.usetex'] = True 
plt.rcParams['font.family'] = 'serif'
fig, ax1 = plt.subplots(figsize=(11, 9))
colors = ['blue', 'orange', 'green', 'magenta', 'pink']
ax1.plot(x, correlation_results[0], color=colors[0], label=r'$x_f = 600$')
ax1.plot(x1, ana_results[0], color=colors[0], linestyle='--')
ax1.plot(x, correlation_results[1], color=colors[1], label=r'$x_f = 900$')
ax1.plot(x1, ana_results[1], color=colors[1], linestyle='--')
ax1.plot(x, correlation_results[2], color=colors[2], label=r'$x_f = 120$')
ax1.plot(x1, ana_results[2], color=colors[2], linestyle='--')
ax1.set_xlabel(r'$\tau~(s)$',fontsize=30)
ax1.set_ylabel(r'$C(\tau)$', fontsize=30)
ax1.yaxis.get_offset_text().set_fontsize(26)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
ax1.grid(True, alpha=0.4)
ax1.legend(loc='lower left',fontsize=18)
ax2 = plt.axes([0,0,1,1])
ip = InsetPosition(ax1, [0.4985,0.5,0.5,0.5])
ax2.set_axes_locator(ip)
for i in range(sample_number):
    ax2.plot(x, correlation_results_normalized[i], color=colors[i])
    ax2.plot(x1, ana_results_normalized[i], color=colors[i], linestyle='--')
ax2.set_xlabel(r'$\tilde{\tau}$', fontsize=30)
ax2.set_ylabel(r'$C(\tau)/C(0)$', fontsize=30)
ax2.grid(True, alpha=0.4)
plt.show()

mean_values = [np.mean(simulation_lights[i]) for i in range (sample_number)]
analytical_means = [analytical_average(a, p, M) for M in M_values_analytical]
variances = [np.var(simulation_lights[i]) for i in range(sample_number)]
analytical_variances = [analytical_var(a, p, M) for M in M_values_analytical]

plt.rcParams['text.usetex'] = True 
plt.rcParams['font.family'] = 'serif'
plt.figure(figsize=(9, 7))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.plot(M_values_simulation, mean_values, 'v', color='blue', label=r'$MC~Estimate$')
plt.plot(M_values_analytical, analytical_means, '-', color='blue', label=r'$Ballistic~Model')
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(loc='lower right', fontsize=22)
plt.show()

plt.rcParams['text.usetex'] = True 
plt.rcParams['font.family'] = 'serif'
plt.figure(figsize=(9, 7))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.plot(M_values_simulation, variances, 's', color='green', label=r'$MC~Estimate$')
plt.plot(M_values_analytical, analytical_variances, color='green', linestyle='--', label=r'$Ballistic~Model')
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(loc='lower right', fontsize=22)
plt.show()

a = 0.1 
p = 6

data1 = np.column_stack((time, density))
data = np.column_stack((time, light600, light900, light1200))

np.savetxt(f'a={a}_p={p}_fluo.csv', data, delimiter=',', header='time, fluo-600, fluo-900, fluo-1200', comments='')
np.savetxt(f'a={a}_p={p}_density.csv', data1, delimiter=',', header='time, density', comments='')

data = np.column_stack((arranged_time, arranged_signal, arranged_signal2, arranged_signal3))
np.savetxt(f'arranged_a={a}_p={p}_dt={interval}.csv', data, delimiter=',', header='time, fluo-600, fluo-900, fluo-1200', comments='')

data = np.column_stack((correlation_results[0], correlation_results_normalized[0]))

np.savetxt(f'correlation_a={a}_p={p}_dt={interval}_M=45.csv', data, delimiter=',', header='non_normalized, normalized', comments='')











