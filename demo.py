from sympy.polys.groebnertools import groebner
from sympy.polys.orderings import lex, grlex, grevlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ, RR, GF

from f5b_gpu import local_is_groebner, local_is_minimal, local_is_reduced

import f5b_gpu

import time, csv, sys, os, datetime, re


input_data = {'file': 'chemequs_5', 'vars': ['x1', 'x2', 'x3', 'x4', 'x5'], 'system': '[4006491052808310*x1*x2 + 526610813156747*x1 - 473964757724582*x5,5313248634539330*x1*x2 + 349185128215561*x1 + 4313634906030440*x2**2 + 7431331204908770*x2*x3**2 + 806989849429520*x2*x3 + 149987275886495*x2*x4 + 132605295558012*x2 - 1047588595859020*x5,18874386880707400*x2*x3**2 + 1024812258191070*x2*x3 + 478801780852764*x3**2 + 101454213888243*x3 - 1064282665103090*x5,169336224248364*x2*x4 + 19504011585425050*x4**2 - 4730926576002525*x5,13941268403110100*x1*x2 + 1832432069217330*x1 + 11318412924828000*x2**2 + 38997679215146700*x2*x3**2 + 4234871305042080*x2*x3 + 787093928408356*x2*x4 + 695877849581671*x2 + 989285552709683*x3**2 + 419243169406760*x3 + 45328426231998600*x4**2 - 203370256289491000000]'}

fname = input_data['file']
var_str_list = input_data['vars']
sys_string = input_data['system']

r_v = xring(var_str_list, GF(65521), grevlex)

var_list = []
for i in range(len(var_str_list)):
    new_var = r_v[1][i]
    exec(var_str_list[i] + ' = ' + 'new_var')
    exec('var_list.append({})'.format(var_str_list[i]))

I = None
R = r_v[0]

exec('I = ' + sys_string)

os.system('clear')

print('-------------------------------------------------')
print('Now computing Grobner Basis for the {} system.\n'.format(fname))
time.sleep(2)
print('Input System:')
time.sleep(2)
for pol in I:
    print('    ' + str(pol))
    time.sleep(1)

print('-------------------------------------------------')

print("Starting SymPy's F5B Implementation...")

start_f5b = time.time()
res_f5b = groebner(I, R, method="f5b")
end_f5b = time.time()
f5b_runtime = end_f5b - start_f5b
print('-------------------------------------------------')
print('Sympy F5B Time: ' + str(f5b_runtime))
print('Result: ')
# print(res_f5b)
for pol in res_f5b:
    print('    ' + str(pol))
    time.sleep(1)
print('-------------------------------------------------')

time.sleep(10)
os.system('clear')

print("Starting GPU-powered Implementation (Critical Pair Only)")

start_cp_gpu = time.time()
res_cp_gpu = f5b_gpu.run(I, R, True, False)  # Only GPU CP
end_cp_gpu = time.time()
cp_gpu_runtime = end_cp_gpu - start_cp_gpu
print('-------------------------------------------------')
print('GPU CP Time: ' + str(cp_gpu_runtime))
# print(res_cp_gpu)
for pol in res_cp_gpu:
    print('    ' + str(pol))
print('-------------------------------------------------')

time.sleep(10)
os.system('clear')

print("Starting GPU-powered Implementation (S-Polynomial Only)")

start_sp_gpu = time.time()
res_sp_gpu = f5b_gpu.run(I, R, False, True)  # Only GPU Spoly
end_sp_gpu = time.time()
sp_gpu_runtime = end_sp_gpu - start_sp_gpu
print('-------------------------------------------------')
print('GPU SP Time: ' + str(sp_gpu_runtime))
# print(res_sp_gpu)
for pol in res_sp_gpu:
    print('    ' + str(pol))
print('-------------------------------------------------')

time.sleep(10)
os.system('clear')

print("Starting GPU-powered Implementation (Critical Pair and S-Polynomial)")

start_cpsp_gpu = time.time()
res_cpsp_gpu = f5b_gpu.run(I, R, True, True)  # Both CP and SPoly
end_cpsp_gpu = time.time()
cpsp_gpu_runtime = end_cpsp_gpu - start_cpsp_gpu
print('-------------------------------------------------')
print('GPU CP+SP Time: ' + str(cpsp_gpu_runtime))
# print(res_cpsp_gpu)
for pol in res_cpsp_gpu:
    print('    ' + str(pol))
print('-------------------------------------------------')

time.sleep(10)
os.system('clear')

print('FINAL TIMES:\n----------------------------------------------------\n|\n|')
print('|    SymPy F5B: {} sec'.format(str(f5b_runtime)))
print('|    Critical Pair Only on GPU: {} sec'.format(str(cp_gpu_runtime)))
print('|    S-Polynomial Only on GPU: {} sec'.format(str(sp_gpu_runtime)))
print('|    Both S-Polynomial and Critical Pair on GPU: {} sec'.format(str(cpsp_gpu_runtime)))
print('|\n|\n----------------------------------------------------')
