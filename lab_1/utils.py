import math
import numpy as np
from scipy.optimize import minimize

def array(f, numval, numdh):
    """Создать N-мерный массив.
    
    param: f - функция, которая приминает N аргументов.
    param: numval - диапазоны значений параметров функции. Список
    param: numdh - шаги для параметров. Список
    
    """
    def rec_for(f, numdim, numdh, current_l, l_i, arr):
        """Рекурсивный цикл.
        
        param: f - функция, которая приминает N аргументов.
        param: numdim - размерность выходной матрицы. Список
        param: numdh - шаги для параметров. Список
        param: current_l - текущая глубина рекурсии.
        param: l_i - промежуточный список индексов. Список
        param: arr - матрица, с которой мы работаем. np.array
        
        """
        for i in range(numdim[current_l]):
            l_i.append(i)
            if current_l < len(numdim) - 1:
                rec_for(f, numdim, numdh, current_l + 1, l_i, arr)
            else:
                args = (np.array(l_i) * np.array(numdh))
                arr[tuple(l_i)] = f(*args)
            l_i.pop()
        return arr
    numdim = [int(numval[i] / numdh[i]) + 1 for i in range(len(numdh))]
    arr = np.zeros(numdim)
    arr = rec_for(f, numdim, numdh, 0, [], arr)
    
    # Надо отобразить так x - j, y - i (для графиков), поэтому используем transpose
    
    arr = np.transpose(arr)
    return arr

def TDMA(a, b, c, f):
    """Метод прогонки.
    
    param: a - левая поддиагональ. 
    param: b - правая поддиагональ.
    param: c - центр.
    param: f - правая часть.
    """
    #a, b, c, f = map(lambda k_list: map(float, k_list), (a, b, c, f))
    
    alpha = [0]
    beta = [0]
    n = len(f)
    x = [0] * n

    for i in range(n - 1):
        alpha.append(-b[i] / (a[i] * alpha[i] + c[i]))
        beta.append((f[i] - a[i] * beta[i]) / (a[i] * alpha[i] + c[i]))

    x[n - 1] = (f[n - 1] - a[n - 1] * beta[n - 1]) / (c[n - 1] + a[n - 1] * alpha[n - 1])

    for i in reversed(range(n - 1)):
        x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]

    return x

def integral(arr, dh):
    val = 0.
    for i in range(0, len(arr) - 1):
        val += arr[i] + arr[i + 1]
        
    return val * dh / 2.

#----------------------------------------------------------------------------------------------------------------------

def criterion_1(model):
    val = 0.
    dt = model.dt
    
    # Вычисление нормы разности
    
    if len(model.p_arr) == 1:
        val = 10000000.
    else:
        arr = (model.p_arr[-1] - model.p_arr[-2]) ** 2
        val = integral(arr, dt)
        
    return val

def criterion_2(model):
    val_1, val_2 = 0., 0.
    val = 0.
    dh = model.dh
    
    # Вычисление нормы разности функционала
    
    if len(model.x_arr) == 1:
        val = 10000000.
    else:
        arr_1 = (model.x_arr[-1][-1,:] - model.y_arr) ** 2
        arr_2 = (model.x_arr[-2][-1,:] - model.y_arr) ** 2
        val_1 = integral(arr_1, dh)
        val_2 = integral(arr_2, dh)
        val = abs(val_1 - val_2) 
        
    return val

def criterion_3(model):
    val = 0.
    dt = model.dt
    
    # Вычисление нормы производной
    
    if len(model.psi_arr) == 1:
        val = 10000000.
    else:
        arr = (model.psi_arr[-1][:, -1] * model.a ** 2 * model.v) ** 2
        val = integral(arr, dt)
        
    return val

#----------------------------------------------------------------------------------------------------------------------

def f_alpha(alpha, model, ind):
    val = .0
    p_min, p_max = model.p_min, model.p_max
    p_arr = model.p_arr[-1]
    psi_l_arr = model.psi_arr[-1][:,-1]
    p_cond = p_arr - alpha * psi_l_arr
    p_cond[p_cond < p_min] = p_min
    p_cond[p_cond > p_max] = p_max
    
    matr = array(model.f, [model.l, model.T], [model.dh, model.dt])
    matr[0,:] = array(model.fi, [model.l], [model.dh])
    buf = 1. / (3. + 2. * model.dh * model.v)
    # Число уравнений
    eq_l = model.N - 1
    f = [0. for i in range(eq_l)]
    a2_dt_dh2 = model.a ** 2 * model.dt / model.dh ** 2
    
    # Решаем 1 задачу
        
    for j in range(0, model.M):
        # f
        f[0:-1] = [-matr[j, i] - model.dt * model.f_arr[j, i] for i in range(1, eq_l)]
        # Эта часть зависит от апроксимации, которую мы используем, поэтому стоит ввести функцию
        f[-1] = -matr[j, -2] - model.dt * model.f_arr[j, -2]
        f[-1] += -a2_dt_dh2 * 2. * model.dh * model.v * buf * p_cond[j + 1]

        matr[j + 1,1:eq_l + 1] = TDMA(model.a_arr, model.b, model.c, f)

        # Вычисляем первый и последний элементы
        # Эта часть зависит от апроксимации, которую мы используем, поэтому стоит ввести функцию
        matr[j + 1, 0] = 4. / 3. * matr[j + 1, 1] - 1. / 3. * matr[j + 1, 2]
        matr[j + 1, -1] = 4. * buf * matr[j + 1, -2]
        matr[j + 1, -1] -= buf * matr[j + 1, -3]
        matr[j + 1, -1] += 2. * model.dh * model.v * buf * p_cond[j + 1]
        
    arr = (matr[-1,:] - model.y_arr) ** 2
    val = integral(arr, model.dh)
    
    return val

def get_alpha_1(model, ind):
    val = 0.
    bnds = ((0, None),)
    res = minimize(f_alpha, 1., args=(model, ind), bounds=bnds, tol=10**-5)
    val = res.x
    
    return val    

def get_alpha_5(model, ind):
    val = 0.
    c, alpha = 1., 3./4.
    
    # Вычисление коэффициента
    
    val = c * (float(ind) + 1.) ** -alpha
    
    return val

def get_alpha_5_1(model, ind):
    val = 0.
    c, alpha = 10., 3./4.
    
    # Вычисление коэффициента
    
    val = c * (float(ind) + 1.) ** -alpha
    
    return val

#-----------------------------------------------------------------------------

# Класс модели для Л.Р №1
class Lab1OptCtrlModel():
    
    def __init__(self, p_d):
        self.a, self.l, self.v, self.T = p_d['a'], p_d['l'], p_d['v'], p_d['T']
        self.p, self.f = p_d['p(t)'], p_d['f(s, t)']
        self.p_min, self.p_max, self.R = p_d['p_min'], p_d['p_max'], p_d['R']
        self.fi, self.y  = p_d['fi(s)'], p_d['y(s)']
        
        self.dh, self.dt = p_d['dh'], p_d['dt']
        self.N, self.M = p_d['N'], p_d['M']
        
        self.p_arr = []
        self.p_arr.append(array(self.p, [self.T], [self.dt]))
        
        self.f_arr = array(self.f, [self.l, self.T], [self.dh, self.dt])
        
        self.x_arr = []
        self.x_arr.append(array(self.f, [self.l, self.T], [self.dh, self.dt]))
        self.x_arr[-1][0,:] = array(self.fi, [self.l], [self.dh])
        
        self.psi_arr = []
        self.psi_arr.append(array(self.f, [self.l, self.T], [self.dh, self.dt]))
        
        self.y_arr = array(self.y, [self.l], [self.dh])
        
        self.alpha = []
        self.final_step = 0
        self.err = []
        
    def solve(self, criterion, get_alpha, eps=10**-2, max_steps=None):
        
        self.eps = eps
        
        # Число уравнений
        eq_l = self.N - 1
        
        # Инициализация элементов для метода прогонки, которые постоянны
        self.a_arr, self.b, self.c = [0. for i in range(eq_l)], [0. for i in range(eq_l)], [0. for i in range(eq_l)]
        f = [0. for i in range(eq_l)]
        
        a2_dt_dh2 = self.a ** 2 * self.dt / self.dh ** 2
        buf = 1. / (3. + 2. * self.dh * self.v)
        
        # a
        self.a_arr[1:-1] = [a2_dt_dh2 for i in range(1, eq_l - 1)]
        # Эта часть зависит от апроксимации, которую мы используем, поэтому стоит ввести функцию
        self.a_arr[-1] = a2_dt_dh2 * (1. - buf)
        
        # b
        # Эта часть зависит от апроксимации, которую мы используем, поэтому стоит ввести функцию
        self.b[0] = 2. / 3. * a2_dt_dh2
        self.b[1:-1] = [a2_dt_dh2 for i in range(1, eq_l - 1)]
        
        # c
        # Эта часть зависит от апроксимации, которую мы используем, поэтому стоит ввести функцию
        self.c[0] = -2. / 3. * a2_dt_dh2 - 1.
        self.c[1:-1] = [-1. - 2. * a2_dt_dh2 for i in range(1, eq_l - 1)]
        
        # Эта часть зависит от апроксимации, которую мы используем, поэтому стоит ввести функцию
        self.c[-1] = -1. + a2_dt_dh2 * (4. * buf - 2.)
        
        # c для 2 задачи
        c_psi = [0. for i in range(eq_l)]
        
        # Эта часть зависит от апроксимации, которую мы используем, поэтому стоит ввести функцию
        c_psi[0] = -1. - 2. / 3. * a2_dt_dh2
        c_psi[1:-1] = [-1. - 2. * a2_dt_dh2 for i in range(1, eq_l - 1)]
        c_psi[-1] = -1. + a2_dt_dh2 * (4. * buf - 2.)
        
        # f для 2 задачи
        f_psi = [0. for i in range(eq_l)]
        
        ind = 0
        apr_max_steps = True
        self.err.append(criterion(self))
        while self.err[-1] > self.eps and apr_max_steps:
            
            # Решаем 1 задачу
            for j in range(0, self.M):

                # f
                f[0:-1] = [-self.x_arr[-1][j, i] - self.dt * self.f_arr[j, i] for i in range(1, eq_l)]
                # Эта часть зависит от апроксимации, которую мы используем, поэтому стоит ввести функцию
                f[-1] = -self.x_arr[-1][j, -2] - self.dt * self.f_arr[j, -2]
                f[-1] += -a2_dt_dh2 * 2. * self.dh * self.v * buf * self.p_arr[-1][j + 1]

                # Решаем задачу

                self.x_arr[-1][j + 1,1:eq_l + 1] = TDMA(self.a_arr, self.b, self.c, f)

                # Вычисляем первый и последний элементы
                # Эта часть зависит от апроксимации, которую мы используем, поэтому стоит ввести функцию
                self.x_arr[-1][j + 1, 0] = 4. / 3. * self.x_arr[-1][j + 1, 1] - 1. / 3. * self.x_arr[-1][j + 1, 2]
                self.x_arr[-1][j + 1, -1] = 4. * buf * self.x_arr[-1][j + 1, -2]
                self.x_arr[-1][j + 1, -1] -= buf * self.x_arr[-1][j + 1, -3]
                self.x_arr[-1][j + 1, -1] += 2. * self.dh * self.v * buf * self.p_arr[-1][j + 1]
            
            # Берем условия по времени для psi
            self.psi_arr[-1][-1,:] = 2. * (self.x_arr[-1][-1,:] - self.y_arr)
            
            # Решаем 2 задачу
            for j in range(self.M - 1, -1, -1):
                
                # f
                f_psi = [-self.psi_arr[-1][j + 1, i] for i in range(1, eq_l + 1)]
                
                # Решаем задачу
                self.psi_arr[-1][j,1:eq_l + 1] = TDMA(self.a_arr, self.b, c_psi, f_psi)

                # Вычисляем первый и последний элементы
                # Эта часть зависит от апроксимации, которую мы используем, поэтому стоит ввести функцию
                self.psi_arr[-1][j, 0] = 4. / 3. * self.psi_arr[-1][j, 1] - 1. / 3. * self.psi_arr[-1][j, 2]
                self.psi_arr[-1][j, -1] = 4. * buf * self.psi_arr[-1][j, -2]
                self.psi_arr[-1][j, -1] -= buf * self.psi_arr[-1][j, -3]
            
            # Вычисляем новое p по методу проекции градиента
            self.alpha.append(get_alpha(self, ind))
            self.p_arr.append(self.p_arr[-1] - self.alpha[-1] * self.a ** 2 * self.v * self.psi_arr[-1][:,-1])
            self.p_arr[-1][self.p_arr[-1] < self.p_min] = self.p_min
            self.p_arr[-1][self.p_arr[-1] > self.p_max] = self.p_max
            
            self.final_step = ind
            ind += 1
            err = criterion(self)
            print(err)
            self.err.append(err)
            
            if max_steps is None:
                apr_max_steps = True
            else:
                apr_max_steps = ind < max_steps
                
            # Для нового шага
            self.x_arr.append(array(self.f, [self.l, self.T], [self.dh, self.dt]))
            self.x_arr[-1][0,:] = array(self.fi, [self.l], [self.dh])
            self.psi_arr.append(array(self.f, [self.l, self.T], [self.dh, self.dt]))
        
        self.x_arr.pop()
        self.psi_arr.pop()
        return self
