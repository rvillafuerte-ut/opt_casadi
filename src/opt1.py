import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# 1. Definición de variables simbólicas
# TODO: Define 'x' (estado) y 'u' (control) como escalares simbólicos usando SX
x = ca.SX.sym('x')  # Estado
u = ca.SX.sym('u')  # Control

# 2. Dinámica del sistema (x_dot = u)
# En sistemas más complejos aquí iría f(x,u)
x_dot = u 

# 3. Función de costo (Lagrange term)
# Queremos minimizar el uso de energía: L = u^2
L = u**2 

# 4. Crear integrador numérico (Discretización)
# Vamos a usar Runge-Kutta 4 fijo para dar un paso de tiempo 'dt'
dt = 0.1
# RK4 manual:
k1 = x_dot
k2 = u  # Porque x_dot no depende de x, simplificamos. Si fuera f(x,u): sustituir
k3 = u
k4 = u
x_next = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
L_step = L * dt # Costo acumulado en un paso

# TODO: Crea una función CasADi que tome [x, u] y devuelva [x_next, L_step]
# Tip: ca.Function('nombre', [entradas], [salidas])
F_rk4 = ca.Function('F_rk4', [x, u], [x_next, L_step])

# --- Configuración del Problema de Optimización (NLP) ---
N = 50 # Horizonte de tiempo
opti = ca.Opti() # Opti stack: una interfaz más amigable que nlpsol puro

# Variables de decisión para todo el horizonte
# TODO: Crea variables de decisión vectores para X (tamaño N+1) y U (tamaño N)
X = opti.variable(N+1) 
U = opti.variable(N)

# Función objetivo: Suma de costos L_step
obj = 0
for k in range(N):
    # Llamamos a nuestra función de dinámica F_rk4
    # F_rk4 devuelve una lista/objeto, accedemos por índice si es necesario
    res = F_rk4(X[k], U[k])
    x_next_k = res[0]
    L_step_k = res[1]
    
    # Restricción de dinámica: El siguiente X debe ser igual al calculado
    opti.subject_to(X[k+1] == x_next_k)
    
    # Acumular costo
    obj = obj + L_step_k

# TODO: Define la función objetivo en el solver
# Tip: opti.minimize(...)
opti.minimize(obj)

# Restricciones de frontera
opti.subject_to(X[0] == 0)   # Empezar en 0
opti.subject_to(X[N] == 10)  # Terminar en 10
opti.subject_to(opti.bounded(-5, U, 5)) # Control limitado entre -5 y 5

# Resolver
opti.solver('ipopt')
sol = opti.solve()

# Resultados
print("Costo final:", sol.value(obj))

# Graficar (opcional)
plt.plot(sol.value(X), label='Estado x')
plt.step(range(N), sol.value(U), label='Control u', where='post')
plt.legend()
plt.show()