import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Definición de la Planta (Física) ---
nx = 12  # Estados: [pos xyz; vel xyz; ori rpy; vel rpy]
nu = 4  # Un control: [fuerza total; 3 torques rpy]

x = ca.SX.sym('x', nx) # Estados 
u = ca.SX.sym('u', nu)
t = ca.SX.sym('t')    # Tiempo
xref = ca.SX.sym('xref', 3) # Referencia posición

roll = x[6]
pitch = x[7]
yaw = x[8]

R=ca.vertcat(ca.horzcat(ca.cos(yaw)*ca.cos(pitch), ca.cos(yaw)*ca.sin(pitch)*ca.sin(roll)-ca.sin(yaw)*ca.cos(roll), ca.cos(yaw)*ca.sin(pitch)*ca.cos(roll)+ca.sin(yaw)*ca.sin(roll)),
               ca.horzcat(ca.sin(yaw)*ca.cos(pitch), ca.sin(yaw)*ca.sin(pitch)*ca.sin(roll)+ca.cos(yaw)*ca.cos(roll), ca.sin(yaw)*ca.sin(pitch)*ca.cos(roll)-ca.cos(yaw)*ca.sin(roll)),
               ca.horzcat(-ca.sin(pitch), ca.cos(pitch)*ca.sin(roll), ca.cos(pitch)*ca.cos(roll)))

# Parámetros físicos
m = 1.587; l = 0.243; g = 9.81
Ix = 0.0213; Iy = 0.02217; Iz = 0.0282
I = ca.diag(ca.vertcat(Ix, Iy, Iz))
I_inv = ca.diag(ca.vertcat(1/Ix, 1/Iy, 1/Iz))
# Dinámica: x_dot = f(x, u)

dx_pos = x[3:6]  # Velocidad lineal
dx_velxyz = ca.vertcat(0, 0, -g) + R@ca.vertcat(0, 0, u[0])/m  # Aceleración lineal
dx_ori = x[9:12]  # Velocidad angular
dx_velrpy = I_inv@(ca.cross(-dx_ori, I@dx_ori) + u[1:4])  # Aceleración angular

### CAMBIO 1: Usar vertcat para crear el vector de derivadas
x_dot = ca.vertcat(dx_pos, dx_velxyz, dx_ori, dx_velrpy)

# Función de dinámica simbólica (útil para el integrador)
f_dyn = ca.Function('f_dyn', [x, u], [x_dot])

# referencia 
ref_x = 5 * ca.cos(t/2) - 5
ref_y = 5 * ca.sin(t/2)
ref_z = t/2 
xref = ca.vertcat(ref_x, ref_y, ref_z)

# --- 2. Integrador Numérico (RK4 Vectorizado) ---
dt = 0.1

Q_pos = 100
R_ctrl = 0.1
L = R_ctrl * ca.dot(u, u) + Q_pos * ca.dot(x[0:3] - xref, x[0:3] - xref)

### CAMBIO 2: RK4 genérico llamando a f_dyn
# Ahora k1, k2... son vectores, CasADi lo maneja solo.
k1 = f_dyn(x, u)
k2 = f_dyn(x + dt/2 * k1, u)
k3 = f_dyn(x + dt/2 * k2, u)
k4 = f_dyn(x + dt * k3, u)
x_next = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
L_step = L

F_rk4 = ca.Function('F_rk4', [x, u, t], [x_next, L_step])


# --- 3. Optimización (NLP) ---
N = 300
opti = ca.Opti()

### CAMBIO 3: Variables son Matrices (nx filas, N+1 columnas)
X = opti.variable(nx, N+1) 
U = opti.variable(nu, N)

obj = 0
for k in range(N):
    # Al llamar a la función, pasamos las COLUMNAS k de las matrices
    res = F_rk4(X[:, k], U[:, k], k*dt)
    x_next_k = res[0]
    L_step_k = res[1]
    
    # Restricción dinámica (columna k+1 debe ser igual a x_next)
    opti.subject_to(X[:, k+1] == x_next_k)
    
    obj += L_step_k

opti.minimize(obj)

# --- 4. Restricciones Físicas ---
# Estado inicial (Péndulo abajo en reposo: 0, 0)
opti.subject_to(X[:, 0] == ca.vertcat(0,0,0, 0,0,0, 0,0,0, 0,0,0))

# Límites de control (Torque máximo)
u_min = [0, -1.2, -1.2, -0.2]
u_max = [36, 1.2, 1.2, 0.2]
opti.subject_to(opti.bounded(u_min, U, u_max))

# Límites de estado (opcional, ej: velocidad máx)
# opti.subject_to(opti.bounded(-10, X[1, :], 10))

# --- Resolver ---
# --- Resolver ---
opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 2000})

try:
    sol = opti.solve()
    print(f"Costo final: {sol.value(obj):.2f}")

    # --- PLOTTING MEJORADO ---
    X_sol = sol.value(X)
    U_sol = sol.value(U)
    time = np.arange(N+1) * dt
    
    # 1. Trayectoria 3D
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Trayectoria del Drone
    ax1.plot(X_sol[0, :], X_sol[1, :], X_sol[2, :], 'b-', linewidth=2, label='Drone')
    
    # Referencia (Calculamos numéricamente para plotear)
    ref_vals = np.zeros((3, N+1))
    for k in range(N+1):
        t_val = k*dt
        ref_vals[0,k] = 5 * np.cos(t_val/2) - 5
        ref_vals[1,k] = 5 * np.sin(t_val/2)
        ref_vals[2,k] = t_val/2
    ax1.plot(ref_vals[0, :], ref_vals[1, :], ref_vals[2, :], 'r--', alpha=0.5, label='Ref')
    
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title('Trayectoria 3D')

    # 2. Controles
    ax2 = fig.add_subplot(122)
    ax2.step(time[:-1], U_sol[0, :], 'k', label='Thrust (N)')
    ax2.step(time[:-1], U_sol[1, :], 'r', label='Tx')
    ax2.step(time[:-1], U_sol[2, :], 'g', label='Ty')
    ax2.step(time[:-1], U_sol[3, :], 'b', label='Tz')
    ax2.grid(True)
    ax2.legend()
    ax2.set_title('Entradas de Control')
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("Error:", e)