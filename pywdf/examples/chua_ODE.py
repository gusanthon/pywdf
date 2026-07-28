import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from pathlib import Path
import sys

script_path = Path(__file__).resolve()
src_dir = script_path.parent.parent

sys.path.append(str(src_dir))

plt_dir = src_dir.parent / "tests" / "plots"
plt_dir.mkdir(exist_ok=True, parents=True)


# Parameters
alpha = 15.6                        # C2 / C1                      
beta = 28                           # C2 * R2 / L
gamma = 0.5                         # C2R RL /L
m0 = -1.143
m1 = -0.714

# Chua's nonlinearity
def f(x):
    return m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))

# Chua's system (ODE)
def chua(t, state):
    x, y, z = state
    dx = alpha * (y - x - f(x))     # x – Voltage across C1
    dy = x - y + z                  # y – Voltage across C2   
    dz = -beta * y                  # z – Current through L
    return [dx, dy, dz]

# Simulation
t_span = (0, 150)
t_eval = np.linspace(*t_span, 10000)
initial_state = [0.1, 0.0, 0.0]

# method='RK45' (default) explicit Runge-Kutta method of order 5(4)
sol = solve_ivp(chua, t_span, initial_state, t_eval=t_eval)

t = sol.t
x, y, z = sol.y

# Plot time series of x, y, z
plt.figure(figsize=(6, 6))

plt.subplot(3, 1, 1)
plt.plot(t, x, color='blue')
plt.title('x(t) - Voltage across C1')
plt.xlabel('Time')
plt.ylabel('x')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, y, color='green')
plt.title('y(t) - Voltage across C2')
plt.xlabel('Time')
plt.ylabel('y')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, z, color='red')
plt.title('z(t) - Current through L')
plt.xlabel('Time')
plt.ylabel('z')
plt.grid(True)

plt.tight_layout()

out_path = plt_dir / f"{script_path.stem}_C1_C2_L.png"
plt.savefig(out_path.with_suffix('.png'))

plt.show()

# 3D Phase Portrait
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_title("3D Phase Portrait of Chua's Circuit")
ax.set_xlabel("x (V_C1)")
ax.set_ylabel("y (V_C2)")
ax.set_zlabel("z (I_L)")
plt.tight_layout()
out_path = plt_dir / f"{script_path.stem}_attractor.png"
plt.savefig(out_path.with_suffix('.png'))
plt.show()

# 2D Projections
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(x, y, color='purple')
plt.title('Projection: x vs y')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, z, color='orange')
plt.title('Projection: x vs z')
plt.xlabel('x')
plt.ylabel('z')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(y, z, color='teal')
plt.title('Projection: y vs z')
plt.xlabel('y')
plt.ylabel('z')
plt.grid(True)

plt.tight_layout()

# plot

out_path = plt_dir / f"{script_path.stem}_2D_projections.png"
plt.savefig(out_path.with_suffix('.png'))

plt.show()
