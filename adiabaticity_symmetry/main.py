import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import math
import pandas as pd


def scal(T1, T2):
    return sum(T1[i] @ T2[i] for i in range(len(T1)))


# Pauli's matrices
S = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]

# Ir-Complex spin operators
# Ha
I1 = [np.kron(a / 2, np.kron(np.eye(2), np.kron(np.eye(2), np.eye(2)))) for a in S]

# Hb
I2 = [np.kron(np.eye(2), np.kron(a / 2, np.kron(np.eye(2), np.eye(2)))) for a in S]

# Sa
I3 = [np.kron(np.eye(2), np.kron(np.eye(2), np.kron(a / 2, np.eye(2)))) for a in S]

# Sb
I4 = [np.kron(np.eye(2), np.kron(np.eye(2), np.kron(np.eye(2), a / 2))) for a in S]

# NEW BASIS FOR HAMILTONIAN
ST = np.array([[1, 0, 0, 0],
               [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
               [0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
               [0, 0, 0, 1]])

# FIRST STEP IS TO GO TO SINGLET-TRIPLET BASIS
U1 = np.kron(ST, ST)

U2 = np.zeros((16, 16))

# 10x10 block
U2[0][0] = 1
U2[1][1] = 1
U2[3][2] = 1
U2[4][3] = 1
U2[5][4] = 1
U2[7][5] = 1
U2[10][6] = 1
U2[12][7] = 1
U2[13][8] = 1
U2[15][9] = 1

# 6x6 block
U2[2][10] = 1
U2[6][11] = 1
U2[8][12] = 1
U2[9][13] = 1
U2[11][14] = 1
U2[14][15] = 1

T = U1 @ U2
iT = la.inv(T)

L1 = [iT @ I1[i] @ T for i in range(len(I1))]
L2 = [iT @ I2[i] @ T for i in range(len(I2))]
L3 = [iT @ I3[i] @ T for i in range(len(I3))]
L4 = [iT @ I4[i] @ T for i in range(len(I4))]

J_HH = -7
J_NN = -0.4
J_NH = -20

'''
# 200 Hz edge
w1_H = 2 * math.pi * 200
delta_w = 2 * math.pi * 205
w1_N_zero = w1_H + 2 * math.pi * 10
# c_zero_six = 0.190671
# c_zero_ten = 0.15861

# 20 Hz edge
w1_H = 2 * math.pi * 20
delta_w = 2 * math.pi * 22.7
w1_N_zero = w1_H + 2 * math.pi * 10
c_zero_six = 0.04544
c_zero_ten = 0.6799
'''

'''
# 200 Hz in
w1_H = 2 * math.pi * 200
delta_w = 2 * math.pi * 2.45245245245244
w1_N_zero = w1_H + 2 * math.pi * 10
c_zero_six = 0.1207
c_zero_ten absent
'''

'''
# 200 Hz exp
w1_H = 2 * math.pi * 200
delta_w = (-255.15 + 253.06304153285) * 40.544834 * 2 * np.pi
w1_N_zero = w1_H + 2 * math.pi * 10
'''

'''
# 20 Hz expt
w1_H = 2 * math.pi * 20
delta_w = (-255.15 + 254.7389322313764) * 40.544834 * 2 * np.pi
w1_N_zero = w1_H + 2 * math.pi * 10
'''
# 200 Hz exp EDGE
w1_H = 2 * math.pi * 200
delta_w = (-255.15 + 250.02746558063828) * 40.544834 * 2 * np.pi
w1_N_zero = w1_H + 2 * math.pi * 10


# delta_w = w_15N - w_RF
def H(w1_H, w1_N, delta_w):
    h_zeeman = -delta_w * (L3[2] + L4[2]) - w1_N * (L3[0] + L4[0]) - w1_H * (L1[0] + L2[0])
    h_coupling = 2 * math.pi * J_HH * scal(L1, L2) + 2 * math.pi * J_NN * scal(L3, L4) + 2 * math.pi * J_NH * \
                 (L1[2] @ L3[2] + L2[2] @ L4[2])
    return h_zeeman + h_coupling


def equation_left_part(w1_H, w1_N, delta_w, first_border, second_border):
    S3 = [L3[i][first_border:second_border, first_border:second_border] for i in range(len(L3))]
    S4 = [L4[i][first_border:second_border, first_border:second_border] for i in range(len(L4))]
    # SOLVE EQUATION SEPARATELY FOR EACH H-BLOCK
    H1 = H(w1_H, w1_N, delta_w)[first_border:second_border, first_border:second_border]
    E, V = la.eig(H1)
    g = 0
    for i in range(second_border - first_border):
        for j in range(second_border - first_border):
            if i != j:
                w = E[i] - E[j]
                Vij = V[:, i].conj().T @ (S3[0] + S4[0]) @ V[:, j] / (V[:, i].conj().T @ V[:, i]) * (
                            V[:, j].conj().T @ V[:, j])
                g += Vij * Vij.conj() / (w ** 4)
    return 1 / np.sqrt(g.real)


# SOLUTION OF THE EQUATION
tsw = 1
N = 1000
t = np.linspace(0, tsw, N)
dt = tsw / N  # integration step
c_zero_six = 0.05
c_zero_ten = 0
# solution vector first for 6x6, second for 10x10


def get_solution_vector(c_zero, first_border, second_border):
    y = np.zeros(N)
    y[0] = w1_N_zero
    for i in range(N - 1):
        k1 = -equation_left_part(w1_H, y[i], delta_w, first_border, second_border) * c_zero
        k2 = -equation_left_part(w1_H, y[i] + dt / 2 * k1, delta_w, first_border, second_border) * c_zero
        k3 = -equation_left_part(w1_H, y[i] + dt / 2 * k2, delta_w, first_border, second_border) * c_zero
        k4 = -equation_left_part(w1_H, y[i] + dt * k3, delta_w, first_border, second_border) * c_zero
        y[i + 1] = y[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.where(y < 0, 0, y)


y_six = get_solution_vector(c_zero_six, 10, 16)
print(y_six)
# y_ten = get_solution_vector(c_zero_ten, 0, 10)
# print(y_ten)
lin = np.linspace(w1_N_zero, 0, N)
plt.plot(t, y_six, color='blue')
# plt.plot(t, y_ten, color='orange')
plt.plot(t, lin, color='black')
plt.show()

'''
# MAGNETIZATION CALCULATION

Ntau = 100
tau = np.linspace(0, 20, Ntau)


def get_magnetization(profile):
    K1 = [np.kron(a / 2, np.eye(2)) for a in S]
    K2 = [np.kron(np.eye(2), a / 2) for a in S]

    # SINGLET P_H = np.eye(4) / 4 - scal(K1, K2)
    # ANTIPHASE

    P_H = np.eye(4) / 4 - scal(K1, K2)

    # NEGLECTING THERMAL POLARIZATION OF 15N
    P_S = np.eye(4) / 4

    P0_zeeman = np.kron(P_H, P_S)
    P0 = iT @ P0_zeeman @ T
    P = P0
    Mz = np.zeros(Ntau)
    for i in range(Ntau):
        dtau = tau[i] / N
        for j in range(N):
            P = la.expm(-1j * H(w1_H, profile[j], delta_w) * dtau) @ P @ la.expm(1j * H(w1_H, profile[j], delta_w) * dtau)
        Mz[i] = np.real(np.trace((L3[2] + L4[2]) @ P)) / np.real((np.trace(P)))
        P = P0
    return Mz


Mz_six = get_magnetization(y_six)
Mz_ten = get_magnetization(y_ten)
Mz_lin = get_magnetization(lin)
'''
'''
phase = np.zeros(N).astype(int)
dfr = pd.DataFrame([(f, c) for (f, c) in zip(y_six / w1_N_zero * 100, phase)])
dfr.to_csv('y_six_200Hz_exp.csv', index = False, sep=' ')
dfr1 = pd.DataFrame([(f, c) for (f, c) in zip(y_ten / w1_N_zero * 100, phase)])
dfr1.to_csv('y_ten_200Hz_exp.csv', index = False, sep=' ')
'''

'''
dfr2 = pd.DataFrame([(f, c) for (f, c) in zip(Mz_six, tau)])
dfr2.to_csv('Mz_six_20Hz_S.csv', index = False)
dfr3 = pd.DataFrame([(f, c) for (f, c) in zip(Mz_ten, tau)])
dfr3.to_csv('Mz_ten_20Hz_S.csv', index = False)
dfr4 = pd.DataFrame([(f, c) for (f, c) in zip(Mz_lin, tau)])
dfr4.to_csv('Mz_lin_20Hz_S.csv', index = False)

plt.plot(tau, Mz_lin, color='black')
# plt.plot(tau, Mz_ten, color='red')
# plt.plot(tau, Mz_six, color='blue')
plt.show()
'''


