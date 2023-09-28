import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import sys


def scal(T1, T2):
    return sum(T1[i] @ T2[i] for i in range(len(T1)))


# Pauli's matrices
Pauli = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]

# Ir-Complex spin operators
# Ha
I1 = [np.kron(a / 2, np.kron(np.eye(2), np.kron(np.eye(2), np.eye(2)))) for a in Pauli]
# Hb
I2 = [np.kron(np.eye(2), np.kron(a / 2, np.kron(np.eye(2), np.eye(2)))) for a in Pauli]
# Sa
I3 = [np.kron(np.eye(2), np.kron(np.eye(2), np.kron(a / 2, np.eye(2)))) for a in Pauli]
# Sb
I4 = [np.kron(np.eye(2), np.kron(np.eye(2), np.kron(np.eye(2), a / 2))) for a in Pauli]

I = [I1, I2, I3, I4]

# Free spin operators
# Sa
S1 = [np.kron(a / 2, np.eye(2)) for a in Pauli]
# Sb
S2 = [np.kron(np.eye(2), a / 2) for a in Pauli]
S = [S1, S2]

J_HH = -7
J_NN = -0.4
J_NH = -20


# delta_w = - abs(res_freq) + abs(rf_freq)
# HAMILTONIAN
def H_complex_super(w1_H, w1_N, delta_w):
    h_zeeman = -delta_w * (I1[2] + I2[2]) - w1_N * (I1[0] + I2[0]) - w1_H * (I3[0] + I4[0])
    h_coupling = 2 * np.pi * J_HH * scal(I3, I4) + 2 * np.pi * J_NN * scal(I1, I2) + 2 * np.pi * J_NH * \
                 (I1[2] @ I3[2] + I2[2] @ I4[2])
    h = h_zeeman + h_coupling
    return np.kron(h, np.eye(16)) - np.kron(np.eye(16), h)


def H_free_super(delta_w):
    freq_diff = (-300 + 255.15) * 40.544834 * 2 * np.pi + delta_w
    h = -freq_diff * (S1[2] + S2[2])
    return np.kron(h, np.eye(4)) - np.kron(np.eye(4), h)


# Complex (T1 in s 1,2 Substrate; 3,4 Protons)
Tc = [3, 3, 1, 1]
# Tc = [3, 3, 5, 5]
# Free-substrate (T1 in s 1,2 Free-substrate)
Tf = [30, 30]


# n is number of spins
def R_complex_super(Tc):
    J = np.zeros((16, 16, 16, 16))
    r = np.zeros((16, 16, 16, 16))
    Sum = np.zeros((16, 16))
    B = np.eye(16)

    for i in range(16):
        for j in range(16):
            for m in range(16):
                for n in range(16):
                    J[i, j, m, n] = sum(1 / Tc[l] * (
                            (B[:, i].conj().T @ I[l][0] @ B[:, j]) * (B[:, n].conj().T @ I[l][0] @ B[:, m]) + (
                            B[:, i].conj().T @ I[l][1] @ B[:, j]) * (B[:, n].conj().T @ I[l][1] @ B[:, m]) + (
                                    B[:, i].conj().T @ I[l][2] @ B[:, j]) * (
                                    B[:, n].conj().T @ I[l][2] @ B[:, m])) for l in range(4))

    for i in range(16):
        for j in range(16):
            Sum[i, j] = sum(J[k, i, k, j] for k in range(16))

    for i in range(16):
        for j in range(16):
            for m in range(16):
                for n in range(16):
                    r[i, j, m, n] = 1 / 2 * (
                            2 * J[i, m, j, n] - np.eye(16)[j, n] * Sum[m, i] - np.eye(16)[i, m] * Sum[j, n])

    return np.reshape(r, (256, 256))


def R_free_super(Tf):  # Rel-operator for Free-substrate
    J = np.zeros((4, 4, 4, 4))
    r = np.zeros((4, 4, 4, 4))
    Sum = np.zeros((4, 4))
    B = np.eye(4)
    for i in range(4):
        for j in range(4):
            for m in range(4):
                for n in range(4):
                    J[i, j, m, n] = sum(1 / Tf[l] * (
                            (B[:, i].conj().T @ S[l][0] @ B[:, j]) * (B[:, n].conj().T @ S[l][0] @ B[:, m]) + (
                            B[:, i].conj().T @ S[l][1] @ B[:, j]) * (B[:, n].conj().T @ S[l][1] @ B[:, m]) + (
                                    B[:, i].conj().T @ S[l][2] @ B[:, j])
                            * (B[:, n].conj().T @ S[l][2] @ B[:, m])) for l in range(2))

    for i in range(4):
        for j in range(4):
            Sum[i, j] = sum(J[k, i, k, j] for k in range(4))

    for i in range(4):
        for j in range(4):
            for m in range(4):
                for n in range(4):
                    r[i, j, m, n] = 1 / 2 * (2 * J[i, m, j, n] - np.eye(4)[j, n] * Sum[m, i] -
                                             np.eye(4)[i, m] * Sum[j, n])

    return np.reshape(r, (16, 16))


Rc = R_complex_super(Tc)
Rf = R_free_super(Tf)

P_H = 1 * (np.eye(4) / 4 - S1[2] @ S2[2])
# NEGLECTING THERMAL POLARIZATION OF 15N
P_S = np.eye(4) / 4

# KINETICS
# Catalyst-to-Substrate ratio = [C] / [S]
L = 1 / 35
f_f = 1 / (1 + L)
f_c = L / (1 + L)

# NON-POLARIZED INITIAL STATE OF THE COMPLEX AND FREE-SUBSTRATE
P_0 = np.append(np.reshape(f_f * np.eye(4) / np.trace(np.eye(4)), (16, 1)),
                np.reshape(f_c * np.eye(16) / np.trace(np.eye(16)), (256, 1)))

# SABRE CHEMICAL EXCHANGE OPERATORS and KINETICS
# Substrate disassociation rate-constant
kd = 80
# Substrate association rate
Wa = kd * L

# EXCHANGE OPERATORS
# Kron operator matrix
S_kron = np.zeros((256, 16))
# Partial trace operator matrix
S_trace = np.zeros((16, 256))

for i in range(4):
    for j in range(4):
        for m in range(4):
            for n in range(4):
                Q = (i * 4 + m) * 16 + j * 4 + n
                W = i * 4 + j
                S_kron[Q, W] = P_H[m, n]
                S_trace[W, Q] = np.eye(4)[m, n]


def A(w1_H, w1_N, delta_w):
    H = H_complex_super(w1_H, w1_N, delta_w)
    full_matrix = np.concatenate((np.concatenate((-1j * H_free_super(delta_w) + Rf - Wa * np.eye(16), Wa * S_kron)),
                                  np.concatenate((kd * S_trace, -1j * H + Rc - kd * np.eye(256)))), axis=1)
    return full_matrix


N = 1000
w1_H = 2 * np.pi * 200
w1_N_zero = w1_H + 2 * np.pi * 10
delta_w = -2 * np.pi * 198
# w1_N = np.linspace(w1_N_zero, 0, N)
P = P_0
'''
Mz_f = np.zeros(N)
Mz_b = np.zeros(N)
t_ev_FD = 2
with open('FD_nu1h_200Hz_dnu1n_198.txt', 'w') as f:
    sys.stdout = f
    for j in range(N):
        P = la.expm(A(w1_H, w1_N[j], delta_w) * t_ev_FD) @ P_0
        P_b = np.reshape([P[k] for k in range(16, 272)], (16, 16))
        P_f = np.reshape([P[i] for i in range(16)], (4, 4))
        Mz_f[j] = np.trace((S1[2] + S2[2]) @ P_f) / np.trace(P_f)
        Mz_b[j] = np.trace((I1[2] + I2[2]) @ P_b) / np.trace(P_b)
        print(w1_N[j] / 2 / np.pi, np.real(Mz_f[j]), np.real(Mz_b[j]))
plt.plot(w1_N / 2 / np.pi, Mz_b)
plt.plot(w1_N / 2 / np.pi, Mz_f)
plt.show()
'''

N_tau = 30
tau = np.linspace(0, 2, N_tau)
Mz_f = np.zeros(N_tau)
Mz_b = np.zeros(N_tau)
opt_profile = open("opt_profile.txt", "r")
w1_N = [float(x) * 2 * np.pi for x in opt_profile.read().split()]
print(w1_N)
with open('opt_adiabatic.txt.txt', 'w') as f:
    sys.stdout = f
    for i in range(N_tau):
            dt = tau[i] / N_tau
            for j in range(N):
                P = la.expm(A(w1_H, w1_N[j], delta_w) * dt) @ P
            P_f = np.reshape([P[k] for k in range(16)], (4, 4))
            P_b = np.reshape([P[k] for k in range(16, 272)], (16, 16))
            Mz_f[i] = np.trace((S1[2] + S2[2]) @ P_f) / np.trace(P_f)
            Mz_b[i] = np.trace((I1[2] + I2[2]) @ P_b) / np.trace(P_b)
            print(tau[i], np.real(Mz_f[i]), np.real(Mz_b[i]))
            P = P_0

'''
N = 1000
w1_H = 2 * np.pi * 50
w1_N_zero = w1_H + 2 * np.pi * 10
# w1_N = np.linspace(w1_N_zero, 0, N)
# delta_w = (-255.15 + 254.4227659478198) * 40.544834 * 2 * np.pi
delta_w = (-255.15 + 254.39110601194642) * 40.544834 * 2 * np.pi
P = P_0
N_tau = 40
tau = np.linspace(0, 2, N_tau)
Mz = np.zeros(N_tau)
opt_profile = open("profile_20_calculated.txt", "r")
w1_N = [float(x) * 2 * np.pi for x in opt_profile.read().split()]
print(w1_N)


with open('tau_20_kd_60_opt_calc.txt', 'w') as f:
    sys.stdout = f
    for i in range(N_tau):
        dt = tau[i] / N_tau
        for j in range(N):
            P = la.expm(A(w1_H, w1_N[j], delta_w) * dt) @ P
        P_f = np.reshape([P[k] for k in range(16)], (4, 4))
        Mz[i] = np.trace((S1[2] + S2[2]) @ P_f) / np.trace(P_f)
        print(tau[i], np.real(Mz[i]))
        P = P_0



N = 1000
w1_H = 2 * np.pi * 50
w1_N_zero = w1_H + 2 * np.pi * 10
# w1_N = np.linspace(w1_N_zero, 0, N)
# delta_w = (-255.15 + 254.4227659478198) * 40.544834 * 2 * np.pi
delta_w = (-255.15 + 253.7586943552351) * 40.544834 * 2 * np.pi
# w1_N = np.linspace(w1_N_zero, 0, N)
# delta_w = (-255.15 + 254.4227659478198) * 40.544834 * 2 * np.pi
w1_N = np.linspace(w1_N_zero, 0, N)
t_ev = 1
Mz = np.zeros(N)

with open('Mz_FD_50_kd_80.txt', 'w') as f:
    sys.stdout = f
    for j in range(N):
        P = la.expm(A(w1_H, w1_N[j], delta_w) * t_ev) @ P_0
        # P_f = np.reshape([P[k] for k in range(16, 272)], (16, 16))
        P_f = np.reshape([P[i] for i in range(16)], (4, 4))
        Mz[j] = np.trace((S1[2] + S2[2]) @ P_f) / np.trace(P_f)
        # Mz[i] = np.trace((I1[2] + I2[2]) @ P_f) / np.trace(P_f)
        print(w1_N[j] / (2 * np.pi), np.real(Mz[j]))

plt.plot(w1_N / (2 * np.pi), Mz)
plt.show()
'''
