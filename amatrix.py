import numpy as np

# Local beam stiffness matrix:
# -----------------------------------------------
def kbeam(ei,lx):
    k = np.zeros((4,4), dtype=float)
    k[0,0] = 12*ei/(lx**3)
    k[0,1] = 6*ei/(lx**2)
    k[0,2] = -k[0,0]
    k[0,3] = k[0,1]
    k[1,1] = 4*ei/lx
    k[1,2] = -k[0,1]
    k[1,3] = 2*ei/lx
    k[2,2] = k[0,0]
    k[2,3] = -k[0,1]
    k[3,3] = k[1,1]

    for i in range(1,4):
        for j in range(i):
            k[i,j] = k[j,i]
    return k

# Problem data
# -----------------------------------------------
eis = np.array([
    45000,
    45000,
    45000
], dtype=float)

lxs = np.array([
    6,
    2,
    2
], dtype=float)

elem_dof = np.array([
    [1,2,3,4],
    [3,4,5,6],
    [5,6,7,8]
], dtype=int)

nelem = len(eis)

# Global stiffness matrix
# -----------------------------------------------
kgg = np.zeros((8,8), dtype=float)
for elem in range(nelem):
    kl = kbeam(eis[elem], lxs[elem])
    for i in range(4):
        for j in range(4):
            ig = elem_dof[elem,i]
            jg = elem_dof[elem,j]
            kgg[ig-1,jg-1] += kl[i,j]

# Kre:
# -----------------------------------------------
tre = np.zeros((8,8), dtype=int)
tre[0,3] = 1
tre[1,4] = 1
tre[2,5] = 1
tre[3,7] = 1
tre[4,0] = 1
tre[5,1] = 1
tre[6,2] = 1
tre[7,6] = 1

kgg_re = np.matmul(tre, np.matmul(kgg, tre.T))

kgg_ff = kgg_re[0:4,0:4].copy()
kgg_fp= kgg_re[0:4,4:8].copy()
kgg_pf = kgg_re[4:8,0:4].copy()
kgg_pp = kgg_re[4:8,4:8].copy()

# Displacements:
# -----------------------------------------------
ugg_p = np.zeros(4, dtype=float)
fgg_f = np.array([80,-50,0,0], dtype=float)
b = fgg_f - np.matmul(kgg_fp, ugg_p)
ugg_f = np.linalg.solve(kgg_ff, b)
print()
print(f"ugg_f =\n{ugg_f}")

# Forces:
# -----------------------------------------------
fgg_p = np.matmul(kgg_pf, ugg_f) + np.matmul(kgg_pp, ugg_p)
print()
print(f"fgg_p =\n{fgg_p}")

# Reactions:
# -----------------------------------------------
fn_p = np.array([-60,-60,-60,0], dtype=float)
rgg = fgg_p - fn_p
print()
print(f"rgg =\n{rgg}")
