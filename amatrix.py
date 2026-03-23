import numpy as np

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

kgg = np.zeros((8,8), dtype=float)
for elem in range(nelem):
    kl = kbeam(eis[elem], lxs[elem])
    for i in range(4):
        for j in range(4):
            ig = elem_dof[elem,i]
            jg = elem_dof[elem,j]
            kgg[ig-1,jg-1] += kl[i,j]
print(kgg)
