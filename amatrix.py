import sys
import numpy as np

print()
print("Program AMatrix")
print("======= =======")

# Model data:
# -----------------------------------------------
nodes = {
    "A":{"coord":[0.,0.], "bc": ["fixed","fixed"],
        "nodal_forces": [-60.,-60.]},
    "B":{"coord":[6.,0.], "bc": ["fixed","free"],
        "nodal_forces": [-60.,80.]},
    "C":{"coord":[8.,0.], "bc": ["free","free"],
        "nodal_forces": [-50.,0.]},
    "D":{"coord":[10.,0.], "bc": ["fixed","free"],
        "nodal_forces": [0.,0.]}
}

elems = {
    "b1":{"ei": 45000., "ni": "A", "nf": "B",
        "fef": [60.,60.,60.,-60.]},
    "b2":{"ei": 45000., "ni": "B", "nf": "C",
        "fef": [0.,0.,0.,0.]},
    "b3":{"ei": 45000., "ni": "C", "nf": "D",
        "fef": [0.,0.,0.,0.]}
}

nnode = len(nodes)
nelem = len(elems)

print()
print(f"Number of nodes = {nnode}")
print(f"Number of elements = {nelem}")

# Degrees of freedom:
# -----------------------------------------------
ndof = 0
ndpr = 0

def build_node_dof(node):
    global ndof, ndpr
    dof_matrix = []
    for dof in node["bc"]:
        if dof == "free":
            dof_matrix.append(ndof)
            ndof += 1
        elif dof == "fixed":
            dof_matrix.append(ndpr)
            ndpr += 1
    node["dof"] = dof_matrix
    node["dof_type"] = node["bc"]
        
for node in nodes.values():
    build_node_dof(node)

print()
print(f"Number of degrees of freedom = {ndof}")
print(f"Number of prescribed displacements = {ndpr}\n")

# Element dof:
# -----------------------------------------------
def build_elem_dof(elem):
    ni = nodes[elem["ni"]]
    nf = nodes[elem["nf"]]
    elem["dof"] = ni["dof"] + nf["dof"]
    elem["dof_type"] = ni["dof_type"] + nf["dof_type"]

for elem in elems.values():
    build_elem_dof(elem)

# End program
# ===============================================
sys.exit()
# ===============================================

# Stiffness matrix for beam element:
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
# eis = np.array([
#     45000,
#     45000,
#     45000
# ], dtype=float)

# lxs = np.array([
#     6,
#     2,
#     2
# ], dtype=float)

# elem_dof = np.array([
#     [1,2,3,4],
#     [3,4,5,6],
#     [5,6,7,8]
# ], dtype=int)


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

# print()
# print(f"kgg_ff =\n{kgg_ff}")
# print()
# print(f"kgg_fp =\n{kgg_fp}")
# print()
# print(f"kgg_pf =\n{kgg_pf}")
# print()
# print(f"kgg_pp =\n{kgg_pp}")

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
