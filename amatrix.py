import sys
import numpy as np
from time import time
import pandas as pd
from tabulate import tabulate

start_time = time()

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

# Elem length:
# -----------------------------------------------
def elem_length(elem):
    ni = nodes[elem["ni"]]
    nf = nodes[elem["nf"]]
    xi = ni["coord"][0]
    yi = ni["coord"][1]
    xf = nf["coord"][0]
    yf = nf["coord"][1]
    return np.sqrt((xf-xi)**2 + (yf-yi)**2)

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

# Global stiffness matrix
# -----------------------------------------------
def global_stiffness_matrix():
    kff = np.zeros((ndof,ndof), dtype=float)
    kfp = np.zeros((ndof,ndpr), dtype=float)
    kpf = np.zeros((ndpr,ndof), dtype=float)
    kpp = np.zeros((ndpr,ndpr), dtype=float)
    for elem in elems.values():
        kl = kbeam(elem["ei"], elem_length(elem))
        for i in range(4):
            for j in range(4):
                if (elem["dof_type"][i] == "free") and (elem["dof_type"][j] == "free"):
                    ig = elem["dof"][i]
                    jg = elem["dof"][j]
                    kff[ig,jg] += kl[i,j]
                elif (elem["dof_type"][i] == "free") and (elem["dof_type"][j] == "fixed"):
                    ig = elem["dof"][i]
                    jg = elem["dof"][j]
                    kfp[ig,jg] += kl[i,j]
                elif (elem["dof_type"][i] == "fixed") and (elem["dof_type"][j] == "free"):
                    ig = elem["dof"][i]
                    jg = elem["dof"][j]
                    kpf[ig,jg] += kl[i,j]
                elif (elem["dof_type"][i] == "fixed") and (elem["dof_type"][j] == "fixed"):
                    ig = elem["dof"][i]
                    jg = elem["dof"][j]
                    kpp[ig,jg] += kl[i,j]
    return kff, kfp, kpf, kpp

kgg_ff, kgg_fp, kgg_pf, kgg_pp = global_stiffness_matrix()

# Nodal force vector:
# -----------------------------------------------
fgg_f = np.zeros(ndof, dtype=float)
fn_p = np.zeros(ndpr, dtype=float)

def build_force_vector(fgg_f, fn_p):
    for node in nodes.values():
        for i in range(2):
            ig = node["dof"][i]
            if node["dof_type"][i] == "free":
                fgg_f[ig] += node["nodal_forces"][i]
            elif node["dof_type"][i] == "fixed":
                fn_p[ig] += node["nodal_forces"][i]

build_force_vector(fgg_f, fn_p)

# Displacements:
# -----------------------------------------------
ugg_p = np.zeros(ndpr, dtype=float)
b = fgg_f - np.matmul(kgg_fp, ugg_p)
ugg_f = np.linalg.solve(kgg_ff, b)

def nodal_displacements(node):
    ul = np.zeros(2, dtype=float)
    for i in range(2):
        ig = node["dof"][i]
        if node["dof_type"][i] == "free":
            ul[i] = ugg_f[ig]
        elif node["dof_type"][i] == "fixed":
            ul[i] = ugg_p[ig]
    return ul

print("Nodal Displacements:")
lines = [(node_id, *nodal_displacements(node)) for node_id, node in nodes.items()]
df = pd.DataFrame(lines, columns=["Node", "u1", "u2"])
print(tabulate(df, headers="keys", tablefmt="fancy_grid", floatfmt=".3e", showindex=False))

# End program
# ===============================================
end_time = time()
print()
print(f"Elapsed time = {end_time - start_time:.3f} seconds")
print()
print("Analysis completed succefully!!!")
print()
sys.exit()
# ===============================================

# Reactions:
# -----------------------------------------------
fgg_p = np.matmul(kgg_pf, ugg_f) + np.matmul(kgg_pp, ugg_p)
print()
print(f"fgg_p =\n{fgg_p}")
fn_p = np.array([-60,-60,-60,0], dtype=float)
rgg = fgg_p - fn_p
print()
print(f"rgg =\n{rgg}")

# Member end Forces
# -----------------------------------------------
def local_displacements(elem):
    ul = np.zeros(4, dtype=float)
    # TODO
    return ul

def member_end_forces(elem):
    pass
