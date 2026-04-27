import sys
from time import time
import json
import numpy as np
import pandas as pd
from tabulate import tabulate

start_time = time()

print()
print("Program AMatrix")
print("======= =======")

# Model data:
# -----------------------------------------------
def load_file(file):
    global nodes, elems
    with open(file, 'r') as f:
        data = json.load(f)
        nodes = data["nodes"]
        elems = data["elems"]

load_file(sys.argv[1])

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
    dx = nf["coord"][0] - ni["coord"][0]
    dy = nf["coord"][1] - ni["coord"][1]
    return np.sqrt(dx*dx + dy*dy)

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
                ig = elem["dof"][i]
                jg = elem["dof"][j]
                if (elem["dof_type"][i] == "free") and (elem["dof_type"][j] == "free"):
                    kff[ig,jg] += kl[i,j]
                elif (elem["dof_type"][i] == "free") and (elem["dof_type"][j] == "fixed"):
                    kfp[ig,jg] += kl[i,j]
                elif (elem["dof_type"][i] == "fixed") and (elem["dof_type"][j] == "free"):
                    kpf[ig,jg] += kl[i,j]
                elif (elem["dof_type"][i] == "fixed") and (elem["dof_type"][j] == "fixed"):
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
    ug = np.zeros(2, dtype=float)
    for i in range(2):
        ig = node["dof"][i]
        if node["dof_type"][i] == "free":
            ug[i] = ugg_f[ig]
        elif node["dof_type"][i] == "fixed":
            ug[i] = ugg_p[ig]
    return ug

print("Nodal Displacements:")
lines = [(node_id, *nodal_displacements(node)) for node_id, node in nodes.items()]
df = pd.DataFrame(lines, columns=["Node", "uY", "rZ"])
print(tabulate(df, headers="keys", tablefmt="fancy_grid", floatfmt=".3e", showindex=False))

# Reactions:
# -----------------------------------------------
fgg_p = np.matmul(kgg_pf, ugg_f) + np.matmul(kgg_pp, ugg_p)
rgg = fgg_p - fn_p

def nodal_reactions(node):
    r = np.zeros(2, dtype=float)
    for idl, dof in enumerate(node["dof"]):
        if node["dof_type"][idl] == "free":
            r[idl] = np.nan
        elif node["dof_type"][idl] == "fixed":
            r[idl] = rgg[dof]
    return r

print()
print("Reactions:")

lines = [(node_id, *nodal_reactions(node)) for node_id, node in nodes.items()]
df = pd.DataFrame(lines, columns=["Node","rY","mZ"])
df_print = df.astype(object).where(pd.notnull(df), None)
print(tabulate(df_print, headers="keys", tablefmt="fancy_grid", floatfmt=".3f", missingval="-", showindex=False))

# Member end Forces
# -----------------------------------------------
def local_displacements(elem):
    ni = nodes[elem["ni"]]
    nf = nodes[elem["nf"]]
    return np.append(nodal_displacements(ni), nodal_displacements(nf))

def member_end_forces(elem):
    ei = elem["ei"]
    kl = kbeam(ei, elem_length(elem))
    ul = local_displacements(elem)
    return np.matmul(kl, ul) + elem["fef"]

print()
print("Member End Forces:")
lines = [(elem_id, *member_end_forces(elem)) for elem_id, elem in elems.items()]
df = pd.DataFrame(lines, columns=["Element","fy1","mz1","fy2","mz2"])
print(tabulate(df, headers="keys", tablefmt="fancy_grid", floatfmt=".3f", showindex=False))

# End program
# -----------------------------------------------
end_time = time()
print()
print(f"Elapsed time = {end_time - start_time:.3f} seconds")
print()
print("Analysis completed succefully!!!\n")
