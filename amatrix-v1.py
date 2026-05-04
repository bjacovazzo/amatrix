import sys
import time
from enum import Enum
import json
import numpy as np
from tabulate import tabulate
import pandas as pd

# Class definition
# ===================================================================
class DofType(Enum):
    FREE = 1
    FIXED = 2
    PRESCRIBED = 3
    SPRING = 4

# -------------------------------------------------------------------
class Dof:
    def __init__(self, type:DofType, idg=None):
        self.type = type
        self.idg = idg

# -------------------------------------------------------------------
class Bc:
    def __init__(self, dof):
        self.dof = dof

# -------------------------------------------------------------------
class Node:
    def __init__(self, coord, bc=None, nodal_force=None):
        self.coord = coord
        self.bc = bc
        self.nodal_force = nodal_force
        self.dof = []

# .....................................
    def build_dof_matrix(self):
        if self.bc is None:
            for i in range(2):
                self.dof.append(Dof(DofType.FREE, model.ndof))
                model.ndof += 1
        else:
            bc = model.bcs[self.bc]
            for dof in bc.dof:
                if dof == "free":
                    self.dof.append(Dof(DofType.FREE, model.ndof))
                    model.ndof += 1
                elif dof == "fixed":
                    self.dof.append(Dof(DofType.FIXED, model.ndpr))
                    model.ndpr += 1

# .....................................
    def spread_nodal_forces(self, ff, fp):
        if self.nodal_force is not None:
            for idl, dof in enumerate(self.dof):
                nforce = model.nodal_forces[self.nodal_force]
                if dof.type == DofType.FREE:
                    ff[dof.idg] += nforce.forces[idl]
                elif dof.type == DofType.FIXED:
                    fp[dof.idg] += nforce.forces[idl]

# .....................................
    def displacements(self):
        ug = np.zeros(2, dtype=float)
        for idl, dof in enumerate(self.dof):
            if dof.type == DofType.FREE:
                ug[idl] = ugg_f[dof.idg]
            elif dof.type == DofType.FIXED:
                ug[idl] = ugg_p[dof.idg]
        return ug

# .....................................
    def reactions(self):
        r = np.zeros(2, dtype=float)
        for idl, dof in enumerate(self.dof):
            if dof.type == DofType.FREE:
                r[idl] = np.nan
            elif dof.type == DofType.FIXED:
                r[idl] = rgg[dof.idg]
        return r

# -------------------------------------------------------------------
class Material:
    def __init__(self, e):
        self.e = e

# -------------------------------------------------------------------
class Section:
    def __init__(self, iz):
        self.iz = iz

# -------------------------------------------------------------------
class Beam:
    def __init__(self, ni, nf, material, section, fef=np.zeros(4, dtype=float)):
        self.ni = ni
        self.nf = nf
        self.material = material
        self.section = section
        self.fef = np.array(fef)

# .....................................
    def dof_matrix(self):
        ni = model.nodes[self.ni]
        nf = model.nodes[self.nf]
        dof = ni.dof + nf.dof
        return dof

# .....................................
    def length(self):
        ni = model.nodes[self.ni]
        nf = model.nodes[self.nf]
        dx = nf.coord[0] - ni.coord[0]
        dy = nf.coord[1] - ni.coord[1]
        return np.sqrt(dx*dx + dy*dy)

# .....................................
    def kl(self):
        material = model.materials[self.material]
        section = model.sections[self.section]
        ei = material.e * section.iz
        lx = self.length()

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

# .....................................
    def ul(self):
        ni = model.nodes[self.ni]
        nf = model.nodes[self.nf]
        return np.append(ni.displacements(), nf.displacements())

# .....................................
    def end_forces(self):
        return np.matmul(self.kl(), self.ul()) + self.fef

# -------------------------------------------------------------------
class Nodal_Force:
    def __init__(self, forces):
        self.forces = forces

# -------------------------------------------------------------------
class Model:
    def __init__(self):
        self.nnode = 0
        self.nbeam = 0
        self.ndof = 0
        self.ndpr = 0

# .....................................
    def load_from_file(self, file):
        with open(file, 'r') as f:
            data = json.load(f)

        self.bcs = {id: Bc(**input_data) for id, input_data in data["bcs"].items()}
        self.nodes = {id: Node(**input_data) for id, input_data in data["nodes"].items()}
        self.materials = {id: Material(**input_data) for id, input_data in data["materials"].items()}
        self.sections = {id: Section(**input_data) for id, input_data in data["sections"].items()}
        self.beams = {id: Beam(**input_data) for id, input_data in data["beams"].items()}
        self.nodal_forces = {id: Nodal_Force(**input_data) for id, input_data in data["nodal_forces"].items()}

# .....................................
    def calculate_ndof(self):
        for node in self.nodes.values():
            node.build_dof_matrix()

# .....................................
    def build_force_vector(self):
        ff = np.zeros(self.ndof, dtype=float)
        fp = np.zeros(self.ndpr, dtype=float)

        for node in self.nodes.values():
            node.spread_nodal_forces(ff, fp)

        return ff, fp

# .....................................
    def build_stiffness_matrix(self):
        kff = np.zeros((self.ndof,self.ndof), dtype=float)
        kfp = np.zeros((self.ndof,self.ndpr), dtype=float)
        kpf = np.zeros((self.ndpr,self.ndof), dtype=float)
        kpp = np.zeros((self.ndpr,self.ndpr), dtype=float)

        for beam in self.beams.values():
            dof = beam.dof_matrix()
            kl = beam.kl()

            for i in range(4):
                for j in range(4):
                    ig = dof[i].idg
                    jg = dof[j].idg
                    if (dof[i].type == DofType.FREE  and dof[j].type == DofType.FREE):
                        kff[ig,jg] += kl[i,j]
                    elif (dof[i].type == DofType.FREE  and dof[j].type == DofType.FIXED):
                        kfp[ig,jg] += kl[i,j]
                    elif (dof[i].type == DofType.FIXED  and dof[j].type == DofType.FREE):
                        kpf[ig,jg] += kl[i,j]
                    elif (dof[i].type == DofType.FIXED  and dof[j].type == DofType.FIXED):
                        kpp[ig,jg] += kl[i,j]
        return kff, kfp, kpf, kpp


# Program start
# ===================================================================
if __name__ == "__main__":

    start_time = time.time()

# Reading model from file
# -------------------------------------------------------------------
    print()
    print("Program AMatrix")
    print("======= =======\n")

    model = Model()

    print("Reading model from file... ", end="")
    file = sys.argv[1]
    model.load_from_file(file)
    print("Ok")

# Printing model information
# -------------------------------------------------------------------
    print(f"File name: {file}")

    model.nnode = len(model.nodes)
    model.nbeam = len(model.beams)

    print(f"Number of nodes = {model.nnode}")
    print(f"Number of beam elements = {model.nbeam}\n")

# Degrees of freedom
# -------------------------------------------------------------------
    print("Calculating degrees of freedom... ", end="")
    model.calculate_ndof()
    print("Ok")
    print(f"Number of degrees of freedom = {model.ndof}")
    print(f"Number of prescribed displacements = {model.ndpr}\n")

# Nodal force vector
# -------------------------------------------------------------------
    print("Building force vector... ", end="")
    fgg_f, fn_p = model.build_force_vector()
    print("Ok\n")

# Stiffness matrix
# -------------------------------------------------------------------
    print("Building stiffness matrix... ", end="")
    kgg_ff, kgg_fp, kgg_pf, kgg_pp = model.build_stiffness_matrix()
    print("Ok\n")

# Displacements
# -------------------------------------------------------------------
    print("Solving system of equations... ", end="")
    ugg_f = np.linalg.solve(kgg_ff, fgg_f)
    print("Ok\n")
    ugg_p = np.zeros(model.ndpr, dtype=float)

    print("Nodal Displacements:")
    lines = [(node_id, *node.displacements()) for node_id, node in model.nodes.items()]
    df = pd.DataFrame(lines, columns=["Node","uY","rZ"])
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", floatfmt=".3e", showindex=False))

# Reactions
# -------------------------------------------------------------------
    fgg_p = np.matmul(kgg_pf, ugg_f) + np.matmul(kgg_pp, ugg_p)
    rgg = fgg_p - fn_p

    print()
    print("Reactions:")
    lines = [(node_id, *node.reactions()) for node_id, node in model.nodes.items()]
    df = pd.DataFrame(lines, columns=["Node","rY","mZ"])
    df_print = df.astype(object).where(pd.notnull(df), None)
    print(tabulate(df_print, headers="keys", tablefmt="fancy_grid", floatfmt=".3f", missingval="-", showindex=False))

# Member end Forces
# -------------------------------------------------------------------
    print()
    print("Member End Forces:")
    lines = [(beam_id, *beam.end_forces()) for beam_id, beam in model.beams.items()]
    df = pd.DataFrame(lines, columns=["Element","fy1","mz1","fy2","mz2"])
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", floatfmt=".3f", showindex=False))

# Final procedures
# -------------------------------------------------------------------
    print()
    print("Analysis completed successfully!!!\n")

    end_time = time.time()
    print(f"Total elapsed time: {end_time - start_time:.3f} seconds\n")
