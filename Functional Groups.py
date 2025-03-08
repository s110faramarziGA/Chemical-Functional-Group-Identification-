# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:32:58 2022
The codes takes the pdb file of a molecule and creates a 3d fingerprint. The fingerprint contains functiona groups, molecular scaffolds, and physical properties such as mass, polar surface area, etc
v3: shifting from a single pdb file to a file containing all pdbs in the training set. 
@author: sadegh-pc
"""

import csv
import logging 
import os
import traceback 
import pandas
from tqdm import tqdm
from biopandas.pdb import PandasPdb
from Bio.PDB import *
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem import Descriptors
import rdkit
import pandas as pd
import os
import sympy as smp
import sys







#%%

MM_of_Elements = {'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182, 'B': 10.811, 'C': 12.0107, 'N': 14.0067,
                  'O': 15.9994, 'F': 18.9984032, 'Ne': 20.1797, 'Na': 22.98976928, 'Mg': 24.305, 'Al': 26.9815386,
                  'Si': 28.0855, 'P': 30.973762, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.0983, 'Ca': 40.078,
                  'Sc': 44.955912, 'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961, 'Mn': 54.938045,
                  'Fe': 55.845, 'Co': 58.933195, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.409, 'Ga': 69.723, 'Ge': 72.64,
                  'As': 74.9216, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90585,
                  'Zr': 91.224, 'Nb': 92.90638, 'Mo': 95.94, 'Tc': 98.9063, 'Ru': 101.07, 'Rh': 102.9055, 'Pd': 106.42,
                  'Ag': 107.8682, 'Cd': 112.411, 'In': 114.818, 'Sn': 118.71, 'Sb': 121.760, 'Te': 127.6,
                  'I': 126.90447, 'Xe': 131.293, 'Cs': 132.9054519, 'Ba': 137.327, 'La': 138.90547, 'Ce': 140.116,
                  'Pr': 140.90465, 'Nd': 144.242, 'Pm': 146.9151, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25,
                  'Tb': 158.92535, 'Dy': 162.5, 'Ho': 164.93032, 'Er': 167.259, 'Tm': 168.93421, 'Yb': 173.04,
                  'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.9479, 'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217,
                  'Pt': 195.084, 'Au': 196.966569, 'Hg': 200.59, 'Tl': 204.3833, 'Pb': 207.2, 'Bi': 208.9804,
                  'Po': 208.9824, 'At': 209.9871, 'Rn': 222.0176, 'Fr': 223.0197, 'Ra': 226.0254, 'Ac': 227.0278,
                  'Th': 232.03806, 'Pa': 231.03588, 'U': 238.02891, 'Np': 237.0482, 'Pu': 244.0642, 'Am': 243.0614,
                  'Cm': 247.0703, 'Bk': 247.0703, 'Cf': 251.0796, 'Es': 252.0829, 'Fm': 257.0951, 'Md': 258.0951,
                  'No': 259.1009, 'Lr': 262, 'Rf': 267, 'Db': 268, 'Sg': 271, 'Bh': 270, 'Hs': 269, 'Mt': 278,
                  'Ds': 281, 'Rg': 281, 'Cn': 285, 'Nh': 284, 'Fl': 289, 'Mc': 289, 'Lv': 292, 'Ts': 294, 'Og': 294,
                  '': 0}

# a function to calculate molecular mass from the formula
def molecular_mass(compound: str, decimal_places=None) -> float:
    is_polyatomic = end = multiply = False
    polyatomic_mass, m_m, multiplier = 0, 0, 1
    element = ''

    for e in compound:
        if is_polyatomic:
            if end:
                is_polyatomic = False
                m_m += int(e) * polyatomic_mass if e.isdigit() else polyatomic_mass + MM_of_Elements[e]
            elif e.isdigit():
                multiplier = int(str(multiplier) + e) if multiply else int(e)
                multiply = True
            elif e.islower():
                element += e
            elif e.isupper():
                polyatomic_mass += multiplier * MM_of_Elements[element]
                element, multiplier, multiply = e, 1, False
            elif e == ')':
                polyatomic_mass += multiplier * MM_of_Elements[element]
                element, multiplier = '', 1
                end, multiply = True, False
        elif e == '(':
            m_m += multiplier * MM_of_Elements[element]
            element, multiplier = '', 1
            is_polyatomic, multiply = True, False
        elif e.isdigit():
            multiplier = int(str(multiplier) + e) if multiply else int(e)
            multiply = True
        elif e.islower():
            element += e
        elif e.isupper():
            m_m += multiplier * MM_of_Elements[element]
            element, multiplier, multiply = e, 1, False
    m_m += multiplier * MM_of_Elements[element]
    if decimal_places is not None:
        return round(m_m, decimal_places)
    return m_m
#%%  create a pandas df for the molecular fingerprint for all the compounds. fpArray will later be used as the input for the machine learning algorithm 
fpArray=pd.DataFrame({})

#%%
# xx=molecular_mass('OH2')
#%% use molecular 3d structure to generate a fingerprint. The fingerprint will be appended to an array of fingerprints to make an input for the NN
def molFp(mNum):
    
    #%%  taken from http://rasbt.github.io/biopandas/tutorials/Working_with_PDB_Structures_in_DataFrames/
#fetch pdb. this command is to fetch PDB online but it will give error if not used for loading files from the local descktop 
    ppdb = PandasPdb()#.fetch_pdb('3eiy')
    #load file from the local desktop
    #temporary test files
    # fpath='D:\Codes\\Structures\\pdb\\temp\\'
    
    fpath='D:Structures\pdb\\'
    
    
    
    # filename=('E:\Codes\Structures\pdb\\'+str(xx)+''.pdb')
    filename=fpath+str(mNum)+'.pdb'
    p_1=ppdb.read_pdb(filename)
    # print('PDB Code: %s' % ppdb.code)
    # print('PDB Header Line: %s' % ppdb.header)
    # print('\nRaw PDB file contents:\n\n%s\n...' % ppdb.pdb_text[:1000])
    #%% ATOM or HETATM names should correspond to the first column of the pdb file
    pdbdf=ppdb.df['HETATM']
    if len(pdbdf)==0:
        pdbdf=ppdb.df['ATOM']
    #[ppdb.df['HETATM']['element_symbol'] != 'OH'].head()
    pdbdf.shape
    #%%Functions
    #diostance in 3d space
    def dista(atom1, atom2):
        distancea=(((pdbdf['x_coord'][atom2]-pdbdf['x_coord'][atom1])**2)+((pdbdf['y_coord'][atom2]-pdbdf['y_coord'][atom1])**2)+((pdbdf['z_coord'][atom2]-pdbdf['z_coord'][atom1])**2))**0.5
        return distancea
    
    # angle between 3 atoms. atom1 is at the center
    # https://pycrawfordprogproj.readthedocs.io/en/latest/Project_01/Project_01.html, https://stackoverflow.com/questions/18945705/how-to-calculate-bond-angle-in-protein-db-file
        
    # list of bond length for some important functional groups: https://calculla.com/bond_lengths
    def angle(atom1,atom2,atom3):
        
        a12=np.subtract([pdbdf['x_coord'][atom2],pdbdf['y_coord'][atom2],pdbdf['z_coord'][atom2]],[pdbdf['x_coord'][atom1],pdbdf['y_coord'][atom1],pdbdf['z_coord'][atom1]])
        a13=np.subtract([pdbdf['x_coord'][atom3],pdbdf['y_coord'][atom3],pdbdf['z_coord'][atom3]],[pdbdf['x_coord'][atom1],pdbdf['y_coord'][atom1],pdbdf['z_coord'][atom1]])
        norm12 = a12 / (np.linalg.norm(a12)+0.00001)
        norm13 = a13 / (np.linalg.norm(a13)+0.00001)
        dotprud=np.dot(norm12,norm13)
        angle=np.degrees(np.arccos(dotprud))
        return angle
    #intersection (overlap) between two llists
    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))
    
    
    #%%
    #calculate molecular weight from smiles
    # http://rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html?highlight=pdb
    
    # m = Chem.MolFromSmiles('c1ccccc1C(=O)O')
    # mol_weight = Descriptors.MolWt(m)
    # m=rdkit.Chem.rdmolfiles.MolFromPDBFile('./water.pdb')
    # mol_weight = Descriptors.MolWt(m)
        #%%
        # reference_point = (9.362, 41.410, 10.542)
        # distances = p_1.distance(xyz=('ATOM',), records=('ATOM',))
        
        # distances.head()
        
    
    #%%  create a pandas df for the molecular fingerprint. 

    fpnames=pd.DataFrame({'molecular mass':[0],'mlogp':[0],'tpsa':[0],'HAccept':[0],'HDon':[0],'nrotate':[0]})
    
    
    #%%
    # for col in pdbdf.columns:
    #     print(col)
        
    # xx=pdbdf['atom_name']
    # print(xx)
    #%% Molecular formulaa: make molecular formula from pdb file. Later the formula will be used to calculare MW.
    # elsymb=pdbdf['element_symbol']
    
    #count number of each element
    countEl = pdbdf['element_symbol'].value_counts()
    # print(countEl)
    # count.index[0]
    formula=''
    
    for i in range(len(countEl)):
        # print(countEl[i])
        formula=formula+countEl.index[i]+str(countEl[i])
    # print(molecular_mass(formula))
    fpnames['molecular mass']=molecular_mass(formula)
    #%% Use SMILES to Calculate MlogP and TPSA (topological polar surface area) etc for lipophilicity assessment. Look at https://www.rdkit.org/docs/index.html for more parameters
    # convert pdb file to smiles  https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html
    mol = rdkit.Chem.rdmolfiles.MolFromPDBFile(filename)
    mlogp=Descriptors.MolLogP(mol)
    fpnames['mlogp']=mlogp
    tpsa=Descriptors.TPSA(mol)
    fpnames['tpsa']=tpsa
    nrotate=rdkit.Chem.Lipinski.NumRotatableBonds(mol)
    fpnames['nrotate']=nrotate
    
    
    # hbond acceptors and donors
    # HAccept=rdkit.Chem.Lipinski.NumHAcceptors(mol)
    ii=0
    jj=0
    for i in range(len(pdbdf['atom_name'])):
        if pdbdf['atom_name'][i]=='O' or pdbdf['atom_name'][i]=='N' or pdbdf['atom_name'][i]=='F':
            jj=jj+1
            for j in range(len(pdbdf['atom_name'])):
                if pdbdf['atom_name'][j]=='H' and dista(j,i)<1.2 and dista(j,i)>0.5:
                    ii=ii+1
                    # print('hydrogen bond donors:',i+1 ,j+1)
    # print('number of hbond donors:',ii, 'number of hbond acceptors:',jj)
    
    HDon=ii
    HAccept=jj
    fpnames['HDon']=ii
    fpnames['HAccept']=jj
    # Number of Rotatable Bonds
    nrotate=rdkit.Chem.Lipinski.NumRotatableBonds(mol)
    #---------------------------------------------------------------------------
    #calculate moments of iertia and principal axes of a molecule. for more information look at mInertia.py
    # calculate the center of mass (COM)
    # x coordinate of the COM
    # xcm=0
    # ycm=0
    # zcm=0
    # #some of atomic masses
    # msum=0
    # for i in range(len(pdbdf['element_symbol'])):
    #     xcm=xcm+((MM_of_Elements[pdbdf['element_symbol'][i]])*(pdbdf['x_coord'][i]))
    #     ycm=ycm+((MM_of_Elements[pdbdf['element_symbol'][i]])*(pdbdf['y_coord'][i]))
    #     zcm=zcm+((MM_of_Elements[pdbdf['element_symbol'][i]])*(pdbdf['z_coord'][i]))
    #     # print(xcm)
    #     msum=msum+MM_of_Elements[pdbdf['element_symbol'][i]]
    #     # print(msum)
    # xcm=xcm/msum
    # ycm=ycm/msum
    # zcm=zcm/msum
    # # principal moments of inertia, Is
    # ixx=0
    # iyy=0
    # izz=0
    # ixy=0
    # ixz=0
    # iyz=0
    # for i in range(len(pdbdf['element_symbol'])):
    #     ixx=ixx+(MM_of_Elements[pdbdf['element_symbol'][i]])*((pdbdf['y_coord'][i]-ycm)**2+(pdbdf['z_coord'][i]-zcm)**2)
    #     iyy=iyy+(MM_of_Elements[pdbdf['element_symbol'][i]])*((pdbdf['z_coord'][i]-zcm)**2+(pdbdf['x_coord'][i]-xcm)**2)
    #     izz=izz+(MM_of_Elements[pdbdf['element_symbol'][i]])*((pdbdf['x_coord'][i]-xcm)**2+(pdbdf['y_coord'][i]-ycm)**2)
    #     ixy=ixy+(MM_of_Elements[pdbdf['element_symbol'][i]])*(pdbdf['x_coord'][i]-xcm)*(pdbdf['y_coord'][i]-ycm)
    #     ixz=ixz+(MM_of_Elements[pdbdf['element_symbol'][i]])*(pdbdf['x_coord'][i]-xcm)*(pdbdf['z_coord'][i]-zcm)
    #     iyz=iyz+(MM_of_Elements[pdbdf['element_symbol'][i]])*(pdbdf['y_coord'][i]-ycm)*(pdbdf['z_coord'][i]-zcm)
        
    # ixy=-ixy
    # ixz=-ixz
    # iyz=-iyz

    # I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

    # Ip = np.linalg.eigvals(I)
    # # Sort and convert principal moments of inertia to SI (kg.m2)
    # Ip.sort()
    # #write Ips to fpnames
    # fpnames['Ip1']=Ip[0]
    # fpnames['Ip2']=Ip[1]
    # fpnames['Ip3']=Ip[2]
    
    #%% find a functional group or an structural scaffold in the PDB file based on the distances between the key atoms in the functional group
    #return coordinates of atoms
    # for col in pdbdf.columns:
    #     print(col)
    # xx=pdbdf['y_coord']
    # print(xx)
    # print(pdbdf['atom_name'])
    #atom names
    ####### carboxylic acid
    #he two carbon‑oxygen bonds in the delocalized carboxylate anion are identical (both 1.27 Å). However, in the structure of a carboxylic acid the  C−O bond (1.20 Å) is shorter than the  C−OH bond (1.34 Å). https://chem.libretexts.org/Bookshelves/Organic_Chemistry/Organic_Chemistry_(Morsch_et_al.)/20%3A_Carboxylic_Acids_and_Nitriles/20.02%3A_Structure_and_Properties_of_Carboxylic_Acids
    
    # carboxylic acid
    ii=0
    carboxylicOx=[]
    carboxylicAr=[]
    for i in range(len(pdbdf['atom_name'])):
        if 'O' in pdbdf['atom_name'][i]:
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.45 and dista(i,j)>1.19 :
                    for k in range(len(pdbdf['atom_name'])):
                        
                        if 'O' in pdbdf['atom_name'][k] and dista(k,j)<1.45 and dista(k,j)>1.19 and k!=i:
                            # print(k,i)
                            for l in range(len(pdbdf['atom_name'])):
                                if ('H' in pdbdf['atom_name'][l] and dista(l,k)<1.1) or (pdbdf['atom_name'][l]=='H' and dista(l,i)<1.1):
                                    ii=ii+1
                                    carboxylicOx.append(k)
                                    carboxylicAr.extend([i,j,k,l])
                                    # print('carboxylic acid for atoms:',j+1,i+1,k+1)
    carboxylicAr=list(set(carboxylicAr))                            
    # print('number of carboxylic acid groups:',ii/2)
    fpnames['carboxylic acid']=ii/2
    #%%
    # carboxylate
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        if 'O' in pdbdf['atom_name'][i]:
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.45 and dista(i,j)>1.19 :
                    for k in range(len(pdbdf['atom_name'])):
                        if 'O' in pdbdf['atom_name'][k] and dista(k,j)<1.45 and dista(k,j)>1.19 and k!=i and dista(i,j)/dista(j,k) <1.1 and dista(i,j)/dista(j,k) >0.95 and dista(j,k)<1.42:
                            ii=ii+1
                            # print('carboxylate found for atoms:',j+1,i+1,k+1)
                            
                            
                            
                                    
    # print('number of carboxylate groups:',ii/2)
    fpnames['carboxylate']=ii/2
                                    
    #%%
    #ester
    estoxy=[]
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        if 'O' in pdbdf['atom_name'][i]:
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.35 and dista(i,j)>1.0 :
                    for k in range(len(pdbdf['atom_name'])):
                        if 'O' in pdbdf['atom_name'][k] and dista(k,j)<1.6 and dista(k,j)>1.1 and k!=i:
                            for l in range(len(pdbdf['atom_name'])):
                                if (pdbdf['atom_name'][l]=='C' and dista(l,k)<1.6 and l!=j) or('C' in pdbdf['atom_name'][l] and dista(l,i)<1.6 and l!=j):
                                    ii=ii+1
                                    estoxy.append(i)
                                    estoxy.append(k)
                                    # print('ester for atoms:',j+1,i+1,k+1)
                                    
    estoxy=list(set(estoxy))
    # print('number of ester groups:', len(estoxy)/2)
    fpnames['ester']=len(estoxy)/2
    
    #ether
    # ii=0
    
    etherOxygenArr=[]
    # for m in range(len(pdbdf['atom_name'])):
    #     if 'O' in pdbdf['atom_name'][m]:
    #         oxyarr.append(m)
    #         oxyarr=list(set(oxyarr))                                    
    for i in range(len(pdbdf['atom_name'])):
        if 'O' in pdbdf['atom_name'][i]:
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.6 and dista(i,j)>1.19 :
                    # duma=[]
                    for k in range(len(pdbdf['atom_name'])):
                        if 'C' in pdbdf['atom_name'][k] and dista(k,i)<1.6 and dista(k,i)>1.19 and k!=j and dista(k,j)>1.6:
                            
                            # for l in range(len(pdbdf['atom_name'])):
                            #     if (dista(k,l)<1.6 and dista(k,l)>0.5 and l!=i) or (dista(j,l)<1.6 and dista(j,l)>0.5 and l!=i):
                                    # print(l)
                                    # duma.append(l)
                                    
                                    # duma=list(set(duma))
                                    
                                    # # print(duma)
                                    # if intersection(duma, oxyarr)==[]:
                                        
                                    #     ii=ii+1
                                        
                                # print('ether found for atoms:',i+1,j+1,k+1,l+1)
                                if i not in etherOxygenArr and i not in estoxy:
                                    etherOxygenArr.append(i)
                                    # print(duma)
                                            
    # print('number of ether groups:',len(etherOxygenArr))
    fpnames['ether']=len(etherOxygenArr) 
    #%% alcohol
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        if 'O' in  pdbdf['atom_name'][i]:
            if i not in carboxylicOx:
                
                for j in range(len(pdbdf['atom_name'])):
                    if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.45 and dista(i,j)>1.19 :
                         for m in range(len(pdbdf['atom_name'])):
                             if 'H' in pdbdf['atom_name'][m] and dista(m,i)<1.2 and dista(m,i)>0.5:
                                ii=ii+1
                                # print('alcohol for atoms:',i+1,j+1,m+1)
    # print('number of alcohol groups:',ii)
    fpnames['alcohol']=ii
      
    
    #%%
    #amide
    amideN=[]
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        if 'O' in pdbdf['atom_name'][i]:
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.30 and dista(i,j)>1.19 :
                    for k in range(len(pdbdf['atom_name'])):
                        
                        if 'N' in pdbdf['atom_name'][k] and dista(k,j)<1.6 and dista(k,j)>1.19:
                            # print(k,i)
                            for l in range(len(pdbdf['atom_name'])):
                                if ('C' in pdbdf['atom_name'][l] and dista(l,k)<1.6 and l!=j) or('C' in pdbdf['atom_name'][l] and dista(l,i)<1.45 and l!=j):
                                    # print('amide for atoms:',j+1,i+1,k+1,l+1)   
                                    ii=ii+1
                                    amideN.append(k)
    amideN=list(set(amideN))
    # print('number of amide groups:',len(amideN)) 
    fpnames['amide']=len(amideN)                                     
    #%% primary amine
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        if 'N' in pdbdf['atom_name'][i] and i not in amideN:
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.6 and dista(i,j)>1.19 :
                    for k in range(len(pdbdf['atom_name'])):                  
                        if 'C' in pdbdf['atom_name'][k] and dista(k,j)<1.6 and dista(k,j)>1.1:
                            for l in range(len(pdbdf['atom_name'])):
                                if 'C' in pdbdf['atom_name'][l] and dista(l,j)<1.6 and dista(l,j)>1.1 and l!=k:
                                    for m in range(len(pdbdf['atom_name'])):
                                        if 'H' in pdbdf['atom_name'][m] and dista(m,i)<1.2 and dista(m,i)>0.5:
                                            for n in range(len(pdbdf['atom_name'])):
                                                if 'H' in pdbdf['atom_name'][n] and dista(n,i)<1.2 and dista(n,i)>0.5 and n!=m:
                                                    ii=ii+1
                                                    # print('primary amine for atoms:',i+1,j+1,k+1,l+1,m+1,n+1)
                                                  
    # print('number of primary amines:', ii/4)                                                
                                                    
                                                    # print('primary amine for atoms:',i+1,j+1,k+1,l+1,m+1)
    fpnames['primary amine']=ii/4
    #%% secondary amine
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        if 'N' in pdbdf['atom_name'][i] and i not in amideN:
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.6 and dista(i,j)>1.19 :
                    for k in range(len(pdbdf['atom_name'])):                  
                        if 'C' in pdbdf['atom_name'][k] and dista(k,i)<1.6 and dista(k,i)>1.1 and k!=j:
                            for l in range(len(pdbdf['atom_name'])):
                                if 'H' in pdbdf['atom_name'][l] and dista(l,i)<1.2 and dista(l,i)>0.5:
                                    ii=ii+1
                                    # print('secondary amine for atoms:',i+1,j+1,k+1,l+1)
    # print('number of secondary amines:', ii/2)
    fpnames['secondary amine']=ii/2
    
    #%% tertiary amine or 4 valence nitrogen
    TertAmAr=[]
    ii=0
    jj=0
    for i in range(len(pdbdf['atom_name'])):
        if 'N' in pdbdf['atom_name'][i] and i not in amideN:
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.6 and dista(i,j)>1.19 :
                    for k in range(len(pdbdf['atom_name'])):                  
                        if 'C' in pdbdf['atom_name'][k] and dista(k,i)<1.6 and dista(k,i)>1.1 and k!=j:
                            for l in range(len(pdbdf['atom_name'])):
                                if 'C' in pdbdf['atom_name'][l] and dista(l,i)<1.6 and dista(l,i)>1.0 and l!=k and l!=j:
                                     for m in range(len(pdbdf['atom_name'])):
                                         
                                         if 'C' in pdbdf['atom_name'][m] and dista(m,i)<1.6 and dista(m,i)>1.0 and m!=k and m!=j and m!=l:
                                    
                                             ii=ii+1
                                             jj=i
                                             # print('4 valence nitrogen:',i+1,j+1,k+1,l+1,m+1)
                                         
    # print('tertiary amine between:',i+1,j+1,k+1,l+1,m+1)
                                             
    # print('number of tertiary amines:', ii/24)
    # print('number of 4 valence nitrogen:',ii/24)                                             
    fpnames['4 valenced nitrogen']=ii/24  
    
    ii=0
    
    for i in range(len(pdbdf['atom_name'])):
        if 'N' in pdbdf['atom_name'][i] and i!=jj and i not in amideN:
            
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.6 and dista(i,j)>1.19 :
                    for k in range(len(pdbdf['atom_name'])):                  
                        if 'C' in pdbdf['atom_name'][k] and dista(k,i)<1.6 and dista(k,i)>1.1 and k!=j:
                            for l in range(len(pdbdf['atom_name'])):
                                if 'C' in pdbdf['atom_name'][l] and dista(l,i)<1.6 and dista(l,i)>1.0 and l!=k and l!=j:
                                    ii=ii+1
                                    TertAmAr.extend([i])  #tertiary amine array
                                                
                                                    
                                                    
                               

                                    # print('tertiary amine between:',i+1,j+1,k+1,l+1)
                                         
    # print('tertiary amine between:',i+1,j+1,k+1,l+1,m+1)
                                             
    # print('number of tertiary amines:', ii/24)
    # print('number of tertiary amine:',ii/6)  
    TertAmAr=list(set(TertAmAr))                                           
    fpnames['tertiary amine']=ii/6 
    
                          
    #%% keton
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        if 'O' in pdbdf['atom_name'][i]:
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.26 and dista(i,j)>1.0 :
                    for k in range(len(pdbdf['atom_name'])):                  
                        if 'C' in pdbdf['atom_name'][k] and dista(k,j)<1.6 and dista(k,j)>1.1:
                            for l in range(len(pdbdf['atom_name'])):
                                if 'C' in pdbdf['atom_name'][l] and dista(l,j)<1.6 and dista(l,j)>1.1 and l!=k:
                                    ii=ii+1
                                    # print('ketone for atoms:',i+1,j+1,k+1,l+1)
    # print('number of ketone groups:',ii/2)
    fpnames['keton']=ii/2
    #%% aldehyde
    # ii=0
    # for i in range(len(pdbdf['atom_name'])):
    #     if 'O' in pdbdf['atom_name'][i]:
    #         for j in range(len(pdbdf['atom_name'])):
    #             if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.26 and dista(i,j)>1.0 :
    #                 for k in range(len(pdbdf['atom_name'])):                  
    #                     if 'C' in pdbdf['atom_name'][k]=='C' and dista(k,j)<1.6 and dista(k,j)>1.1:
    #                         for l in range(len(pdbdf['atom_name'])):
    #                             if 'H' in pdbdf['atom_name'][l] and dista(l,j)<1.5 and dista(l,j)>0.8 and angle(j,l,i)>118:
    #                                 print('aldehyde for atoms:',i+1,j+1,k+1,l+1)
    #                                 ii=ii+1
    # print('number of aldehyde groups:',ii)
    # fpnames['aldehyde']=ii
      
    #%%                                                                               
    
    
    #%%                                                                               
    #ether
    # ii=0
    # duma=[]
    # oxyarr=[]
    # for m in range(len(pdbdf['atom_name'])):
    #     if pdbdf['atom_name'][m]=='O':
    #         oxyarr.append(m)
    #         oxyarr=list(set(oxyarr))                                    
    # for i in range(len(pdbdf['atom_name'])):
    #     if pdbdf['atom_name'][i]=='O':
    #         for j in range(len(pdbdf['atom_name'])):
    #             if pdbdf['atom_name'][j]=='C' and dista(i,j)<1.6 and dista(i,j)>1.19 :
    #                 duma=[]
    #                 for k in range(len(pdbdf['atom_name'])):
    #                     if pdbdf['atom_name'][k]=='C' and dista(k,i)<1.6 and dista(k,i)>1.19 and k!=j:  
    #                         for l in range(len(pdbdf['atom_name'])):
    #                             if (dista(k,l)<1.6 and dista(k,l)>0.5 and l!=i) or (dista(j,l)<1.6 and dista(j,l)>0.5 and l!=i):
    #                                 duma.append(l)
    #                                 duma=list(set(duma))
    #                                 # print(duma)
    #                                 if intersection(duma, oxyarr)==[] and len(duma)==6:
    #                                     ii=ii+1
    #                                     print('ether found for atoms:',i+1,j+1,k+1,l+1)
    # print('number of ether groups:',ii/2)
    # fpnames['ether']=ii/2   
    #%%
                               
    #%%                                                                               
    #cyanide
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        if 'N' in pdbdf['atom_name'][i]:
            for j in range(len(pdbdf['atom_name'])):
                if pdbdf['atom_name'][j]=='C' and dista(i,j)<1.25 :
                    ii=ii+1
                    # print('cyanide for atoms:',j+1,i+1)
    # print('number of cyanide groups:',ii)
    fpnames['cyanide']=ii
    # #%%                                                                               
    # #cyanate
    # ii=0
    # for i in range(len(pdbdf['atom_name'])):
    #     if 'N' in pdbdf['atom_name'][i]:
    #         for j in range(len(pdbdf['atom_name'])):
    #             if pdbdf['atom_name'][j]=='C' and dista(i,j)<1.20:
    #                 for k in range(len(pdbdf['atom_name'])):
    #                     if 'O' in pdbdf['atom_name'][k] and dista(k,j)<1.5 and dista (i,k) <3 and angle(j,i,k) >150 and angle(j,i,k) <190:
    #                         ii=ii+1
    #                         print('cyanate for atoms:',j+1,i+1,k+1)
    # print('number of cyanide groups:',ii)
    # fpnames['cyanate']=ii      
    
    #%%
    #epoxide
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        if 'O' in pdbdf['atom_name'][i]:
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.60:
                    for k in range(len(pdbdf['atom_name'])):
                        if 'C' in pdbdf['atom_name'][k] and dista(k,i)<1.6 and angle(j,i,k) <65 and dista(k,j)<1.6:
                            ii=ii+1
                            # print('epoxide for atoms:',j+1,i+1,k+1)
    # print('number of epoxide groups:',ii/2)
    fpnames['epoxide']=ii/2        
                                                                                                                                        
    #%%
    #Sulfonyl
    Sulfonylsu=[]
    
    for i in range(len(pdbdf['atom_name'])):
        if 'S' in pdbdf['atom_name'][i]:
            # print(i)
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.9:
                    # print(j)
                    for k in range(len(pdbdf['atom_name'])):
                        
                        if dista(k,i)<1.6 and k!=j:
                            # print(k,i)
                            for l in range(len(pdbdf['atom_name'])):
                                if ('O' in pdbdf['atom_name'][l] and dista(l,i)<1.9):
                                    for m in range(len(pdbdf['atom_name'])):
                                        if 'O' in pdbdf['atom_name'][m] and dista(m,i)<1.6 and m!=l:
                                            # print('Sulfonyl for atoms:',j+1,i+1,k+1,l+1, m+1)
                                            Sulfonylsu.append(i)
                                    # print('amide for atoms:',j+1,i+1,k+1,l+1)   
                                    
                                    
    Sulfonylsu=list(set(Sulfonylsu))
    # print('number of sulfonyl groups:',len(Sulfonylsu)) 
    fpnames['sulfonyl']=len(Sulfonylsu)  
    
    #%% cyclohexane
    cyclohexAr=[]
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        if 'C' in pdbdf['atom_name'][i]:
            # print("###########################",i)
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.6 and dista(i,j)>1.45 and i!=j:
                    # print("************************",i,j, dista(i,j))
                    for k in range(len(pdbdf['atom_name'])):                  
                        if 'C' in pdbdf['atom_name'][k] and dista(k,j)<1.6 and dista(k,j)>1.45 and k not in [i,j]:
                            for l in range(len(pdbdf['atom_name'])):
                                if 'C' in pdbdf['atom_name'][l] and dista(l,k)>1.45 and dista(l,k)<1.6 and l not in[i, j, k]:
                                    for m in range(len(pdbdf['atom_name'])):
                                        if 'C' in pdbdf['atom_name'][m] and dista(l,m)>1.45 and dista(l,m)<1.6 and m not in [i, j, k, l]:
                                            for n in range(len(pdbdf['atom_name'])):
                                                if 'C' in pdbdf['atom_name'][n] and dista(n,m)>1.45 and dista(n,m)<1.6 and dista(n,i)>1.45 and dista(n,i)<1.6 and n not in [i,j,k,l,m]:
                                                    
                                                    ii=ii+1
                                                    # print(i+1,j+1,k+1,l+1,m+1,n+1)
                                                    cyclohexAr.extend([i,j,k,l,m,n])
                                   
    cyclohexAr=list(set(cyclohexAr))                               
    fpnames['cyclohexane']=ii/12   
    # print([x + 1 for x in cyclohexAr])
    # print(ii)
    #%% cyclopentane
    cyclopenAr=[]
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        if 'C' in pdbdf['atom_name'][i]:
            # print("###########################",i)
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.6 and dista(i,j)>1.45 and i!=j:
                    # print("************************",i,j, dista(i,j))
                    for k in range(len(pdbdf['atom_name'])):                  
                        if 'C' in pdbdf['atom_name'][k] and dista(k,j)<1.6 and dista(k,j)>1.45 and k not in [i,j]:
                            for l in range(len(pdbdf['atom_name'])):
                                if 'C' in pdbdf['atom_name'][l] and dista(l,k)>1.45 and dista(l,k)<1.6 and l not in[i, j, k]:
                                    for m in range(len(pdbdf['atom_name'])):
                                        if 'C' in pdbdf['atom_name'][m] and dista(l,m)>1.45 and dista(l,m)<1.6 and dista(i,m)>1.45 and dista(i,m)<1.6 and m not in [i, j, k, l]:
                                                ii=ii+1
                                                # print(i+1,j+1,k+1,l+1,m+1,n+1)
                                                cyclopenAr.extend([i,j,k,l,m])
                                   
    cyclopenAr=list(set(cyclopenAr))                               
    # fpnames['cyclopentane']=len(cyclopenAr)/5  
    # # print([x + 1 for x in cyclopenAr])                   
    #%% piperidine
    pprAr=[]
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        if 'N' in pdbdf['atom_name'][i]:
            # print("###########################",i)
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.6 and dista(i,j)>1.19 and i!=j:
                    # print("************************",i,j, dista(i,j))
                    for k in range(len(pdbdf['atom_name'])):                  
                        if 'C' in pdbdf['atom_name'][k] and dista(k,j)<1.6 and dista(k,j)>1.45 and k not in [i,j]:
                            # print("_-_",i+1,j+1,k+1)
                            for l in range(len(pdbdf['atom_name'])):
                                if 'C' in pdbdf['atom_name'][l] and dista(l,k)>1.45 and dista(l,k)<1.6 and l not in[i, j, k]:
                                    # print("_-_",l+1)
                                    for m in range(len(pdbdf['atom_name'])):
                                        if 'C' in pdbdf['atom_name'][m] and dista(l,m)>1.45 and dista(l,m)<1.6 and m not in [i, j, k, l]:
                                            # print("_-_",m+1)
                                            for n in range(len(pdbdf['atom_name'])):
                                                if 'C' in pdbdf['atom_name'][n] and dista(n,m)>1.45 and dista(n,m)<1.6 and dista(n,i)>1.45 and dista(n,i)<1.6 and n not in [i,j,k,l,m]:
                                                    
                                                    ii=ii+1
                                                    # print(i+1,j+1,k+1,l+1,m+1,n+1)
                                                    pprAr.extend([i])
                                   
    pprAr=list(set(pprAr))                               
    fpnames['piperidine']=len(pprAr)  
    # print([x + 1 for x in pprAr])
    #%% pyrrolidine
    pyrrolidineAr=[]
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        if 'N' in pdbdf['atom_name'][i]:
            # print("###########################",i)
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.6 and dista(i,j)>1.19 and i!=j:
                    # print("************************",i,j, dista(i,j))
                    for k in range(len(pdbdf['atom_name'])):                  
                        if 'C' in pdbdf['atom_name'][k] and dista(k,j)<1.6 and dista(k,j)>1.45 and k not in [i,j]:
                            for l in range(len(pdbdf['atom_name'])):
                                if 'C' in pdbdf['atom_name'][l] and dista(l,k)>1.45 and dista(l,k)<1.6 and l not in[i, j, k]:
                                    for m in range(len(pdbdf['atom_name'])):
                                        if 'C' in pdbdf['atom_name'][m] and dista(l,m)>1.45 and dista(l,m)<1.6 and dista(i,m)>1.45 and dista(i,m)<1.6 and m not in [i, j, k, l]:
                                                ii=ii+1
                                                # print(i+1,j+1,k+1,l+1,m+1,n+1)
                                                pyrrolidineAr.extend([i])
                                   
    pyrrolidineAr=list(set(pyrrolidineAr))                               
    fpnames['pyrrolidine']=len(pyrrolidineAr) 
    # print([x + 1 for x in pyrrolidineAr])
    #%% benzene
    benzAr=[]
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        if 'C' in pdbdf['atom_name'][i]:
            # print("###########################",i)
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.45 and dista(i,j)>1.34 and i!=j:
                    # print("************************",i,j, dista(i,j))
                    for k in range(len(pdbdf['atom_name'])):                  
                        if 'C' in pdbdf['atom_name'][k] and dista(k,j)<1.45 and dista(k,j)>1.34 and k not in [i,j]:
                            for l in range(len(pdbdf['atom_name'])):
                                if 'C' in pdbdf['atom_name'][l] and dista(l,k)>1.34 and dista(l,k)<1.45 and l not in[i, j, k]:
                                    for m in range(len(pdbdf['atom_name'])):
                                        if 'C' in pdbdf['atom_name'][m] and dista(l,m)>1.34 and dista(l,m)<1.45 and m not in [i, j, k, l]:
                                            for n in range(len(pdbdf['atom_name'])):
                                                if 'C' in pdbdf['atom_name'][n] and dista(n,m)>1.34 and dista(n,m)<1.45 and dista(n,i)>1.36 and dista(n,i)<1.45 and n not in [i,j,k,l,m]:
                                                    
                                                    ii=ii+1
                                                    # print(i+1,j+1,k+1,l+1,m+1,n+1)
                                                    benzAr.extend([i,j,k,l,m,n])
                                   
    benzAr=list(set(benzAr)) 
            
            
            
                              
    fpnames['benzene']=round(ii/12)  
    # print([x + 1 for x in cyclohexAr])
    # print(ii)
    #%% Methoxybenzene, methylaniline 
    MethoxybenzeneAr=[]
    methylanilineAr=[]
    # this one is seen in some drugs of abuse such as opioids
    PhenethylamineAr=[]
    for i in range(len(benzAr)):
        for j in range(len(pdbdf['atom_name'])):
            if 'O' in pdbdf['atom_name'][j] and dista(benzAr[i],j)<1.6 and dista(benzAr[i],j)>1:
                MethoxybenzeneAr.extend([j])
            elif 'N' in pdbdf['atom_name'][j] and dista(benzAr[i],j)<1.6 and dista(benzAr[i],j)>1:
                methylanilineAr.extend([j])
            elif 'C' in pdbdf['atom_name'][j] and dista(benzAr[i],j)<1.6 and dista(benzAr[i],j)>1 and j not in benzAr:
                for k in range(len(pdbdf['atom_name'])):
                    if 'C' in pdbdf['atom_name'][k] and dista(k,j)<1.6 and dista(k,j)>1 and k not in benzAr:
                        for l in range(len(pdbdf['atom_name'])):
                            if 'N' in pdbdf['atom_name'][l] and dista(k,l)<1.6 and dista(k,l)>1 and l not in PhenethylamineAr and l not in methylanilineAr:
                                PhenethylamineAr.extend([l])
                                
                                
    MethoxybenzeneAr=list(set(MethoxybenzeneAr))
    fpnames['Methoxybenzene']=len(MethoxybenzeneAr)
    methylanilineAr=list(set(methylanilineAr))
    fpnames['methylaniline']=len(methylanilineAr)
    PhenethylamineAr=list(set(PhenethylamineAr))
    fpnames['Phenethylamine']=len(PhenethylamineAr)
    
                
    
    
    #%% pyridine
    pyrAr=[]
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        if 'N' in pdbdf['atom_name'][i]:
            # print("###########################",i)
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)>1.19 and dista(i,j)<1.5 and i!=j:
                    # print("************************",i,j, dista(i,j))
                    for k in range(len(pdbdf['atom_name'])):                  
                        if 'C' in pdbdf['atom_name'][k] and dista(k,j)<1.45 and dista(k,j)>1.38 and k not in [i,j]:
                            for l in range(len(pdbdf['atom_name'])):
                                if 'C' in pdbdf['atom_name'][l] and dista(l,k)>1.38 and dista(l,k)<1.45 and l not in[i, j, k]:
                                    for m in range(len(pdbdf['atom_name'])):
                                        if 'C' in pdbdf['atom_name'][m] and dista(l,m)>1.38 and dista(l,m)<1.45 and m not in [i, j, k, l]:
                                            for n in range(len(pdbdf['atom_name'])):
                                                if 'C' in pdbdf['atom_name'][n] and dista(n,m)>1.38 and dista(n,m)<1.45 and dista(n,i)>1.19 and dista(n,i)<1.5 and n not in [i,j,k,l,m]:
                                                    # print("))))))))))))))",i)
                                                    ii=ii+1
                                                    # print(i+1,j+1,k+1,l+1,m+1,n+1)
                                                    pyrAr.extend([i])
                                                    
                                                        
                                                        
                                   
    pyrAr=list(set(pyrAr))                               
    fpnames['pyridine']=len(pyrAr) 
    # print(pyrAr)
    #%% pyrrole
    pyrroleAr=[]
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        if 'N' in pdbdf['atom_name'][i]:
            # print("###########################",i)
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.6 and dista(i,j)>1.19 and i!=j:
                    # print("************************",i,j, dista(i,j))
                    for k in range(len(pdbdf['atom_name'])):                  
                        if 'C' in pdbdf['atom_name'][k] and dista(k,j)<1.45 and dista(k,j)>1.3 and k not in [i,j]:
                            for l in range(len(pdbdf['atom_name'])):
                                if 'C' in pdbdf['atom_name'][l] and dista(l,k)>1.19 and dista(l,k)<1.6 and l not in[i, j, k]:
                                    for m in range(len(pdbdf['atom_name'])):
                                        if 'C' in pdbdf['atom_name'][m] and dista(l,m)>1.19 and dista(l,m)<1.6 and dista(i,m)>1.19 and dista(i,m)<1.6 and m not in [i, j, k, l]:
                                                ii=ii+1
                                                # print(i+1,j+1,k+1,l+1,m+1,n+1)
                                                pyrroleAr.extend([i])
                                   
    pyrroleAr=list(set(pyrroleAr))                               
    fpnames['pyrrole']=len(pyrroleAr) 
    # print([x + 1 for x in pyrrolidineAr])
    #%% number of fluorine atoms. F atoms are believed to favor BBB penetration. DOI: 10.1021/acs.jmedchem.1c00910
    FlAr=[]
    for i in range(len(pdbdf['atom_name'])):
        if 'F' in pdbdf['atom_name'][i]:
            FlAr.extend([i])
    FlAr=list(set(FlAr))                               
    fpnames['fluorine']=len(FlAr)
    #%% LAT1 substrate alert. DOI:10.1021/acs.jmedchem.1c00910
    latAr=[]
    ii=0
    
    for o in range(len(pdbdf['atom_name'])):
        if 'C' in pdbdf['atom_name'][o]:
            for i in range(len(benzAr)) :
                if dista(benzAr[i],o)>1.19 and dista(benzAr[i],o)<1.6 and o not in benzAr:
                    for p in range(len(pdbdf['atom_name'])):
                        if 'C' in pdbdf['atom_name'][p] and dista(p,o)>1.19 and dista(p,o)<1.6 and p not in [o]:
                            for q in range(len(pdbdf['atom_name'])):
                                if 'N' in pdbdf['atom_name'][q] and dista(p,q)>1.19 and dista(p,q)<1.6 :
                                    for r in range(len(pdbdf['atom_name'])):
                                        if 'C' in pdbdf['atom_name'][r] and dista(p,r)>1.19 and dista(p,r)<1.6 and r not in [o,p] and r in carboxylicAr:
                                            ii=ii+1
                                            latAr.extend([o])
                                                    
                                                    # ii=ii+1
                                                    # # print(i+1,j+1,k+1,l+1,m+1,n+1)
                                                    # latAr.extend([i])
                                   
    latAr=list(set(latAr))
    # print(latAr)                               
    fpnames['lat1']=len(latAr)
    # print([x + 1 for x in cyclohexAr])
    # print(ii)  
    #%% linear alkane chain carbons
    laccAr=[]
    ii=0
    
    for i in range(len(pdbdf['atom_name'])):
        if 'C' in pdbdf['atom_name'][i] and i not in cyclohexAr and i not in cyclopenAr:
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)>1.19 and dista(i,j)<1.6 and j not in laccAr:
                    for k in range(len(pdbdf['atom_name'])):
                        if 'C' in pdbdf['atom_name'][k] and dista(i,k)>1.19 and dista(i,k)<1.6 and k!=j:
                            for l in range(len(pdbdf['atom_name'])):
                                if 'H' in pdbdf['atom_name'][l] and dista(i,l)>0.9 and dista(i,l)<1.5:
                                    for m in range(len(pdbdf['atom_name'])):
                                        if 'H' in pdbdf['atom_name'][m] and dista(i,m)>0.9 and dista(i,m)<1.5 and m!=l:
                                            laccAr.extend([i])
           
                                            
                                   
    laccAr=list(set(laccAr))
    # print(laccAr)                               
    fpnames['lacc']=len(laccAr)
    
    
    # butylAr=[]
    # pentylAr=[]
    # hexylPlusAr=[]
    # for i in range(len(laccAr)) :
    #     for j in range(len(laccAr)):
    #         if dista(laccAr[i],laccAr[j])>1.19 and dista(laccAr[i],laccAr[j])<1.6:
    #             for k in range(len(laccAr)):
    #                 if dista(laccAr[k],laccAr[j])>1.19 and dista(laccAr[k],laccAr[j])<1.6 and k!=j:
    #                     for l in range(len(laccAr)):
    #                         if dista(laccAr[k],laccAr[l])>1.19 and dista(laccAr[k],laccAr[l])<1.6 and l not in [i,j,k]:
    #                             butylAr.extend([laccAr[i],laccAr[j],laccAr[k],laccAr[l]])
    #                             for m in range(len(laccAr)):
    #                                 if dista(laccAr[m],laccAr[l])>1.19 and dista(laccAr[m],laccAr[l])<1.6 and m not in [i,j,k,l]:
    #                                     pentylAr.extend([laccAr[i],laccAr[j],laccAr[k],laccAr[l],laccAr[m]])
    #                                     for o in range(len(laccAr)):
    #                                         if dista(laccAr[m],laccAr[o])>1.19 and dista(laccAr[m],laccAr[o])<1.6 and o not in [i,j,k,l,m]:
    #                                             hexylPlusAr.extend([laccAr[i],laccAr[j],laccAr[k],laccAr[l],laccAr[m],laccAr[o]])
                                
    # butylAr=list(set(butylAr))
    # pentylAr=list(set(pentylAr))
    # hexylPlusAr=list(set(hexylPlusAr))
    # print("but  ",butylAr)
    # print("pent  ",pentylAr)
    # print("hex   ",hexylPlusAr)
    # print(laccAr)
                        
                
            
            
    #%% any 6 atom ring
    saAr=[]
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        
       
        # print("###########################",i)
        for j in range(len(pdbdf['atom_name'])):
            if  dista(i,j)<1.57 and dista(i,j)>1 and 'H' not in pdbdf['atom_name'][j] and 'H' not in pdbdf['atom_name'][i]:
                # print("************************",i,j, dista(i,j))
                for k in range(len(pdbdf['atom_name'])):                  
                    if  dista(k,j)<1.6 and dista(k,j)>1 and k not in [i,j] and 'H' not in pdbdf['atom_name'][k]:
                        # print(i,j,k)
                        for l in range(len(pdbdf['atom_name'])):
                            if  dista(l,k)>1 and dista(l,k)<1.6 and l not in[i,j,k] and 'H' not in pdbdf['atom_name'][l]:
                                for m in range(len(pdbdf['atom_name'])):
                                    if dista(l,m)>1.0 and dista(l,m)<1.6 and m not in [i,j, k, l] and 'H' not in pdbdf['atom_name'][m]:
                                        for n in range(len(pdbdf['atom_name'])):
                                            if dista(n,m)>1.0 and dista(n,m)<1.6 and dista(n,i)>1.0 and dista(n,i)<1.6 and n not in [i,j,k,l,m] and 'H' not in pdbdf['atom_name'][n]:
                                                
                                                ii=ii+1
                                                
                                                # print(i+1,j+1,k+1,l+1,m+1,n+1)
                                                saAr.extend([i,j,k,l,m,n])
                                   
    saAr=list(set(saAr)) 
    # print([x + 1 for x in saAr])  
    # print('number of rings', ii)   
    # print(saAr)
                              
    fpnames['6 atom ring']=round(ii/12) # it is rounded to avoid giving floats 
    fpnames['common atoms in fused 6 atom rings']= round((ii/2)-(len(saAr))) # if an atom is common in 3 rings it will be counted twice.
    # print([x + 1 for x in cyclohexAr])
    # print(ii)

    #%% a benzene ring connected to a tertiary amine via two carbon atoms. The feature is seen in opioids
    opAr=[]
    ii=0
    for i in range(len(pdbdf['atom_name'])):
        if i in benzAr:
            # print("###########################",i)
            for j in range(len(pdbdf['atom_name'])):
                if 'C' in pdbdf['atom_name'][j] and dista(i,j)<1.6 and dista(i,j)>1.0 and i!=j:
                    # print("************************",i,j, dista(i,j))
                    for k in range(len(pdbdf['atom_name'])):                  
                        if 'C' in pdbdf['atom_name'][k] and dista(k,j)<1.6 and dista(k,j)>1.0 and k not in [i,j]:
                            for l in range(len(pdbdf['atom_name'])):
                                if l in TertAmAr and dista(l,k)>1.0 and dista(l,k)<1.6 and l not in[i, j, k]:
                                    ii=ii+1
                                    opAr.extend([l])
                                                    
                                                    
                                                    # print(i+1,j+1,k+1,l+1,m+1,n+1)
                                                   
                                   
    opAr=list(set(opAr)) 
    fpnames['opioid feature']=len(opAr)     
    #%% longest distance
    # distanceAr=[]
    # ii=0
    # for i in range(len(pdbdf['atom_name'])):
    #     for j in range(len(pdbdf['atom_name'])):
    #         if dista(i,j) not in distanceAr:
    #             distanceAr.extend([dista(i,j)])
    #             ii=ii+1
                
        
                                               
                                   
    # distanceAr=list(set(distanceAr)) 
    # # print(distanceAr)                              
    # fpnames['LongestDistance']=max(distanceAr)
    
    

    #%%                                 
                                            
                                   
      
    # fpArray.append(fpnames) 
    return  fpnames  
#%%  append fpnames to fpArray 
fpath='D:\Codes\Structures\pdb\\'
#temporary test files
# fpath='D:\Codes\\Structures\\pdb\\temp\\'
count = 0
# Iterate directory
for path in os.listdir(fpath):
    # check if current path is a file
    if os.path.isfile(os.path.join(fpath, path)):
        count += 1
        fpnamesO=molFp(count)
        # fpArray = fpArray.append(fpnamesO, ignore_index = True) 
        fpArray = pd.concat([fpArray,fpnamesO])
        print('*************************  File count:', count,'********************')

#%% write fpArray as a file
fpArray.to_csv('fpArray.csv')

                     
