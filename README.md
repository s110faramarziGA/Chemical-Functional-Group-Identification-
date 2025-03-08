# 3D Molecular Fingerprint Generator

## Overview

This Python script processes Protein Data Bank (PDB) files to generate **3D molecular fingerprints**. These fingerprints include:
- Functional groups (e.g., hydrogen bond donors, carboxylic acids, etc.).
- Molecular scaffolds.
- Physical properties like molecular mass, polar surface area, and rotatable bonds.

The fingerprints are compiled into a pandas DataFrame (`fpArray`) which can be used as input for machine learning algorithms.

---

## Features

- **Batch Processing of PDB Files**:
  - Iteratively processes all PDB files in a specified directory (`D:\Professional\TheProject\Codes\Structures\pdb\`).

- **Physical Property Calculation**:
  - Molecular mass.
  - Octanol-water partition coefficient (MLogP).
  - Topological polar surface area (TPSA).
  - Number of rotatable bonds.
  - Hydrogen bond acceptors and donors.

- **Functional Group Identification**:
  - Detects and quantifies functional groups such as:
    - Carboxylic acids.
    - Carboxylates.

- **SMILES Conversion**:
  - Converts PDB files to SMILES strings for additional molecular property computation.

- **Detailed 3D Molecular Data**:
  - Calculates distances and angles between atoms in 3D space.
  - Identifies overlapping and interacting atomic groups.

- **Output**:
  - The final DataFrame, `fpArray`, is written to a CSV file (`fpArray.csv`) for downstream analysis.

---

## Requirements

To run this script, you need:

- Python 3.x
- Required packages:
  - `numpy`
  - `pandas`
  - `tqdm`
  - `rdkit`
  - `biopandas`
  - `sympy`
  - `biopython`

Install missing dependencies with:
```bash
pip install numpy pandas tqdm rdkit biopandas sympy biopython

