# Alloy Surface Simulator

## About

This code is designed to simulate the likely arrangement of atoms in a binary metal alloy nanomaterial and to calculate the energy and properties for a given alloy configuration. Given the metals (and adsorbates, if applicable) and nominal composition for a nanomaterial of specified size and set of Miller indices, the code generates an arbitrary configuration that satisfies the composition and size requirements. A Metropolis Monte Carlo simulation then samples the space of potential configurations by swapping atoms to generate new arrangements, calculating the energy of the new arrangements, and accepting the swaps if they meet probability criteria. If a simulation is run long enough, the alloy nanomaterial will reach “equilibrium,” i.e. the sample space will have been explored sufficiently enough that calculating the average of the energy and properties of configurations in the sample space gives an accurate representation of what we would expect from the actual nanomaterial. 

## Uses

As written, the code is used to study binary Pt-Au nanomaterials, with and without oxygen adsorbates, in support of a wet lab experimental project that explores use of these materials as catalysts for fuel cell and organic polymer reactions (*LINK TO PAPER HERE*). However, the code can easily be adapted to study nanomaterials comprised of various combinations of metals. Size, nominal composition, lattice geometry (Miller indices), and fraction of adsorbate coverage can all be easily modified.
See the documentation for specific uses of each function.

## Getting Started

Before using the code it is necessary to install the Atomic Simulation Environment (ASE) Python library. See documentation here: https://wiki.fysik.dtu.dk/ase/.

All other libraries and packages should automatically come with Python. 

Work is done in a Jupyter notebook (see https://jupyter.org/ for installation instructions, or alternatively obtain Jupyter by downloading Anaconda for Python: https://www.anaconda.com/products/individual#Downloads). Jupyter notebooks are also compatible with Google Colab. 

Once Jupyter is installed, download WF_1.csv and WF_2.csv in the directory from which you will use Jupyter notebooks. These files contain necessary weight factor parameters to calculate the relative weights of each atom in the alloy's bonds.

Open a new Jupyter notebook and import the code alloy_surface_simulator.py:
```
import alloy_surface_simulator.py
```
Then import the functions and dependencies you wish to work with. For a complete list of all functions to be imported, see initialize.py and simply copy the script over.

## Demo
See the demo.ipynb notebook for example uses of the functions and an interpretation of the output.

## Modifications

To work with metals other than Pt and Au, you must first update the BCN (bulk coordination number) dictionary and the LCN (lattice constant number) dictionary. The bulk coordination number is the number of atoms that typically surrounds a given atom in a crystal. Pt and Au, for instance, are both fcc crystals and have a bulk coordination number of 12. The lattice constants are reported in angstroms. Here is a website to obtain the values: https://periodictable.com/Properties/A/LatticeConstants.html. Note they are in pm and must be converted to angstroms.
As an example, say you want to work with palladium. To update the LCN dictionary you would type:
```
 LCN[‘Pd’] = 3.89
```
To update the BCN dictionary, type:
```
BCN[‘Pd’] = 6
```
