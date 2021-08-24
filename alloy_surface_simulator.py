import ase 
from ase.io import read,write
from ase.data import vdw_radii,chemical_symbols
from ase.atoms import *
from ase.neighborlist import *
from ase import Atoms, build, visualize
from ase.build import fcc100,fcc110, fcc111
from ase.visualize import view

import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import random

def csv_read(filename):
    # reads a .csv file and returns data as a matrix of floats
    dataraw = np.array(list(csv.reader(open(filename, "r"), delimiter=",")))
    data_raw1=np.delete(dataraw,0,0) #delete first row and column
    data_pure=np.delete(data_raw1,0,1).astype("float")
    return data_pure

WF_1=csv_read("WF_refer_1.csv")
WF_2=csv_read("WF_refer_2.csv")

WF_index={"Sc":0,"Ti":1,"V":2,"Cr":3,"Mn":4,"Fe":5,"Co":6,"Ni":7,"Cu":8,"Y":9,"Zr":10,"Nb":11, \
    "Mo":12,"Tc":13,"Ru":14,"Rh":15,"Pd":16,"Ag":17,"La":18,"Hf":19,"Ta":20,"W":21,"Re":22, \
    "Os":23,"Ir":24,"Pt":25,"Au":26,"Th":27, "O":28}

# Dictionary of Bulk Cohesive Energies (BCE)
# Reference: Charles Kittel. Introduction to Solid State Physics, 8th edition. Hoboken, NJ: John Wiley & Sons, Inc, 2005.

BCE={"Sc":-3.90,"Ti":-4.85,"V":-5.31,"Cr":-4.10,"Mn":-2.92,"Fe":-4.28,"Co":-4.39,"Ni":-4.44, \
    "Cu":-3.49,"Y":-4.37,"Zr":-7.1914,"Nb":-7.57,"Mo":-6.82,"Tc":-6.85,"Ru":-6.74 ,"Rh":-5.75, \
    "Pd":-3.89,"Ag":-2.95,"La":-4.47,"Hf":-6.44,"Ta":-8.10,"W":-8.90,"Re":-8.03,"Os":-8.17, \
    "Ir":-6.94,"Pt":-5.84,"Au":-3.81,"Th":-6.20, "O":-2.60}

#Dictionary of bulk coordination numbers (BCN)
BCN={"Pt":12, "Au": 12, "O":2}

#Dictionary of lattice constant numbers (LCN)
LCN = {"Pt":3.92, "Au": 4.07}
    
def calcLC(lc1, lc2, x):
    """
    Returns the lattice constant for unit cells of a bimetallic alloy using a Vegard's Law approximation.
    
    Parameters
    ----------
    lc1: float specifying lattice constant of element 1, in angstroms 
    lc2: float specifying lattice constant of element 2, in angstroms 
    x: integer specifying percent of element 1 in the alloy
    
    """
    b=lc1
    m=lc2-lc1
    y=m*((100-x)/100)+b
    return y

def createSurface (dimensions,MI, lc, e1, per1, e2, filename): 
    """
    Returns a face-centered cubic rectangular bimetallic nanoslab in .xyz file format. 
    Atoms are randomly assigned positions.

    Parameters
    ----------
    dimensions: tuple of 3 integers describing nanoslab dimensions along each Cartesian plane

    MI: Miller indices, entered as an integer. 100, 110, or 111.
    
    lc: integer specifying lattice constant.
    
    e1: string indicating chemical symbol for element 1
    
    per1: float or int indicating the percent composition for first element (e.g. if 43% Pt, per1=43)
    
    e2: string indicating chemical symbol for element 2
    
    filename: string with .xyz extension indicating name of file in which to save data
    """
    
    #generates a slab cut along the desired Miller index (the element in the slab is arbitrarily chosen)
    if MI==111:
        unitCell = fcc111('Pt', size=dimensions, a=lc)
    elif MI==100:
        unitCell = fcc100('Pt', size=dimensions, a=lc)
    elif MI ==110:
        unitCell = fcc110('Pt', size=dimensions, a=lc)
   
    #obtains an array of the positions of each atom in the slab
    positions = unitCell.positions
    
    #generates a list of all the atoms in the slab (by chemical symbol) and shuffles it
    count = unitCell.get_number_of_atoms()
    blank=[]
    if count%2==0:
        for i in range(int((per1/100)*count)):
            blank.append(e1)
        for j in range(int((100-per1)/100*count)):
            blank.append(e2)
        atomlist = blank
        while len(atomlist)<count:
            atomlist.append(e1)
        shuffled = tuple(random.sample(atomlist, count))
    else:
        for i in range(int((per1/100)*count)+1):
                blank.append(e1)
        for j in range(int((100-per1)/100*count)):
            blank.append(e2)
        atomlist = blank
        while len(atomlist)<count:
            atomlist.append(e1)
        shuffled = tuple(random.sample(atomlist, count))
    
    #writes an .xyz file in which the atoms in the shuffled list are assigned positions in the slab
    f=open(filename,mode='w+', newline='\n')
    f.write(str(count)+'\n\n')
    for atoms, pos in zip(shuffled, positions):
        f.write("{} {} {} {} \n".format(atoms, pos[0], pos[1], pos[2]))
    f.close()

def cns(neiglist, filename):
    """
    code, updated for Python 3, borrowed from
    Yan, Z.; Taylor, M. G.; Mascareno, A.; Mpourmpakis, G. 
    "Size-, Shape-, and Composition-Dependent Model for Metal Nanoparticle Stability Prediction" 
    Nano Lett. 2018, 18 (4), 2696–2704.
    
    For use within findCNS().
    """
    cns=[]
    complete_nl=[] #include neighborlist of all atoms
    atoms1=read(filename)
    for i in range(len(atoms1)):
        neighs=neiglist.get_neighbors(i)
        neighs=neighs[0].tolist()
        neighs=[x for x in neighs if x != i]
        #print neighs
        cns.append(len(neighs))
        complete_nl.append(neighs)
        #print final_nl
    return np.array(cns), np.array(complete_nl) #turn into an array

def findCNS(filename):
    """
    code, updated for Python 3, borrowed from
    Yan, Z.; Taylor, M. G.; Mascareno, A.; Mpourmpakis, G. 
    "Size-, Shape-, and Composition-Dependent Model for Metal Nanoparticle Stability Prediction" 
    Nano Lett. 2018, 18 (4), 2696–2704.
    """
    #takes as input a file with .xyz extension; modifies the file to include the coordination number and a list with numbers of all neighbors for each atom
    count_0=0
    #list of vdw radii for all the elements in order of atomic number, with 0 placeholders for element 0 and all nonmetals
    newvdw=[ 0.  ,  0.  ,  0.  ,  2.14,  1.69,  1.68,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  ,  2.38,  2.  ,  1.92,  1.93,  0.  ,  0.  ,  0.  ,
            0.  ,  2.52,  2.27,  2.15,  2.11,  2.07,  2.06,  2.05,  2.04,
            2.  ,  1.97,  1.96,  2.01,  2.03,  2.05,  2.08,  0.  ,  0.  ,
            0.  ,  2.61,  2.42,  2.32,  2.23,  2.18,  2.17,  2.16,  2.13,
            2.1 ,  2.1 ,  2.11,  2.18,  2.21,  2.23,  2.24,  0.  ,  0.  ,
            0.  ,  2.75,  2.59,  2.43,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
            2.23,  2.22,  2.18,  2.16,  2.16,  2.13,  2.13,  2.14,  2.23,
            2.27,  2.37,  2.38,  2.49,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
            2.45,  0.  ,  2.41,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ]
    

    # step_size parameter-determines how much the radii will be shifted when CN exceptions occur
    step_size=-0.001
    # difference between changes in scales for more frequent element that has higher CN than 12
    difference=0.3
    
    
    #read in .xyz file, determine number and type of atoms, set scaling
    outflag=0
    nl=None
    moleculename=filename
    atoms1=read(moleculename)
    atomic_numbers=atoms1.get_atomic_numbers()
    num_atoms=len(atomic_numbers)
    max_more_num=int(num_atoms**(1./3)) #max num of atoms allowed with CN>12
        # min and max anums determine the identity of the two metals
    min_anum=min(atomic_numbers)
    max_anum=max(atomic_numbers)
    # The following line are calculating ratio of vdw_radii of elements
    vdw_radi_1=newvdw[min_anum]
    vdw_radi_2=newvdw[max_anum]
        # Decide "larger" atom by comparison of vdw_radii of elements
    if vdw_radi_1>vdw_radi_2:
        large_atom=min_anum
        small_atom=max_anum
    else:
        large_atom=max_anum
        small_atom=min_anum
    min_vdw_radi=min(vdw_radi_1,vdw_radi_2)
    max_vdw_radi=max(vdw_radi_1,vdw_radi_2)
    ratio_vdw_r=max_vdw_radi/min_vdw_radi
        # Determine if pure metal MNP - set initial scales to 0.875 
        # scale_1 assigned to the smaller element in bimetallic MNP
        # scale_2 assigned to the larger element in bimetallic MNP
    if ratio_vdw_r==1: 
        scale_1=0.875
        scale_2=0.875
    else:
        scale_1=1
        scale_2=1
    
    while outflag==0: # while used to continue loop until outflag is true
        cutoffs=[]
            # Generate cutoffs (distance for which an atom is considered neighboring) by atomic_numbers
        for i in range(len(atomic_numbers)):
            temp=atomic_numbers[i]
            if temp==min_anum:
                cutoffs.append(scale_1*newvdw[temp])
            else: 
                cutoffs.append(scale_2*newvdw[temp])
            # update neighborlist and calculate new cns
        nl=NeighborList(cutoffs,bothways=True,skin=0.0)
        nl.update(atoms1)
        cordns,final_nl=cns(nl, filename) #final_nl = complete_nl
                    # countmin/max are counters for the number of atoms which have cns>12
        count_small=0
        count_large=0
        less=0
        more=0
        too_much=0 #atom with CN>13
        bulk_large=0
        bulk_small=0
        for i in range(len(cordns)):
                        # Check if the cn of atom i is greater than 12 or less than 3
            if cordns[i]<3:
                less+=1
            if cordns[i]>13: 
                too_much+=1
            if cordns[i]>12:
                            # Check element of atom i that has deviated from normal CN range
                more+=1
                if atomic_numbers[i]==small_atom:
                    count_small+=1
                else: count_large+=1
            if cordns[i]>=12 and atomic_numbers[i]==large_atom:
                bulk_large+=1 #looking for large atoms in the bulk
            if cordns[i]>=12 and atomic_numbers[i]==small_atom:
                bulk_small+=1
                    # check if any of the counters were incremented (i.e. if any atoms were CN>12 or CN<3)
        if ratio_vdw_r >=1.1: #when ratio>=1.1, the structure is "amporphous":	
            if bulk_small==0:#when all small atoms are not in bulk
                if (count_large+count_small)>max_more_num or too_much>0:
                    scale_2+=step_size
                    scale_1+=step_size*difference
                else:
                    outflag=1
            elif bulk_large==0: # No large atoms in the bulk (all large atoms on surface)
                if (count_large+count_small)>0: # No overcutting will happen in this case, to cut down to 0
                    scale_2+=step_size
                    scale_1+=step_size*difference
                else:
                    outflag=1
            elif (count_large+count_small)>max_more_num or too_much>0:
                # If any are, increment their scale by the the step_size, favor "larger" element 
                if count_large>=count_small:
                    scale_2+=step_size
                    scale_1+=step_size*difference
                else: 
                    scale_1+=step_size
                    scale_2+=step_size*difference
            else: outflag=1
                    # set outflag=1 to exit while loop!
        elif ratio_vdw_r==1.0: # Only true if this is a pure metal MNP, reduce scale factor
            if (count_large+count_small)>0:
                scale_2+=step_size
                scale_1=scale_2
            else: outflag=1
        else: #Structure is not over distorted
            if (count_large+count_small)>0:
                    # if any are, increment their scale by the the step_size, favor "larger" element 
                if count_large>=count_small:
                    scale_2+=step_size
                    scale_1+=step_size*difference
                else:
                    scale_1+=step_size
                    scale_2+=step_size*difference
            else: outflag=1
    count=0
    file1=open(moleculename,'r')
    lines=file1.readlines()
    relevantLines = lines[2:]
    file1.close()
    file2=open(moleculename,'w+', newline='\n')
    file2.write(str(num_atoms)+'\n\n')
    for line in relevantLines:
        vect=line.split()
        a=float(cordns[count])
        b=final_nl[count]
        vect.append('{}'.format(a))
        vect.append('{}\n'.format(b))
        out=' '.join(vect)
        file2.write(out)
        count+=1
    file2.close()
    
def readability(filename):
    """
    Takes as input an .xyz file and returns a list with indices from 0-4 corresponding to:
        -atom numbers,
        -chemical symbols,
        -position coordinates, 
        -CNs, and
        -neighbor lists.
    
    Used to convert .xyz files into formats useful for parameterLookup and energyCalculator functions in an MC simulation.
    """
    
    #opens the file and loads the lines into memory
    f=open(filename, 'r')
    data=f.readlines()
    atoms=[]
    positions=[]
    CNs=[]
    neighbors=[]
    
    #for each line, appends the chemical symbol, position coordinates, CN, and neighborlist of the atom to a list
    for line in data[2:]:
        vals=line.split()
        atoms.append(vals[0])
        positions.append(vals[1:4])
        CNs.append(vals[4])
        vect=line.split("[") 
        nl_i_raw=vect[1][0:-2] 
        nl_i_raw_2=nl_i_raw.split(",") 
        nl_i = [int(i) for i in nl_i_raw_2]
        neighbors.append(nl_i)
    
    #returns a list of lists with all relevant data
    config=[]
    for i in range(len(data[2:])):
        config.append([i, atoms[i], positions[i], CNs[i], neighbors[i]])
    return config

def parameterLookup(config, WF_1, WF_2, WF_index, BCE, adsorb=None):
    '''
    For a bimetallic alloy nanoslab, returns, as lists indexed as 0-3 respectively:
        -the atom types (i.e. chemical symbols of elements that show up in the file),
        -binding energies,
        -bulk coordination numbers, 
        -and weight factors.
    
    These lists are used to compute the cohesive energy of the configuration throughout an MC simulation
    (speeds up calculation time).
    
    Parameters
    ------------
    config: a nanoslab configuration obtained by applying readability() to an .xyz file
    
    adsorb: string indicating chemical symbol of adsorbate, if applicable. Default is None (for a nanoslab in vacuum)
    '''
    
    #determines identities of elements in the alloy configuration. stored as a list.
    atoms=[]
    for line in config:
        atoms.append(line[1])
    atom_types = list(dict.fromkeys(atoms))
    
    if adsorb in WF_index: #separates cases of systems with adsorbates and systems without, while also ensuring that the adsorbate is a valid input
        #for each atom in alloy and for adsorbate, determines reference number used to pull BDE data from WF csv file
        refnum_e1=WF_index[atom_types[0]]
        refnum_e2=WF_index[atom_types[1]]
        refnum_adsorb=WF_index[adsorb]

        #for each atom in alloy and for adsorbate, determines bulk cohesive energy and bulk coordination number. each is stored as a list.
        BCE_e1=BCE[atom_types[0]]
        BCE_e2=BCE[atom_types[1]]
        BCE_adsorb=BCE[adsorb]
        
        BCN_e1=BCN[atom_types[0]]
        BCN_e2=BCN[atom_types[1]]
        BCN_adsorb=BCN[adsorb]

        #use reference numbers to look up BDEs for each pair of atoms
        if WF_1[refnum_e1][refnum_e2]!=0: #reference 1 is in priority since it's experimental value
            ij= WF_1[refnum_e1][refnum_e2]
            ii= WF_1[refnum_e1][refnum_e1]
            jj= WF_1[refnum_e2][refnum_e2]
            ia= WF_1[refnum_e1][refnum_adsorb]
            ja= WF_1[refnum_e2][refnum_adsorb]
            aa= WF_1[refnum_adsorb][refnum_adsorb]
        else: #Switch to reference 2 if reference 1 doesn't have the data
            ij= WF_2[refnum_e1][refnum_e2]
            ii= WF_2[refnum_e1][refnum_e1]
            jj= WF_2[refnum_e2][refnum_e2]
            ia= WF_2[refnum_e1][refnum_adsorb]
            ja= WF_2[refnum_e2][refnum_adsorb]
            aa= WF_2[refnum_adsorb][refnum_adsorb]
        
        #computes weight factors using BDEs
        weightfactor_i_withj=(ij-jj)/(ii-jj)
        weightfactor_j_withi=1-weightfactor_i_withj
        weightfactor_i_witha= (ia-aa)/(ii-aa) 
        weightfactor_a_withi= 1- weightfactor_i_witha
        weightfactor_j_witha=(ja-aa)/(jj-aa)
        weightfactor_a_withj= 1-weightfactor_j_witha
        
        return [atom_types[0], atom_types[1], adsorb],[BCE_e1, BCE_e2, BCE_adsorb], [BCN_e1, BCN_e2, BCN_adsorb], [weightfactor_i_withj, weightfactor_j_withi, weightfactor_i_witha, weightfactor_a_withi, weightfactor_j_witha, weightfactor_a_withj]
    #if no valid adsorbate is specified, code functions as above except is only considering and determining values for the two elements in the alloy
    else:
        refnum_e1=WF_index[atom_types[0]]
        refnum_e2=WF_index[atom_types[1]]

        BCE_e1=BCE[atom_types[0]]
        BCE_e2=BCE[atom_types[1]]
        
        BCN_e1=BCN[atom_types[0]]
        BCN_e2=BCN[atom_types[1]]

        if WF_1[refnum_e1][refnum_e2]!=0: #reference 1 is in priority since it's experimental value
            ij= WF_1[refnum_e1][refnum_e2]
            ii= WF_1[refnum_e1][refnum_e1]
            jj= WF_1[refnum_e2][refnum_e2]
        else: #Switch to reference 2 if reference 1 doesn't have the data
            ij= WF_2[refnum_e1][refnum_e2]
            ii= WF_2[refnum_e1][refnum_e1]
            jj= WF_2[refnum_e2][refnum_e2]

        weightfactor_i_withj=(ij-jj)/(ii-jj)
        weightfactor_j_withi=1-weightfactor_i_withj
        
        return [atom_types[0], atom_types[1]], [BCE_e1, BCE_e2], [BCN_e1, BCN_e2], [weightfactor_i_withj, weightfactor_j_withi]

def energyCalculator(config, atom_types,BCEs, BCNs, weightfactors, cover):
    '''
    Returns the cohesive energy per atom, in eV, of a nanoslab configuration.
    Also returns list of pairs of atoms, to speed up subsequent calculations of energy in an MC simulation.
    
    Parameters
    ------------
    config: a nanoslab configuration obtained by applying readability() to an .xyz file
    
    atom_types, BCEs, BCNs, weightfactors: output of the parameterLookup() function applied to the configuration
    
    cover: float between 0 and 1 indicating fraction of undercoordinated surface sites to be covered with adsorbates
    '''
    
    BE=0
    pairs=[]

    
    if len(atom_types)==2: #binary alloy without adsorbate
        
        #creates a list of all pairs of atoms in the configuration
        for num, line in enumerate(config):
            for k in line[4]:
                pairs.append([num, k])
        
        #for each pair, obtains CN, BCE, BCN, and weight factor of each atom. computes the pair's contribution to energy and adds this value to the total energy
        for [i, j] in pairs:
            
            CN_i=float(config[i][3])
            CN_j=float(config[j][3])

            if config[i][1]==atom_types[0]:
                BCE_i=BCEs[0]
                BCN_i=BCNs[0]
            elif config[i][1] ==atom_types[1]:
                BCE_i=BCEs[1]
                BCN_i=BCNs[1]
            else:
                print('error')

            if config[j][1]==atom_types[0]:
                BCE_j=BCEs[0]
                BCN_j=BCNs[0]
            elif config[j][1]==atom_types[1]:
                BCE_j=BCEs[1]
                BCN_j=BCNs[1]
            else:
                print('error')

            if config[i][1]==config[j][1]:
                weightfactor_i=0.5
                weightfactor_j=0.5
            else:
                if config[i][1]==atom_types[0]:
                    weightfactor_i=weightfactors[0]
                    weightfactor_j=weightfactors[1]
                else:
                    weightfactor_i=weightfactors[1]
                    weightfactor_j=weightfactors[0]

            BE_perpair = weightfactor_i*(BCE_i/CN_i)*(CN_i/BCN_i)**0.5 + weightfactor_j*(BCE_j/CN_j)*(CN_j/BCN_j)**0.5
            BE+=BE_perpair
        
        return BE, pairs

    elif len(atom_types)==3: #binary alloy with adsorbate
        
        #creates a list of all pairs of atoms in the configuration
        
        new_CNs=[]
        
        for num, line in enumerate(config):
            for k in line[4]:
                pairs.append([num, k])
            
            if line[1]==atom_types[0]:
                BCN_constraint=BCNs[0]
            elif line[1]==atom_types[1]:
                BCN_constraint=BCNs[1]
            
            if float(line[3])<BCN_constraint:
                coord_sites=int(cover*(BCN_constraint-float(line[3])))
                new_CNs.append(float(line[3])+coord_sites)
                for i in range(coord_sites):
                    pairs.append([num, atom_types[2]]) #for all undercoordinated atoms, 
                    #if "cover"=1 (i.e. entirely coated in oxygen) append [atom#, adosorbateSymbol] pair as many times as it takes to attain a full coordination shell
                     #if "cover" is a fraction append [atom#, adosorbateSymbol] pair as many times as it takes to achieve the desired coordination fraction
            else: 
                new_CNs.append(float(line[3]))
                
        #for each pair, obtains BCE, BCN, and weight factor of each atom. computes the pair's contribution to energy and adds this value to the total energy
        for [i, j] in pairs:

            if config[i][1]==atom_types[0]:
                BCE_i=BCEs[0]
                BCN_i=BCNs[0]
            elif config[i][1] ==atom_types[1]:
                BCE_i=BCEs[1]
                BCN_i=BCNs[1]
            else:
                print('error')

            if type(j)==str: #pairs of an element with an adsorbate
                
                CN_i=float(new_CNs[i])
                
                BCE_j=BCEs[2]
                BCN_j=BCNs[2]
                if config[i][1]==atom_types[0]:
                    weightfactor_i=weightfactors[2]
                    weightfactor_j=weightfactors[3]
                elif config[i][1] ==atom_types[1]:
                    weightfactor_i=weightfactors[4]
                    weightfactor_j=weightfactors[5]
                
                BE_perpair = weightfactor_i*(BCE_i/CN_i)*(CN_i/BCN_i)**0.5 + weightfactor_j*(BCE_j/BCN_j)*(1/BCN_j)**0.5
                BE+=BE_perpair
                
            else: #pairs of an element with another element
                CN_i=float(new_CNs[i])
                CN_j=float(new_CNs[j])
                
                if config[j][1]==atom_types[0]:
                    BCE_j=BCEs[0]
                    BCN_j=BCNs[0]
                elif config[j][1]==atom_types[1]:
                    BCE_j=BCEs[1]
                    BCN_j=BCNs[1]
                else:
                    print('error')

                if config[i][1]==config[j][1]:
                    weightfactor_i=0.5
                    weightfactor_j=0.5
                else:
                    if config[i][1]==atom_types[0]:
                        weightfactor_i=weightfactors[0]
                        weightfactor_j=weightfactors[1]
                    elif config[i][1]==atom_types[1]:
                        weightfactor_i=weightfactors[1]
                        weightfactor_j=weightfactors[0]

                BE_perpair = weightfactor_i*(BCE_i/CN_i)*(CN_i/BCN_i)**0.5 + weightfactor_j*(BCE_j/CN_j)*(CN_j/BCN_j)**0.5

                BE+=BE_perpair
        
        return BE, pairs, new_CNs
                               
        
    else:
        print("Error-incorrect data input type.")
        

def energyCalculator_FAST(initialData, original, final, swap1, swap2, atom_types, BCEs, BCNs, weightfactors, cover):
    
    '''
    Used within a PT MC simulation.
    Calculates energy of a "final" configuration that differs from the original by 2 lines which were swapped. 
    
    Parameters
    -----------
    initialData: output of energyCalculator() function applied to the original configuration
    
    original: starting nanoslab configuration obtained by applying readability() to the .xyz file
    
    final: new configuration upon swapping atoms
    
    swap1, swap2: integers denoting lines of the atoms that were swapped (obtained from atomSwapper function)
    
    atom_types, BCEs, BCNs, weightfactors: output of the parameterLookup() function applied to the configuration
    
    cover: float between 0 and 1 indicating fraction of undercoordinated surface sites to be covered with adsorbates
    
      
    '''
    originalEnergy=initialData[0] 
    pairs=initialData[1]
    
    BE=originalEnergy
    
    if len(atom_types)==2: #binary alloy without adsorbates
        
        for [i, j] in pairs:
            #add in lines that are different
            if i==swap1 or j==swap1 or i==swap2 or j==swap2: 
                CN_i=float(final[i][3])
                CN_j=float(final[j][3])

                if final[i][1]==atom_types[0]:
                    BCE_i=BCEs[0]
                elif final[i][1] ==atom_types[1]:
                    BCE_i=BCEs[1]
                else:
                    print('error')

                if final[j][1]==atom_types[0]:
                    BCE_j=BCEs[0]
                elif final[j][1]==atom_types[1]:
                    BCE_j=BCEs[1]
                else:
                    print('error')

                if final[i][1]==final[j][1]:
                    weightfactor_i=0.5
                    weightfactor_j=0.5
                else:
                    if final[i][1]==atom_types[0]:
                        weightfactor_i=weightfactors[0]
                        weightfactor_j=weightfactors[1]
                    else:
                        weightfactor_i=weightfactors[1]
                        weightfactor_j=weightfactors[0]

                BE_perpair = weightfactor_i*(BCE_i/CN_i)*(CN_i/12)**0.5 + weightfactor_j*(BCE_j/CN_j)*(CN_j/12)**0.5

                BE+=BE_perpair

        for [i, j] in pairs:
            #subtract lines that are no longer in the file
            if i==swap1 or j==swap1 or i==swap2 or j==swap2: 
                CN_i=float(original[i][3])
                CN_j=float(original[j][3])

                if original[i][1]==atom_types[0]:
                    BCE_i=BCEs[0]
                elif original[i][1] ==atom_types[1]:
                    BCE_i=BCEs[1]
                else:
                    print('error')

                if original[j][1]==atom_types[0]:
                    BCE_j=BCEs[0]
                elif original[j][1]==atom_types[1]:
                    BCE_j=BCEs[1]
                else:
                    print('error')

                if original[i][1]==original[j][1]:
                    weightfactor_i=0.5
                    weightfactor_j=0.5
                else:
                    if original[i][1]==atom_types[0]:
                        weightfactor_i=weightfactors[0]
                        weightfactor_j=weightfactors[1]
                    else:
                        weightfactor_i=weightfactors[1]
                        weightfactor_j=weightfactors[0]

                BE_perpair = weightfactor_i*(BCE_i/CN_i)*(CN_i/12)**0.5 + weightfactor_j*(BCE_j/CN_j)*(CN_j/12)**0.5

                BE-=BE_perpair
        
    
    elif len(atom_types)==3: #binary alloy with adsorbates
        
        new_CNs=initialData[2]
        
        for [i, j] in pairs:
            #add in lines that are different
            if i==swap1 or j==swap1 or i==swap2 or j==swap2: 
        
                if final[i][1]==atom_types[0]:
                    BCE_i=BCEs[0]
                    BCN_i=BCNs[0]
                elif final[i][1] ==atom_types[1]:
                    BCE_i=BCEs[1]
                    BCN_i=BCNs[1]
                else:
                    print('error')

                if type(j)==str:
                    CN_i=float(final[i][3])
                    if CN_i<12:
                        coord_sites=int(cover*(12-CN_i))
                        CN_i=CN_i+coord_sites
                    
                    BCE_j=BCEs[2]
                    BCN_j=BCNs[2]
                    if final[i][1]==atom_types[0]:
                        weightfactor_i=weightfactors[2]
                        weightfactor_j=weightfactors[3]
                    elif final[i][1]==atom_types[1]:
                        weightfactor_i=weightfactors[4]
                        weightfactor_j=weightfactors[5]
                        
                    BE_perpair = weightfactor_i*(BCE_i/CN_i)*(CN_i/12)**0.5 + weightfactor_j*(BCE_j/BCN_j)*(1/BCN_j)**0.5
                
                    BE+=BE_perpair
                
                else:
                    CN_i=float(final[i][3])
                    CN_j=float(final[j][3])
                    if CN_i<12:
                        coord_sites=int(cover*(12-CN_i))
                        CN_i=CN_i+coord_sites
                    if CN_j<12:
                        coord_sites=int(cover*(12-CN_j))
                        CN_j=CN_j+coord_sites
                
                    if final[j][1]==atom_types[0]:
                        BCE_j=BCEs[0]
                        BCN_j=BCNs[0]
                    elif final[j][1]==atom_types[1]:
                        BCE_j=BCEs[1]
                        BCN_j=BCNs[1]
                    else:
                        print('error')

                    if final[i][1]==final[j][1]:
                        weightfactor_i=0.5
                        weightfactor_j=0.5
                    else:
                        if final[i][1]==atom_types[0]:
                            weightfactor_i=weightfactors[0]
                            weightfactor_j=weightfactors[1]
                        else:
                            weightfactor_i=weightfactors[1]
                            weightfactor_j=weightfactors[0]

                    BE_perpair = weightfactor_i*(BCE_i/CN_i)*(CN_i/12)**0.5 + weightfactor_j*(BCE_j/CN_j)*(CN_j/12)**0.5

                    BE+=BE_perpair

        for [i, j] in pairs:
            #subtract lines that are no longer in the file
            if i==swap1 or j==swap1 or i==swap2 or j==swap2: 

                if original[i][1]==atom_types[0]:
                    BCE_i=BCEs[0]
                    BCN_i=BCNs[0]
                elif original[i][1] ==atom_types[1]:
                    BCE_i=BCEs[1]
                    BCN_i=BCNs[1]
                else:
                    print('error')

                if type(j)==str:
                    CN_i=float(original[i][3])
                   
                    if CN_i<12:
                        coord_sites=int(cover*(12-CN_i))
                        CN_i=CN_i+coord_sites
                    
                    BCE_j=BCEs[2]
                    BCN_j=BCNs[2]
                    if original[i][1]==atom_types[0]:
                        weightfactor_i=weightfactors[2]
                        weightfactor_j=weightfactors[3]
                    elif original[i][1]==atom_types[1]:
                        weightfactor_i=weightfactors[4]
                        weightfactor_j=weightfactors[5]
                        
                    BE_perpair = weightfactor_i*(BCE_i/CN_i)*(CN_i/12)**0.5+ weightfactor_j*(BCE_j/BCN_j)*(1/BCN_j)**0.5
                
                    BE-=BE_perpair
                
                else:
                    CN_i=float(original[i][3])
                    CN_j=float(original[j][3])
                    if CN_i<12:
                        coord_sites=int(cover*(12-CN_i))
                        CN_i=CN_i+coord_sites
                    if CN_j<12:
                        coord_sites=int(cover*(12-CN_j))
                        CN_j=CN_j+coord_sites
                
                    if original[j][1]==atom_types[0]:
                        BCE_j=BCEs[0]
                        BCN_j=BCNs[0]
                    elif original[j][1]==atom_types[1]:
                        BCE_j=BCEs[1]
                        BCN_j=BCNs[1]
                    else:
                        print('error')

                    if original[i][1]==original[j][1]:
                        weightfactor_i=0.5
                        weightfactor_j=0.5
                    else:
                        if original[i][1]==atom_types[0]:
                            weightfactor_i=weightfactors[0]
                            weightfactor_j=weightfactors[1]
                        else:
                            weightfactor_i=weightfactors[1]
                            weightfactor_j=weightfactors[0]

                    BE_perpair = weightfactor_i*(BCE_i/CN_i)*(CN_i/12)**0.5 + weightfactor_j*(BCE_j/CN_j)*(CN_j/12)**0.5

                    BE-=BE_perpair
    
    #return final energy
    return BE


def atomSwapper(config):
    '''
    Given an input nanoslab configuration, swaps the positions of two atoms of different elements, chosen randomly.
    Returns the new configuration and the line numbers of the swapped rows.
    '''
    
    #generates two random numbers within the range of the number of atoms in the configuration
        #these will be the swapped rows
    numAtoms=len(config)
    swap1= random.randint(0,numAtoms-1)
    swap2= random.randint(0,numAtoms-1)
    
    #eliminate possibility that an atom is switching with itself or that two atoms of the same element are swapping
    while swap1==swap2 or config[swap1][1]==config[swap2][1]:
        swap2= random.randint(0,numAtoms-1)
    
    #generates a new configuration that includes the swapped rows
    newConfig=[]
    for line in config:
        if line[0]!=swap1 and line[0]!=swap2:
            newConfig.append(line)
        if line[0]==swap1:
            newConfig.append([line[0],config[swap2][1], line[2], line[3], line[4]])
        if line[0]==swap2:
            newConfig.append([line[0],config[swap1][1], line[2], line[3], line[4]])
    return newConfig, swap1, swap2


def PT_step(config, kT, atom_types,BCEs, BCNs, weightfactors, cover):
    '''
    Runs one step (trial) of a PT-MC simulation and returns the configuration and CE at the end of the step.
    
    Parameters
    ----------
    config: starting nanoslab configuration obtained by applying readability() to the .xyz file
    
    kT: float indicating temperature in eV multiplied by the Boltzmann constant (at room temp kT = 0.025 eV)
    
    atom_types, BCEs, BCNs, weightfactors: output of the parameterLookup() function applied to the configuration
    
    cover: float between 0 and 1 indicating fraction of undercoordinated surface sites to be covered with adsorbates
    
    '''
    
    #swap two atoms
    run = atomSwapper(config)
    oldConfig = config
    newConfig = run[0]
    swap1=run[1]
    swap2=run[2]
    #calculate energies of original and new config, as well as energy difference
    initialData=energyCalculator(oldConfig, atom_types,BCEs, BCNs, weightfactors, cover)
    initialEnergy=initialData[0]
    newEnergy=energyCalculator_FAST(initialData, oldConfig, newConfig, swap1, swap2, atom_types, BCEs, BCNs, weightfactors, cover) 
    
    #determine whether or not to accept the change. if accepted, return new configuration and its energy. if rejected, return old configuration and its energy
    if newEnergy<initialEnergy:
        return newConfig, newEnergy
    if newEnergy==initialEnergy:
        return newConfig, newEnergy
    if newEnergy>initialEnergy:
        if random.uniform(0, 1)<= math.exp(-(newEnergy-initialEnergy)/kT): 
            return newConfig, newEnergy    
        else:
            return oldConfig, initialEnergy


def PT_simulation(filename, trials, frac_swap, kT_cold, kT_med, kT_hot, atom_types,BCEs, BCNs, weightfactors, cover):
    '''
    Runs a PT MC simulation and returns the cohesive energies per atom (for all three reservoirs), pair site
    statistics, and surface distribution statistics for each configuration.
        Each entry in the returned pair site list is a list, [e1-e1 pairs on surface, e2-e2 pairs on surface]
        Each entry in the returned distribution list is a list, [# of e1 atoms on surface, # of e2 atoms on surface]
    
    Also returns a list (length=2) of the elements in the configuration, 
    to be able to determine which is e1 and which is e2, for data processing
    
    Parameters
    ----------
    filename: string with .xyz extension corresponding to starting configuration
    
    trials: number of trials in the simulation 
    
    frac_swap: float between 0 and 1 corresponding to % of trials for which config from the medium-temp simulation 
    is swapped with that from the cold-temp, and for which config from hot-temp simulation is swapped with that from 
    the medium-temp (literature recommends using 0.1)
    
    kT_cold: float indicating temperature in eV multiplied by the Boltzmann constant for low-temp reservoir
    
    kT_med: float indicating temperature in eV multiplied by the Boltzmann constant for med-temp reservoir
    
    kT_hot: float indicating temperature in eV multiplied by the Boltzmann constant for high-temp reservoir
    '''
    
    #read in file and initialize variables
    original = readability(filename)
    e1 = atom_types[0]
    e2 = atom_types[1]
    
    energies_cold=[]
    energies_med=[]
    energies_hot=[]
    
    configs_cold=[original]
    configs_med=[original]
    configs_hot=[original]
    
    pairStats = []
    distributionStats = []
    
    for i in range(trials): 
        #for each temperature, apply PT step to each configuration and append the resulting configuration, energy, and stats to their respective lists
        run_cold = PT_step(configs_cold[i], kT_cold, atom_types,BCEs, BCNs, weightfactors, cover)
        run_med = PT_step(configs_med[i], kT_med, atom_types,BCEs, BCNs, weightfactors, cover)
        run_hot = PT_step(configs_hot[i], kT_hot, atom_types,BCEs, BCNs, weightfactors, cover)
        config_cold = run_cold[0]
        config_med = run_med[0]
        config_hot = run_hot[0]
        energy_cold = run_cold[1]/len(original)
        energy_med = run_med[1]/len(original)
        energy_hot = run_hot[1]/len(original)
 
        #for some randomly chosen moves, swap the reservoirs and accept changes with Bolztmann probability
        if random.random() < frac_swap: 
            if energy_med<=energy_cold:
                configs_cold.append(run_med[0])
                energies_cold.append(run_med[1]/len(original))
                pairStat = pairSiteCalculator(run_med[0], e1, e2)
                pairStats.append([pairStat[0], pairStat[1]])
                distributionStat = distributionCalculator(run_med[0], e1, e2)
                distributionStats.append([distributionStat[2], distributionStat[3]])
            
            elif energy_med>energy_cold:
                if random.uniform(0, 1)<= math.exp(-(energy_med-energy_cold)/kT_cold):
                    configs_cold.append(run_med[0])
                    energies_cold.append(run_med[1]/len(original))
                    pairStat = pairSiteCalculator(run_med[0], e1, e2)
                    pairStats.append([pairStat[0], pairStat[1]])
                    distributionStat = distributionCalculator(run_med[0], e1, e2)
                    distributionStats.append([distributionStat[2], distributionStat[3]])
                else:
                    configs_cold.append(run_cold[0])
                    energies_cold.append(run_cold[1]/len(original))
                    pairStat = pairSiteCalculator(run_cold[0], e1, e2)
                    pairStats.append([pairStat[0], pairStat[1]])
                    distributionStat = distributionCalculator(run_cold[0], e1, e2)
                    distributionStats.append([distributionStat[2], distributionStat[3]])
            #above, nothing was appended to the intermediate(med) reservoir because a config is appended below from hot/med swap. appending during cold/med swap as well would double the number of configurations in the intermediate list, which would incorrectly affect indexing of configurations 
            
            if energy_hot<=energy_med:
                configs_med.append(run_hot[0])
                configs_hot.append(run_med[0])
                energies_med.append(run_hot[1]/len(original))
                energies_hot.append(run_med[1]/len(original))
                
            elif energy_hot>energy_med:
                if random.uniform(0, 1)<= math.exp(-(energy_med-energy_cold)/kT_cold):
                    configs_med.append(run_hot[0])
                    configs_hot.append(run_med[0])
                    energies_med.append(run_hot[1]/len(original))
                    energies_hot.append(run_med[1]/len(original))
                else:
                    configs_med.append(run_med[0])
                    configs_hot.append(run_hot[0])
                    energies_med.append(run_med[1]/len(original))
                    energies_hot.append(run_hot[1]/len(original))
        
        #for all other moves, perform an MC simulation as normal       
        else:
            configs_cold.append(run_cold[0])
            configs_med.append(run_med[0])
            configs_hot.append(run_hot[0])
            energies_cold.append(run_cold[1]/len(original))
            energies_med.append(run_med[1]/len(original))
            energies_hot.append(run_hot[1]/len(original))
            pairStat = pairSiteCalculator(run_cold[0], e1, e2)
            pairStats.append([pairStat[0], pairStat[1]])
            distributionStat = distributionCalculator(run_cold[0], e1, e2)
            distributionStats.append([distributionStat[2], distributionStat[3]])
    
      
    return [e1, e2], energies_cold, energies_med, energies_hot, pairStats, distributionStats

def pairSiteCalculator(config, e1, e2):
    
    '''
    Given a binary alloy configuration (obtained from applying readability function to an .xyz file) 
    with elements e1 and e2, counts number of surface pairs for element e1 and for element e2.
    
    Criteria for a surface pair is that both atoms are on the surface (CN<12).
    '''
    
    pairs_e1_surf=[]
    pairs_dict1={}
    
    pairs_e2_surf=[]
    pairs_dict2={}
    
    for num, line in enumerate(config):
        if line[1]==e1:#element 1
            if line[3]!='12.0': #on the surface
                for k in line[4]:
                    if config[k][1]==e1:
                        if config[k][3]!='12.0':
                            pairs_e1_surf.append([num, k])
                            
        elif line[1]==e2:#element 2
            if line[3]!='12.0': #on the surface
                for k in line[4]:
                    if config[k][1]==e2:
                        if config[k][3]!='12.0':
                            pairs_e2_surf.append([num, k])
                            
    for [i,j] in pairs_e1_surf:
        pairs_dict1.update({i:j})
        
    for [k,l] in pairs_e2_surf:
        pairs_dict2.update({k:l})
    
    return len(pairs_dict1), len(pairs_dict2)
 

def distributionCalculator(config, e1, e2):
    
    '''
    Given a binary alloy configuration (obtained from applying readability function to an .xyz file) 
    with elements e1 and e2, counts number of total atoms in bulk and on surface 
    as well as how many of the bulk atoms and how many of the surface atoms are each type of element.
    '''
    
    num_atoms=len(config)
    
    bulk=0
    surface=0
    e1surface=0
    e2surface=0
    e1bulk=0
    e2bulk=0

    for line in config:
        if line[3]=='12.0':
            bulk+=1
            if line[1]==e1:
                e1bulk+=1
            elif line[1]==e2:
                e2bulk+=1
        else:
            surface+=1
            if line[1]==e1:
                e1surface+=1
            elif line[1]==e2:
                e2surface+=1
    
    return bulk, surface, e1surface, e2surface, e1bulk, e2bulk

def cnCalculator(config, e1, e2):
    '''
    Given a binary alloy configuration (obtained from applying readability function to an .xyz file) 
    with elements e1 and e2, calculates average coordination number overall and of surface only, 
    for all atoms, for atoms of element e1, and for atoms of element e2.
    '''
    
    CNs=[]
    surfaceCNs=[]
    e1CNs=[]
    e2CNs=[]
    e1surfaceCNs=[]
    e2surfaceCNs=[]

    for line in config:
        CNs.append(float(line[3]))
        if line[1]==e1:
            e1CNs.append(float(line[3]))
        if line[1]==e2:
            e2CNs.append(float(line[3]))
        if line[3]!='12.0':
            surfaceCNs.append(float(line[3]))
            if line[1]==e1:
                e1surfaceCNs.append(float(line[3]))
            elif line[1]==e2:
                e2surfaceCNs.append(float(line[3]))
    
    avgCN = (np.array(CNs)).mean()
    avgSurfCN=(np.array(surfaceCNs)).mean()
    avgCN_e1 = (np.array(e1CNs)).mean()
    avgCN_e2 = (np.array(e2CNs)).mean()
    avgCN_e1surface= (np.array(e1surfaceCNs)).mean()
    avgCN_e2surface= (np.array(e2surfaceCNs)).mean()
    
    return avgCN, avgSurfCN, avgCN_e1, avgCN_e2, avgCN_e1surface, avgCN_e2surface

def exportData(fileName, simulation, trials, e1, e2, a, cover):
    '''
    Generates a .csv file to store data about each configuration over the course of a parallel-tempering MC simulation.
    The first row is the data for the initial configuration, stored in an .xyz file named fileName.
    The number of subsequent rows is equal to the number of trials in the simulation, which is the output of PT_simulation
    
    The columns, from left to right, contain the following information:
        the atom number, 
        the cohesive energy per atom for the configuration,
        the numnber of surface pair sites for element 1, 
        the number of surface pair sites for element 2, 
        the number of atoms of element 1 on the surface, 
        and the number of atoms of element 2 on the surface.
    '''
    config=readability(fileName)
    saveName=fileName[:-4]
    parameters=parameterLookup(config, WF_1, WF_2, WF_index, BCE, adsorb=a)
    atom_types=parameters[0]
    BCEs = parameters[1]
    BCNs = parameters[2]
    weightfactors = parameters[3]
    with open('{}.csv'.format(saveName), 'w', newline='') as file:
        writer = csv.writer(file)
        #write a row of data for the initial configuration 
        writer.writerow(['init', energyCalculator(config, atom_types,BCEs, BCNs, weightfactors, cover)[0]/len(config), pairSiteCalculator(config, 'Pt', 'Au')[0], pairSiteCalculator(config, 'Pt', 'Au')[1], distributionCalculator(config, 'Pt', 'Au')[2], distributionCalculator(config, 'Pt', 'Au')[3]])  
        #write rows of data for each configuration generated during the the simulation
        if simulation[0][0]==e1:
            for i in range(trials):
                writer.writerow([i, simulation[1][i], simulation[4][i][0], simulation[4][i][1], simulation[5][i][0], simulation[5][i][1]])
        elif simulation[0][0]==e2:
            for i in range(trials):
                writer.writerow([i, simulation[1][i], simulation[4][i][1], simulation[4][i][0], simulation[5][i][1], simulation[5][i][0]])
                
def runSimulation(e1, e2, per1, x,y,z, MI, fileName, a, trials, kTcold, kTmed, kThot, cover):
    '''
    Generates a bimetallic alloy nanoslab containing elements e1 and e2 
    (where per 1= percentage of alloy that is element e1)
    with dimensions x by y by z cut along Miller Index MI
    
    Runs a parallel-tempering MC simulation of specified number of trials.
    Temperatures of the low, medium, and high-temp reservoirs, respectively, are kTcold, kTmed, kThot (in eV)
    
    Other Parameters
    ----------------
    a: string indicating chemical symbol of adsorbate
    
    cover: float between 0 and 1 indicating fraction of undercoordinated surface sites to be covered with adsorbates
    '''
    
    lc=calcLC(LCN[e1], LCN[e2], per1)
    dimensions=(x,y,z) 
    createSurface(dimensions, MI, lc, e1, per1, e2, fileName)
    findCNS(fileName)

    config = readability(fileName)
    parameters=parameterLookup(config, WF_1, WF_2, WF_index, BCE, adsorb=a)
    atom_types=parameters[0]
    BCEs = parameters[1]
    BCNs = parameters[2]
    weightfactors = parameters[3]
    
    run = PT_simulation(fileName, trials, 0.1, kTcold, kTmed, kThot, atom_types,BCEs, BCNs,weightfactors, cover)
    
    exportData(fileName, run, trials, e1, e2, a, cover)

    return run

def readData(filename): #read in the csv file and save each column as a separate list
    f = open(filename, 'r')
    data=f.readlines()
    
    energies=[]
    pairs_e1=[]
    pairs_e2=[]
    surface_e1=[]
    surface_e2=[]
    

    for line in data:
        vals = line.split(',')
        energies.append(float(vals[1]))
        pairs_e1.append(float(vals[2]))
        pairs_e2.append(float(vals[3]))
        surface_e1.append(float(vals[4]))
        surface_e2.append(float(vals[5]))
    return energies, pairs_e1, pairs_e2, surface_e1, surface_e2

def equilibrate(dataset, eq_point): #discard data points before the equilibration point (to determine eq_point, graph the data set as a whole before)
    save=[]
    for num, line in enumerate(dataset):
        if num > eq_point:
            save.append(line)
    return save 
