from re import L
from Bio.PDB import PDBParser, PPBuilder, Selection
from Bio import SeqIO
import pandas as pd
import os
from Bio.SeqUtils import IUPACData


"""
raw input directories interface_pdbs/ and receptor_pdbs/ are saved at
s3://ubiquitx-datasets/raw-data/propedia/interface_pdbs/
and 
s3://ubiquitx-datasets/raw-data/propedia/receptor_pdbs/

output file binding_regions.csv saved to s3://ubiquitx-datasets/processed-data/propedia-binding-regions/binding_regions.csv
"""

peptide_dict = {}
peptide_seqs = SeqIO.parse(open("peptides.fasta"),'fasta')
for entry in peptide_seqs:
    name = entry.id.split("|")[0]
    seq = str(entry.seq)
    peptide_dict[name] = seq

parser = PDBParser()
peptide_builder = PPBuilder()
residue_dict = {k.upper(): v for k, v in IUPACData.protein_letters_3to1.items()}

df = pd.DataFrame(columns=['pdb_name', 
                          'peptide_chain', 
                          'partner_chain', 
                          'peptide_seq', 
                          'partner_seq', 
                          'binding_region_indices', 
                          'closure_region_indices', 
                          'closure_region_seq'])

df.index.name = "propedia_id"

polypeptide_errors = 0

for interface_filename in os.listdir("interface_pdbs"):
    interface_id = interface_filename.split(".")[0]
    pdb_name, peptide_chain, partner_chain = interface_id.split("_")
    
    receptor_structure = parser.get_structure("X", f"receptor_pdbs/{pdb_name}_{partner_chain}.pdb")
        
    interface_structure = parser.get_structure("X", f"interface_pdbs/{pdb_name}_{peptide_chain}_{partner_chain}.pdb")
    binding_region_indices = []
    
    partner_interface_residues = interface_structure[0][partner_chain].get_unpacked_list()
    partner_interface_residue_positions = [r.get_id()[1] for r in partner_interface_residues]

    receptor_residues = receptor_structure[0][partner_chain].get_unpacked_list()

    binding_region_indices = []

    i = 0
    receptor_seq = ""


    # only select amino acids
    # PDBs have metals in the proteins that count as residues
    for res in receptor_residues:

        residue_name = res.get_resname()
        if residue_name in residue_dict:
            receptor_seq = receptor_seq + residue_dict[residue_name]
            if res.get_id()[1] in partner_interface_residue_positions:
                binding_region_indices.append(i)
            
            # i indexes into receptor sequence so only increment if residue
            # was amino acid
            i += 1

            
            
    binding_region_indices = sorted(binding_region_indices)
    

    # (closure region is the region containing all of the binding regions)
    closure_region_start = min(binding_region_indices)
    closure_region_end = max(binding_region_indices)

    # binding region indices are valid
    assert closure_region_start >= 0
    assert closure_region_end < len(receptor_seq)

    df.loc[interface_id] =   {'pdb_name': pdb_name,
                            'peptide_chain': peptide_chain,
                            'partner_chain': partner_chain,
                            'peptide_seq': peptide_dict[f"{pdb_name}-{peptide_chain}-{partner_chain}"],
                            'partner_seq': receptor_seq,
                            'binding_region_indices': binding_region_indices,
                            'closure_region_indices': (closure_region_start, closure_region_end),
                            'closure_region_seq': receptor_seq[closure_region_start:closure_region_end]
    }


df.to_csv("binding_regions.csv")
print(polypeptide_errors)
