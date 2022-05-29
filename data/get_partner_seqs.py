from posixpath import split
import pandas as pd
import os
import pickle
from Bio import PDB
from Bio.SeqUtils import IUPACData
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_csv', dest='input_csv', default="all_peptides.csv")
    parser.add_argument('--output_csv', dest='output_csv', default="pepgen_20_mer_seqs.csv")
    parser.add_argument('--split', dest='split', type=int, default=1)
    parser.add_argument('--n_splits', dest='n_splits', type=int, default=1)

    args = parser.parse_args()

    peptide_data = pd.read_csv(args.input_csv)

    twenty_mers = peptide_data[peptide_data['peptide_length'] == 20]
    split_size = len(twenty_mers) // args.n_splits
    start = split_size * (args.split - 1)
    end = split_size * args.split
    twenty_mers = twenty_mers.iloc[start:end]

    # chain 1 is binding partner
    # chain 2 contains peptide
    pdb_directory = "train_pdbs/"
    pdb_parser = PDB.PDBParser()
    residue_dict = {k.upper(): v for k, v in IUPACData.protein_letters_3to1.items()}
    seqs = []

    with open("MSA_id_to_all_ids.pkl", "rb") as f:
        msa_to_id = pickle.load(f)

    id_to_msa = {}
    for msa, pdb_list in msa_to_id.items():
        for pdb in pdb_list:
            id_to_msa[pdb] = msa

    for i, row in twenty_mers.iterrows():
        try:
            if i % 100 == 0:
                print(i)
            msa_id = id_to_msa[f"{row['pdb_code']}_{row['chain_1'].strip()}"]

            pdb_code = row['pdb_code']
            chain_id = row['chain_1'].strip()
            pdb_path = os.path.join(pdb_directory, f"pdb{pdb_code}.pdb")
            chain = pdb_parser.get_structure("", pdb_path)[0][chain_id]

            seq = "".join([residue_dict[r.get_resname()] for r in chain.get_unpacked_list()])
            seqs.append(seq)
            
        except:
            seqs.append(None)
            print(i, row)
            
    twenty_mers = twenty_mers.assign(partner_sequence=seqs)

    twenty_mers.to_csv(args.output_csv)