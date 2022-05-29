import os
import string
from Bio import SeqIO
import itertools
import pandas as pd

"""
input data: 
msas stored at
s3://ubiquitx-datasets/processed-data/propedia-binding-regions/binding_regions_msas.tar.gz

output data:
msas
"""


#next 3 fns from https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename):
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence):
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename, nseq):
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]


if __name__ == "__main__":
    import argparse
    import torch
    import pickle
    import sys
    sys.path.append("/home/gridsan/kalyanpa/DNAInteract_shared/kalyan/esm")
    import esm
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_csv', dest='input_csv', default="pepgen_20_mer_seqs.csv")
    parser.add_argument('--output_file', dest='output_file', default="manu_data_embeddings.pkl")
    parser.add_argument('--msa_dir', dest='msa_dir', default="/home/gridsan/kalyanpa/DNAInteract_shared/kalyan/dips_msas/msas/")
    parser.add_argument('--msa_dict_pickle', dest='msa_dict_pickle', default="MSA_id_to_all_ids.pkl")
    parser.add_argument('--esm_weights_path', dest="esm_weights_path")

    args = parser.parse_args()

    twenty_mers = pd.read_csv(args.input_csv)


    with open("MSA_id_to_all_ids.pkl", "rb") as f:
        msa_to_id = pickle.load(f)
    
    id_to_msa = {}
    for msa, pdb_list in msa_to_id.items():
        for pdb in pdb_list:
            id_to_msa[pdb] = msa

    embeddings = {}

    model_data = torch.load(args.esm_weights_path, map_location="cpu")

    esm_model, esm_alphabet = esm.pretrained.load_model_and_alphabet_core(model_data, None)
    esm_model = esm_model.eval()
    esm_model = esm_model.cuda()
    esm_batch_converter = esm_alphabet.get_batch_converter()
    error_ids = []
    iter = 0
    for i, row in twenty_mers.iterrows():
        iter += 1
        print(iter)
        try:
            pdb_id = f"{row['pdb_code']}_{row['chain_1'].strip()}"

            if pdb_id in embeddings:
                continue

            msa_id = id_to_msa[pdb_id]
            
            msa_path = os.path.join(args.msa_dir, f"{msa_id}.a3m")
            msa = read_msa(os.path.join(args.msa_dir, msa_path), 64)
            esm_batch_labels, esm_batch_strs, esm_batch_tokens = esm_batch_converter(msa)
            esm_batch_tokens = esm_batch_tokens.cuda()
            with torch.no_grad():
                outputs = esm_model(esm_batch_tokens, repr_layers=[12], return_contacts=False)

            # just representation for the main sequence
            embeddings[pdb_id] = outputs['representations'][12][:, 0, :, :].cpu()
            del esm_batch_tokens
        except:
            error_ids.append(i)


    with open(args.output_file, 'wb') as f:
        pickle.dump(embeddings, f)

    print(error_ids)