import os
import string
from Bio import SeqIO
import itertools
import pandas as pd
import torch
import sys
sys.path.append("/home/gridsan/kalyanpa/DNAInteract_shared/kalyan/esm")
import esm
import torch.nn.functional as F
from collections import defaultdict


sys.path.append("..")
from model import PeptideCLIP
import argparse

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

clip_model_configs = [
    {
        'config': {'regions_csv_path': 'pepgen_dataset.csv', 'peptide_embeddings_path': 'peptide_embeddings.pkl', 'partner_embeddings_path': 'partner_embeddings.pkl', 'random_seed': 42, 'batch_size': 256, 'model_id': 'caramel-dingo', 'peptide_input_dim': 1280, 'partner_input_dim': 768, 'use_interface': False, 'n_gpus': 1, 'precision': 32, 'max_epochs': 50, 'early_stopping': True, 'early_stopping_patience': 15, 'anomaly_detection': False, 'optuna': False, 'optuna_trials': 100, 'prune_optuna': False, 'lr': 0.0011892083317287768, 'peptide_sequence_mlp_layers': 2, 'partner_sequence_mlp_layers': 2, 'peptide_transformer_heads': 2, 'peptide_transformer_layers': 0, 'partner_transformer_heads': 2, 'partner_transformer_layers': 0, 'peptide_length_reducer': 'avg', 'partner_length_reducer': 'avg', 'embedding_dim': 448, 'peptide_embedding_mlp_layers': 2, 'partner_embedding_mlp_layers': 4, 'layer_norm_eps': 3.1249402606181844e-05, 'sequence_dropout': 0.1, 'embedding_dropout': 0.1828798928101121, 'temperature': 0.07, 'learn_temperature': True, 'fast_dev_run': False, 'resume_checkpoint': None, 'save_top_k': -1, 'submit': False},
        'weights_path': "/home/gridsan/kalyanpa/DNAInteract_shared/kalyan/peptide-clip/manual_models/caramel-dingo/top_models/epoch=12-step=1286.ckpt"
    },
    {
        'config': {'regions_csv_path': 'pepgen_dataset.csv', 'peptide_embeddings_path': 'peptide_embeddings.pkl', 'partner_embeddings_path': 'partner_embeddings.pkl', 'random_seed': 42, 'batch_size': 256, 'model_id': 'no_interfaces_optuna', 'peptide_input_dim': 1280, 'partner_input_dim': 768, 'use_interface': False, 'n_gpus': 1, 'precision': 32, 'max_epochs': 50, 'early_stopping': True, 'early_stopping_patience': 10, 'anomaly_detection': False, 'optuna': True, 'optuna_trials': 200, 'prune_optuna': True, 'lr': 0.0015705727133902706, 'peptide_sequence_mlp_layers': 2, 'partner_sequence_mlp_layers': 2, 'peptide_transformer_heads': 4, 'peptide_transformer_layers': 0, 'partner_transformer_heads': 2, 'partner_transformer_layers': 0, 'peptide_length_reducer': 'avg', 'partner_length_reducer': 'avg', 'embedding_dim': 512, 'peptide_embedding_mlp_layers': 3, 'partner_embedding_mlp_layers': 2, 'layer_norm_eps': 0.0001206183023734721, 'sequence_dropout': 0.5853923010065312, 'embedding_dropout': 0.3771066209433185, 'temperature': 0.07, 'learn_temperature': True, 'fast_dev_run': False, 'resume_checkpoint': None, 'submit': False},
        'weights_path': "/home/gridsan/kalyanpa/DNAInteract_shared/kalyan/peptide-clip/optuna_models/no_interfaces_optuna/28/top_models/epoch=11-step=1187.ckpt"
    },
    {
        'config': {'regions_csv_path': 'pepgen_dataset.csv', 'peptide_embeddings_path': 'peptide_embeddings.pkl', 'partner_embeddings_path': 'partner_embeddings.pkl', 'random_seed': 42, 'batch_size': 256, 'model_id': 'no_interfaces_optuna', 'peptide_input_dim': 1280, 'partner_input_dim': 768, 'use_interface': False, 'n_gpus': 1, 'precision': 32, 'max_epochs': 50, 'early_stopping': True, 'early_stopping_patience': 10, 'anomaly_detection': False, 'optuna': True, 'optuna_trials': 200, 'prune_optuna': True, 'lr': 0.0009374579539475231, 'peptide_sequence_mlp_layers': 2, 'partner_sequence_mlp_layers': 3, 'peptide_transformer_heads': 4, 'peptide_transformer_layers': 0, 'partner_transformer_heads': 2, 'partner_transformer_layers': 0, 'peptide_length_reducer': 'avg', 'partner_length_reducer': 'avg', 'embedding_dim': 512, 'peptide_embedding_mlp_layers': 3, 'partner_embedding_mlp_layers': 2, 'layer_norm_eps': 0.0004957444357116003, 'sequence_dropout': 0.6447688463223357, 'embedding_dropout': 0.35934966293444226, 'temperature': 0.07, 'learn_temperature': True, 'fast_dev_run': False, 'resume_checkpoint': None, 'submit': False},
        'weights_path': "/home/gridsan/kalyanpa/DNAInteract_shared/kalyan/peptide-clip/optuna_models/no_interfaces_optuna/32/top_models/epoch=16-step=1682.ckpt"
    },
    {
        'config': {'regions_csv_path': 'pepgen_dataset.csv', 'peptide_embeddings_path': 'peptide_embeddings.pkl', 'partner_embeddings_path': 'partner_embeddings.pkl', 'random_seed': 42, 'batch_size': 256, 'model_id': 'no_interfaces_optuna', 'peptide_input_dim': 1280, 'partner_input_dim': 768, 'use_interface': False, 'n_gpus': 1, 'precision': 32, 'max_epochs': 50, 'early_stopping': True, 'early_stopping_patience': 10, 'anomaly_detection': False, 'optuna': True, 'optuna_trials': 200, 'prune_optuna': True, 'lr': 0.0027389681800264534, 'peptide_sequence_mlp_layers': 3, 'partner_sequence_mlp_layers': 3, 'peptide_transformer_heads': 4, 'peptide_transformer_layers': 0, 'partner_transformer_heads': 2, 'partner_transformer_layers': 0, 'peptide_length_reducer': 'avg', 'partner_length_reducer': 'avg', 'embedding_dim': 288, 'peptide_embedding_mlp_layers': 2, 'partner_embedding_mlp_layers': 2, 'layer_norm_eps': 1.141714814984072e-06, 'sequence_dropout': 0.673231483110263, 'embedding_dropout': 0.29676813399657576, 'temperature': 0.07, 'learn_temperature': True, 'fast_dev_run': False, 'resume_checkpoint': None, 'submit': False},
        'weights_path': "/home/gridsan/kalyanpa/DNAInteract_shared/kalyan/peptide-clip/optuna_models/no_interfaces_optuna/12/top_models/epoch=16-step=1682.ckpt"
    }
]

# {
#     'config': {'regions_csv_path': 'pepgen_dataset.csv', 'peptide_embeddings_path': 'peptide_embeddings.pkl', 'partner_embeddings_path': 'partner_embeddings.pkl', 'random_seed': 42, 'batch_size': 512, 'model_id': '512_test', 'peptide_input_dim': 1280, 'partner_input_dim': 768, 'use_interface': False, 'n_gpus': 1, 'precision': 32, 'max_epochs': 50, 'early_stopping': False, 'early_stopping_patience': 10, 'anomaly_detection': False, 'optuna': False, 'optuna_trials': 100, 'prune_optuna': False, 'lr': 0.001, 'peptide_sequence_mlp_layers': 2, 'partner_sequence_mlp_layers': 2, 'peptide_transformer_heads': 2, 'peptide_transformer_layers': 0, 'partner_transformer_heads': 2, 'partner_transformer_layers': 0, 'peptide_length_reducer': 'avg', 'partner_length_reducer': 'avg', 'embedding_dim': 512, 'peptide_embedding_mlp_layers': 2, 'partner_embedding_mlp_layers': 2, 'layer_norm_eps': 1e-05, 'sequence_dropout': 0.1, 'embedding_dropout': 0.1, 'temperature': 0.07, 'learn_temperature': False, 'fast_dev_run': False, 'resume_checkpoint': None, 'submit': False},
#     'weights_path': "/home/gridsan/kalyanpa/DNAInteract_shared/kalyan/peptide-clip/manual_models/512_test/top_models/epoch=15-step=783.ckpt"
# }

# clip_config = 
# clip_config = 

# clip_weights_path = "/home/gridsan/kalyanpa/DNAInteract_shared/kalyan/peptide-clip/manual_models/caramel-dingo/top_models/epoch=12-step=1286.ckpt"
# clip_weights_path = 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train peptide-clip')
    parser.add_argument('--hhblits_path', dest='hhblits_path', help="Path to hhblits executable", default="/home/gridsan/kalyanpa/DNAInteract_shared/manu/hh-suite/build/bin/hhblits")
    parser.add_argument('--uniref_path', dest="uniref_path", help="path to Uniref", default="/home/gridsan/groups/DNAInteract/kalyan/MSA_generation/UniRef30_2020_06/UniRef30_2020_06")
    parser.add_argument('--n_cpus', help="number of CPUs used in MSA generation", dest="n_cpus")
    parser.add_argument('--hhblits_iters', help="number of iterations used in MSA generation", dest="hhblits_iters")

    parser.add_argument('--esm_msa_weights_path', dest="esm_msa_weights_path", help="path to ESM MSA transformer weights", default="/home/gridsan/kalyanpa/DNAInteract_shared/kalyan/esm_weights/esm_msa1_t12_100M_UR50S.pt")
    parser.add_argument('--esm1b_weights_path', dest="esm1b_weights_path", help="path to ESM-1b weights", default="/home/gridsan/kalyanpa/DNAInteract_shared/kalyan/esm_weights/esm1b_t33_650M_UR50S.pt")

    parser.add_argument('--target_fasta', help="FASTA file for target protein", dest="target_fasta")
    parser.add_argument('--binder_fasta', help="FASTA file for binding protein which will be sliced up into peptides", dest="binder_fasta")
    parser.add_argument('--peptide_length', help="Length of peptides cut from binder protein", dest="peptide_length", type=int)

    parser.add_argument('--n_peptides', help="Number of peptide candidates desired", dest="n_peptides", type=int)
    parser.add_argument('--output_dir', help="Directory to output peptides", dest="output_dir", default="outputs")

    parser.add_argument('--filter_overlapping',  help="Filter out peptides with high overlap with other top peptides", action='store_true')
    parser.add_argument('--n_overlapping_peptides', help="Number of highly overlapping peptides to allow before filtering", dest="n_overlapping_peptides", default=2, type=int)
    parser.add_argument('--overlap_threshold', help="Difference in starting index required to constitute peptides which don't overlap", dest="overlap_threshold", default=4, type=int)

    args = parser.parse_args()
    target_name = args.target_fasta.split("/")[-1].split(".")[0]
    output_msa_path = f"msas/{target_name}.a3m"
    if not os.path.exists(output_msa_path):
        print("Computing target MSA...")
        cmd = f"{args.hhblits_path} -cpu {args.n_cpus} -i {args.target_fasta} -d {args.uniref_path} -oa3m {output_msa_path} -n {args.hhblits_iters}"
        result = os.system(cmd)
    else:
        print("Target MSA already exists! Using it...")

    esm_msa_weights = torch.load(args.esm_msa_weights_path, map_location="cpu")

    esm_msa_model, esm_msa_alphabet = esm.pretrained.load_model_and_alphabet_core(esm_msa_weights, None)
    esm_msa_model = esm_msa_model.eval()
    esm_msa_batch_converter = esm_msa_alphabet.get_batch_converter()
    
    print("Computing target embedding...")
    msa = read_msa(output_msa_path, 64)
    _, _, msa_tokens = esm_msa_batch_converter(msa)
    with torch.no_grad():
        target_outputs = esm_msa_model(msa_tokens, repr_layers=[12], return_contacts=False)

    # just representation for the main sequence
    target_embedding = target_outputs['representations'][12][0, 0, :, :]

    _, binder_seq = read_sequence(args.binder_fasta)
    binder_name = args.binder_fasta.split("/")[-1].split(".")[0]

    print("Computing peptide embeddings...")
    peptide_seqs = []
    for i in range(len(binder_seq) - args.peptide_length):
        peptide_seqs.append((str(i), binder_seq[i:i + args.peptide_length]))

        # for sub_length in range(10, args.peptide_length + 1, 2):
        #     peptide_seqs.append((str(i), binder_seq[i:i + sub_length]))

    esm1b_data = torch.load(args.esm1b_weights_path, map_location="cpu")
    esm1b_model, esm1b_alphabet = esm.pretrained.load_model_and_alphabet_core(esm1b_data, None)
    esm1b_model = esm1b_model.eval()
    esm1b_batch_converter = esm1b_alphabet.get_batch_converter()
    _, _, peptide_input = esm1b_batch_converter(peptide_seqs)
    with torch.no_grad():
        outputs = esm1b_model(peptide_input, repr_layers=[33], return_contacts=False)

    peptide_embeddings = outputs['representations'][33]

    print("Running CLIP...")
    avg_scores = defaultdict(lambda: 0)
    for config in clip_model_configs:
        model = PeptideCLIP(config['config'])
        model.eval()
        model.load_state_dict(torch.load(config['weights_path'], map_location=torch.device('cpu'))['state_dict'])

        num_peptides = peptide_embeddings.shape[0]
        target_embedding_stacked = target_embedding.unsqueeze(0).repeat(num_peptides, 1, 1)
        with torch.no_grad():
            scores = F.softmax(model(peptide_embeddings, target_embedding_stacked)[:, 0], dim=0).tolist()
        for i, score in enumerate(scores):
            avg_scores[i] += score
    
    table_data = {
        "Sequence": [],
        "Score": [],
        "Score relative to average": [],
        "Name": [],
        "Starting index": []
    }

    # average of all scores
    avg_score = 1 / len(scores)

    for i, seq in peptide_seqs:
        table_data["Sequence"].append(seq)
        table_data["Score"].append(avg_scores[int(i)] / len(clip_model_configs))
        table_data["Score relative to average"].append(avg_scores[int(i)] / len(clip_model_configs) / avg_score)
        table_data["Name"].append(f"{binder_name}_{i}")
        table_data["Starting index"].append(int(i))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    df = pd.DataFrame(data=table_data)

    df.sort_values(by="Score", ascending=False, inplace=True)
    df.reset_index(inplace=True, drop=True)

    # TODO: redo this. If n_peptides is close to the total possible number of peptides this will maybe fail in a way we don't want
    if args.filter_overlapping:
        n_selected_peptides = 0
        for row in range(len(df)):
            if n_selected_peptides >= args.n_peptides:
                break
            
            # count number of peptides higher ranked which are highly overlapping with given peptide
            if ((df.loc[0:(row - 1)]['Starting index'] - df.loc[row]['Starting index']).abs() < args.overlap_threshold).sum() >= args.n_overlapping_peptides:
                df.drop(row, inplace=True)



    save_file = f"{target_name}-{binder_name}-{args.peptide_length}-mers.csv"
    df.head(args.n_peptides).to_csv(os.path.join(args.output_dir, save_file))
    print("Top peptides saved at " + save_file)
    print(df.head(args.n_peptides))
    print(f"Average score for reference: {avg_score}")
        

    # top_peptide_idxs = scores.argsort()[-args.n_peptides:]
    # top_peptide_scores = scores[top_peptide_idxs].tolist()
    # top_peptide_seqs = []
    # for idx in top_peptide_idxs:
    #     top_peptide_seqs.append(binder_seq[idx:(idx+args.peptide_length)])

    # df = pd.DataFrame(data={"Sequence": top_peptide_seqs[::-1], "Score": top_peptide_scores[::-1], "Name": [f"{binder_name}_{i}" for i in range(1, args.n_peptides + 1)], "Starting index": top_peptide_idxs.tolist()[::-1]})
    
    # df.to_csv(os.path.join(args.output_dir, f"{target_name}-{binder_name}-{args.peptide_length}-mers.csv"))
