import os

def generate_fasta(seq, fname):
    with open(fname, 'w') as f:
        f.write(">tmp\n")
        f.write(seq)

if __name__ == "__main__":
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--commands_file', dest='commands_file', default="commands.txt")
    parser.add_argument('--fasta_dir', dest='fasta_dir', default="fasta/")
    parser.add_argument('--msa_dir', dest='msa_dir', default="msas/")
    parser.add_argument('--regions_csv', dest='regions_csv', default="non_duplicate_seqs_dna_removed.csv")
    parser.add_argument('--hhblits_path', dest='hhblits_path')
    parser.add_argument('--database_path', dest='database_path')
    parser.add_argument('--n_cpu', dest='n_cpu', type=int)
    parser.add_argument('--mact', dest='mact', default=0.35, type=float)
    parser.add_argument('--e_value', dest='e_value', default=0.001, type=float)
    parser.add_argument('--n_iters', dest='n_iters', default=1, type=float)


    args = parser.parse_args()
    if not os.path.exists(args.fasta_dir):
        os.mkdir(args.fasta_dir)
    if not os.path.exists(args.msa_dir):
        os.mkdir(args.msa_dir)

    regions = pd.read_csv(args.regions_csv)

    f = open(args.commands_file, 'w')

    for i, row in regions.iterrows():
        pdbchain_id = row["pdbchain_id"]
        print(pdbchain_id)

        fasta_path = os.path.join(args.fasta_dir, f"{pdbchain_id}.fasta")
        generate_fasta(row["sequence"], fasta_path)

        if not os.path.exists(f"{os.path.join(args.msa_dir, pdbchain_id)}.a3m"):
            cmd = f"{args.hhblits_path} -cpu {args.n_cpu} -mact {args.mact} -e {args.e_value} -i {fasta_path} -d {args.database_path} -oa3m {os.path.join(args.msa_dir, pdbchain_id)}.a3m -n {args.n_iters}"
            f.write(f"{cmd}\n")
    
    f.close()



        

