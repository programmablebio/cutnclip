import pandas as pd

if __name__ == "__main__":
    import argparse
    import torch
    import pickle
    import sys
    sys.path.append("/home/gridsan/kalyanpa/DNAInteract_shared/kalyan/esm")
    import esm
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_csv', dest='input_csv', default="/home/gridsan/kalyanpa/DNAInteract_shared/kalyan/pepgen_data/pepgen_dataset.csv")
    parser.add_argument('--output_file', dest='output_file', default="pepgen_peptide_embeddings.pkl")
    parser.add_argument('--esm_weights_path', dest="esm_weights_path", default="/home/gridsan/kalyanpa/DNAInteract_shared/kalyan/esm_weights/esm1b_t33_650M_UR50S.pt")

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df.drop_duplicates(subset=['peptide_chain'])

    embeddings = {}

    model_data = torch.load(args.esm_weights_path, map_location="cpu")

    esm_model, esm_alphabet = esm.pretrained.load_model_and_alphabet_core(model_data, None)
    esm_model = esm_model.eval()
    esm_model = esm_model.cuda()
    esm_batch_converter = esm_alphabet.get_batch_converter()
    error_ids = []
    iter = 0
    for i, row in df.iterrows():
        iter += 1
        print(iter)
        try:
            esm_input = [("_", row['peptide_seq'])]
            esm_batch_labels, esm_batch_strs, esm_batch_tokens = esm_batch_converter(esm_input)
            esm_batch_tokens = esm_batch_tokens.cuda()
            with torch.no_grad():
                outputs = esm_model(esm_batch_tokens, repr_layers=[33], return_contacts=False)

            # just representation for the main sequence
            embeddings[row['peptide_chain']] = outputs['representations'][33][0, :].cpu()
            del esm_batch_tokens
        except:
            print("error")
            error_ids.append(row['peptide_chain'])


    with open(args.output_file, 'wb') as f:
        pickle.dump(embeddings, f)

    print(error_ids)