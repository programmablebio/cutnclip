Order of what was done:

ran get_partner_seqs.py. Selects peptides with length 20 and gets their sequences from PDB files. Saves to dips_20_mer_seqs.csv 
ran get_esm_embeddings.py on dips_20_mers_seqs.csv. Gets embeddings for most of the sequences in pepgen_20_mer_seqs.csv. Saves to embeddings.pkl (don't need to do for prot trans)
Run Dataset\ Preparation.ipynb. Cluster the sequences and make train tests splits, also combining in the partner embeddings for propedia saving to pepgen_dataset.csv


in peptide_embeddings/

