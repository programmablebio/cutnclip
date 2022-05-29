from unittest import makeSuite
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
import pickle
from torch.nn.utils.rnn import pad_sequence
import ast


class PartnerPeptideDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, peptide_embedding_dict, partner_embedding_dict, use_interface=False):
        self.dataframe = dataframe[dataframe["partner_chain"].isin(partner_embedding_dict.keys())]
        self.dataframe = self.dataframe[dataframe["peptide_chain"].isin(peptide_embedding_dict.keys())]
        if use_interface:
            self.dataframe = self.dataframe.dropna(subset=["binding_region_indices"])
        self.dataframe = self.dataframe.reset_index()

        print(f"Loaded {len(self.dataframe)} sequences")

        self.peptide_embedding_dict = peptide_embedding_dict
        self.partner_embedding_dict = partner_embedding_dict

        self.use_interface = use_interface

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # Select row
        row = self.dataframe.iloc[index]
        pdb_id = row["pdb_id"]
        partner_chain = row["partner_chain"]
        peptide_chain = row["peptide_chain"]
        
        # L x E
        peptide_embedding = self.peptide_embedding_dict[peptide_chain][:, :]
        partner_embedding = self.partner_embedding_dict[partner_chain][0, :, :]

        return_dict = {
            "pdb_id": pdb_id,
            "peptide_seq": row["peptide_seq"],
            "peptide_input": peptide_embedding,
            "partner_input": partner_embedding,
            "binding_region_indices": row['binding_region_indices']
        }

        if self.use_interface:
            return_dict['binding_region_indices'] = row['binding_region_indices']

        return return_dict

class PartnerPeptideCollator:
    def __init__(self, use_interface=False):
        self.use_interface = use_interface

    def __call__(self, raw_batch):
        batch = {}
        batch_size = len(raw_batch)

        peptide_input_list = [v['peptide_input'] for v in raw_batch]

        if self.use_interface:
            partner_input_list = []
            for d in raw_batch:
                interface_row = torch.zeros(d['partner_input'].shape[0], 1)
                if isinstance(d['binding_region_indices'], str):
                    indices = ast.literal_eval(d['binding_region_indices'])
                else:
                    indices = d['binding_region_indices']

                for i in indices:
                    interface_row[int(i), 0] = 1
                partner_input_list.append(torch.cat([d['partner_input'], interface_row], dim=1))
        else:
            partner_input_list = [d['partner_input'] for d in raw_batch]

        batch['peptide_input'] = pad_sequence(peptide_input_list, batch_first=True, padding_value=0.)
        batch['peptide_padding_mask'] = self.get_padding_mask(peptide_input_list, batch['peptide_input'])

        batch['partner_input'] = pad_sequence(partner_input_list, batch_first=True, padding_value=0.)
        batch['partner_padding_mask'] = self.get_padding_mask(partner_input_list, batch['partner_input'])

        # if torch.any(batch['peptide_input'].isnan() | batch['peptide_input'].isinf()):
        #     print("peptide input nan")
        
        # if torch.any(batch['partner_input'].isnan() | batch['partner_input'].isinf()):
        #     print("partner input nan")
        
        batch['labels'] = torch.arange(batch_size)
        batch['pdb_ids'] = [v['pdb_id'] for v in raw_batch]

        return batch

    def get_padding_mask(self, input_list, padded_input):
        mask = torch.full(padded_input.shape[:2], False, dtype=torch.bool)
        for i, embedding in enumerate(input_list):
            mask[i, embedding.shape[0]:] = True
        return mask


class PartnerPeptideDataModule(pl.LightningDataModule):
    def __init__(self,
                 regions_csv_path,
                 peptide_embeddings_path,
                 partner_embeddings_path,
                 batch_size=512,
                 use_interface=False):
        super().__init__()
        
        self.regions_csv_path = regions_csv_path
        self.peptide_embeddings_path = peptide_embeddings_path
        self.partner_embeddings_path = partner_embeddings_path
        self.batch_size = batch_size
        self.use_interface = use_interface

    def setup(self, stage):
        regions = pd.read_csv(self.regions_csv_path)
        with open(self.peptide_embeddings_path, "rb") as f:
            peptide_embeddings = pickle.load(f)
        with open(self.partner_embeddings_path, "rb") as f:
            partner_embeddings = pickle.load(f)

        train_df = regions[regions['split'] == 'train']
        test_df = regions[regions['split'] == 'test']
        val_df = regions[regions['split'] == 'val']

        self.train_dataset = PartnerPeptideDataset(train_df,
                                                  peptide_embeddings,
                                                  partner_embeddings,
                                                  use_interface=self.use_interface)
        self.test_dataset = PartnerPeptideDataset(test_df, 
                                                  peptide_embeddings,
                                                  partner_embeddings,
                                                  use_interface=self.use_interface)
        self.val_dataset = PartnerPeptideDataset(val_df, 
                                                 peptide_embeddings,
                                                 partner_embeddings,
                                                 use_interface=self.use_interface)
        
        self.collator = PartnerPeptideCollator(use_interface=self.use_interface)
    
    def train_dataloader(self):
        # shuffling the dataloader is extremely important for a CLIP model! 
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collator, num_workers=1, shuffle=True, drop_last=True)

    def val_dataloader(self):
        full_batch = DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collator, num_workers=1, shuffle=True, drop_last=True)
        binary_batch = DataLoader(self.val_dataset, batch_size=2, collate_fn=self.collator, num_workers=4, shuffle=True, drop_last=True)
        return [full_batch, binary_batch]
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collator, num_workers=0, shuffle=True, drop_last=True)