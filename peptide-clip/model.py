from argparse import ArgumentError
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math


class MLP(pl.LightningModule):
    def __init__(self, input_dim, embedding_dim, num_layers, dropout=0, output_relu=False, length_reducer='avg'):
        super().__init__()
        # length_reducer reduces the length dimension of sequence embedding
        # can be either "avg", "cls", "rnn"
        # "avg": average over amino acid token representations as done in the ESM paper
        # "cls": use CLS token representation
        # "rnn": runs sequence embedding through RNN
        # None: don't reduce sequence length, output full sequence

        self.length_reducer = length_reducer

        if length_reducer == "rnn":
            self.rnn = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=input_dim,
                batch_first=True
            )

        layers_list = [nn.Linear(input_dim, embedding_dim)]
        for i in range(num_layers - 1):
            # relu for previous layer gets added first
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(p=dropout))
            layers_list.append(nn.Linear(embedding_dim, embedding_dim))

        if output_relu:
            layers_list.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers_list)

    def forward(self, input_embedding, padding_mask=None):
        if self.length_reducer == 'cls':
            embedding = input_embedding[:, 0, :]
        elif self.length_reducer == 'avg':
            # don't use CLS token in average, as suggested in ESM README
            embedding = input_embedding[:, 1:, :].sum(1)
            if padding_mask is not None:
                # also don't average padded tokens
                embedding = embedding / ((~padding_mask).sum(1, keepdim=True) - 1)
        elif self.length_reducer == 'rnn':
            _, (embedding, _) = self.rnn(input_embedding)
            embedding = embedding[0, :, :]
        elif self.length_reducer is None:
            embedding = input_embedding
        else:
            raise ValueError("length reducer should be 'avg', 'cls', 'rnn', or None")
        
        embedding = self.layers(embedding)
        return embedding



class PeptideCLIP(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
                
        # start by adjusting dimensionality of input embedding
        self.peptide_sequence_mlp = MLP(
            input_dim=self.config['peptide_input_dim'],
            embedding_dim=self.config['embedding_dim'],
            num_layers=self.config['peptide_sequence_mlp_layers'],
            dropout=config['sequence_dropout'],
            length_reducer=None,
            output_relu=True
        )
        self.partner_sequence_mlp = MLP(
            input_dim=self.config['partner_input_dim'],
            embedding_dim=self.config['embedding_dim'],
            num_layers=self.config['partner_sequence_mlp_layers'],
            dropout=config['sequence_dropout'],
            length_reducer=None,
            output_relu=True
        )

        peptide_transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.config['embedding_dim'],
            nhead=self.config['peptide_transformer_heads'],
            dim_feedforward=self.config['embedding_dim'] * 2,
            dropout=self.config['sequence_dropout'],
            layer_norm_eps=self.config['layer_norm_eps'],
            batch_first=True
        )
        self.peptide_transformer = nn.TransformerEncoder(peptide_transformer_layer, self.config['peptide_transformer_layers'])

        partner_transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.config['embedding_dim'],
            nhead=self.config['partner_transformer_heads'],
            dim_feedforward=self.config['embedding_dim'] * 2,
            dropout=self.config['sequence_dropout'],
            layer_norm_eps=self.config['layer_norm_eps'],
            batch_first=True
        )
        self.partner_transformer = nn.TransformerEncoder(partner_transformer_layer, self.config['partner_transformer_layers'])

        self.peptide_embedding_mlp = MLP(
            input_dim=self.config['embedding_dim'],
            embedding_dim=self.config['embedding_dim'],
            num_layers=self.config['peptide_embedding_mlp_layers'],
            dropout=config['embedding_dropout'],
            length_reducer=config['peptide_length_reducer'],
            output_relu=False
        )

        self.partner_embedding_mlp = MLP(
            input_dim=self.config['embedding_dim'],
            embedding_dim=self.config['embedding_dim'],
            num_layers=self.config['partner_embedding_mlp_layers'],
            dropout=config['embedding_dropout'],
            length_reducer=config['partner_length_reducer'],
            output_relu=False
        )
        
        self.temperature = nn.Parameter(torch.full((1,), self.config['temperature']), requires_grad=self.config['learn_temperature'])
        self.save_hyperparameters()
            
    def forward(self, peptide_input, partner_input, peptide_padding_mask=None, partner_padding_mask=None):
        # based off of figure 3 from CLIP paper
        peptide_embedding = self.peptide_sequence_mlp(peptide_input)
        partner_embedding = self.partner_sequence_mlp(partner_input)

        peptide_embedding = self.peptide_transformer(peptide_embedding, src_key_padding_mask=peptide_padding_mask)
        partner_embedding = self.partner_transformer(partner_embedding, src_key_padding_mask=partner_padding_mask)

        # B x E
        peptide_embedding = F.normalize(self.peptide_embedding_mlp(peptide_embedding, padding_mask=peptide_padding_mask), dim=1)
        partner_embedding = F.normalize(self.partner_embedding_mlp(partner_embedding, padding_mask=partner_padding_mask), dim=1)

        # B x B
        logits = torch.matmul(peptide_embedding, partner_embedding.T) * torch.exp(self.temperature)

        return logits
            
    
    def training_step(self, batch, batch_idx):
        logits = self(
            batch['peptide_input'], 
            batch['partner_input'], 
            peptide_padding_mask=batch['peptide_padding_mask'],
            partner_padding_mask=batch['partner_padding_mask']
        )

        # loss of predicting partner using peptide
        partner_prediction_loss = F.cross_entropy(logits, batch['labels'])
        # loss of predicting peptide using partner
        peptide_prediction_loss = F.cross_entropy(logits.T, batch['labels'])

        loss = (partner_prediction_loss + peptide_prediction_loss) / 2
        
        self.log("train_loss", loss, sync_dist=True, batch_size=logits.shape[0])
        self.log("train_partner_prediction_loss", partner_prediction_loss, sync_dist=True, prog_bar=False, batch_size=logits.shape[0])
        self.log("train_peptide_prediction_loss", peptide_prediction_loss, sync_dist=True, prog_bar=False, batch_size=logits.shape[0])
        self.log("train_loss", loss, sync_dist=True, batch_size=logits.shape[0])
        self.log("train_partner_perplexity", torch.exp(partner_prediction_loss), sync_dist=True, prog_bar=True, batch_size=logits.shape[0])
        self.log("train_peptide_perplexity", torch.exp(peptide_prediction_loss), sync_dist=True, prog_bar=True, batch_size=logits.shape[0])

        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            # Predict on random batches of training batch size
            logits = self(
                batch['peptide_input'], 
                batch['partner_input'], 
                peptide_padding_mask=batch['peptide_padding_mask'],
                partner_padding_mask=batch['partner_padding_mask']
            )
            partner_prediction_loss = F.cross_entropy(logits, batch['labels'])
            peptide_prediction_loss = F.cross_entropy(logits.T, batch['labels'])
            loss = (partner_prediction_loss + peptide_prediction_loss) / 2


            # prediction of peptides for each partner
            peptide_predictions = logits.argmax(dim=0)
            # prediction of partners for each peptide
            partner_predictions = logits.argmax(dim=1)

            peptide_ranks = logits.argsort(dim=0).diag() + 1
            peptide_mrr = (peptide_ranks).float().pow(-1).mean()

            partner_ranks = logits.argsort(dim=1).diag() + 1
            partner_mrr = (partner_ranks).float().pow(-1).mean()

            partner_accuracy = partner_predictions.eq(batch['labels']).float().mean()
            peptide_accuracy = peptide_predictions.eq(batch['labels']).float().mean()

            k = int(logits.shape[0] / 10)
            peptide_topk_accuracy = torch.any((logits.topk(k, dim=0).indices - batch['labels'].reshape(1, -1)) == 0, dim=0).sum() / logits.shape[0]
            partner_topk_accuracy = torch.any((logits.topk(k, dim=1).indices - batch['labels'].reshape(-1, 1)) == 0, dim=1).sum() / logits.shape[0]


            self.log("val_loss", loss, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
            self.log("val_perplexity", torch.exp(loss), sync_dist=False, prog_bar=True, batch_size=logits.shape[0], add_dataloader_idx=False)
            self.log("val_partner_prediction_loss", partner_prediction_loss, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
            self.log("val_peptide_prediction_loss", peptide_prediction_loss, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
            self.log("val_partner_perplexity", torch.exp(partner_prediction_loss), sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
            self.log("val_peptide_perplexity", torch.exp(peptide_prediction_loss), sync_dist=True, prog_bar=True, batch_size=logits.shape[0], add_dataloader_idx=False)
            self.log("val_partner_accuracy", partner_accuracy, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
            self.log("val_peptide_accuracy", peptide_accuracy, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
            self.log("val_partner_top10p", partner_topk_accuracy, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
            self.log("val_peptide_top10p", peptide_topk_accuracy, sync_dist=True, prog_bar=True, batch_size=logits.shape[0], add_dataloader_idx=False)
            self.log("val_peptide_mrr", peptide_mrr, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
            self.log("val_partner_mrr", partner_mrr, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)

        else:
            # Given a protein, predict the correct peptide out of 2
            logits = self(
                batch['peptide_input'], 
                batch['partner_input'], 
                peptide_padding_mask=batch['peptide_padding_mask'],
                partner_padding_mask=batch['partner_padding_mask']
            )

            binary_cross_entropy = F.cross_entropy(logits.T, batch['labels'])

            binary_predictions = logits.argmax(dim=0)
            binary_accuracy = binary_predictions.eq(batch['labels']).float().mean()

            self.log("binary_loss", binary_cross_entropy, sync_dist=True, prog_bar=False, batch_size=2, add_dataloader_idx=False)
            self.log("binary_accuracy", binary_accuracy, sync_dist=False, prog_bar=True, batch_size=2, add_dataloader_idx=False)




    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer