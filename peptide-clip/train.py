"""
Script for hyperparameter search for training
"""

import randomname

import torch
import optuna
from model import PeptideCLIP
from data import PartnerPeptideDataModule
import os
import pytorch_lightning as pl
import argparse
import pickle
from pytorch_lightning.loggers import WandbLogger
import json

import torch.multiprocessing

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

# random name and submit script modified from https://github.com/AllanSCosta/InterDocker/blob/main/utils.py

def random_name():
    return randomname.get_name(
        adj=('speed', 'emotions', 'temperature', 'weather', 'character', 'algorithms', 'geometry', 'complexity', 'physics', 'shape', 'taste', 'colors', 'size', 'appearance'),
        noun=('astronomy', 'set_theory', 'military_navy', 'infrastructure', 'chemistry', 'physics', 'algorithms', 'geometry', 'coding', 'architecture', 'metals', 'apex_predators')
    )

def submit_script(script_path, base_path, config):
    model_id = config['model_id']
    worskpace_dir = os.path.join(base_path, 'scripts')
    os.makedirs(worskpace_dir, exist_ok=True)
    script = os.path.join(worskpace_dir, f'{model_id}.sh')
    with open(script, 'w') as file:
        preamble = f'#!/bin/bash\n'
        preamble += f'#SBATCH --gres=gpu:volta:1\n'
        preamble += f'#SBATCH -o {os.path.join(worskpace_dir, model_id)}.sh.log\n'
        preamble += f'#SBATCH --cpus-per-task=10\n'
        preamble += f'#SBATCH --job-name={model_id}\n\n'
        preamble += f'module load anaconda/2021b\n'
        file.write(preamble)
        config = [(key, value) for key, value in config.items() if (key != 'submit')]
        config_strings = []
        for key, value in config:
            if type(value) == bool and value:
                config_strings.append(f'--{key}')
            elif type(value) != bool and type(value) is not None:
                config_strings.append(f'--{key} {str(value)}')

        config_string = ' '.join(config_strings)
        file.write(f'python -u {script_path} {config_string}')
        file.close()
    os.system(f'LLsub {script}')
    print(f'submitted {model_id}!')

def model_init(trial, args):
    config = vars(args).copy()

    config['lr'] = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    # config['peptide_transformer_layers'] = trial.suggest_int('peptide_transformer_layers', 0, 1, step=1)
    config['peptide_transformer_layers'] = 0
    config['peptide_transformer_heads'] = trial.suggest_int('peptide_transformer_heads', 2, 4, step=2)
    config['peptide_length_reducer'] = trial.suggest_categorical("peptide_length_reducer", ["avg"])

    config['partner_transformer_layers'] = trial.suggest_int('partner_transformer_layers', 0, 1, step=1)
    config['partner_transformer_heads'] = 2

    # config['partner_transformer_heads'] = trial.suggest_int('partner_transformer_heads', 2, 4, step=2)
    config['partner_length_reducer'] = trial.suggest_categorical("partner_length_reducer", ["avg"])

    config['embedding_dim'] = trial.suggest_int('hidden_size', 256, 512, step=32)

    config['peptide_sequence_mlp_layers'] = trial.suggest_int('peptide_sequence_mlp_layers', 2, 4, step=1)
    if config['partner_transformer_layers'] == 1:
        config['partner_sequence_mlp_layers'] = 1
    else:
        config['partner_sequence_mlp_layers'] = trial.suggest_int('partner_sequence_mlp_layers', 2, 4, step=1)

    config['peptide_embedding_mlp_layers'] = trial.suggest_int('peptide_embedding_mlp_layers', 2, 4, step=1)
    config['partner_embedding_mlp_layers'] = trial.suggest_int('partner_embedding_mlp_layers', 2, 4, step=1)
    config['sequence_dropout'] = trial.suggest_float("sequence_dropout", 0., 0.7)
    config['embedding_dropout'] = trial.suggest_float("embedding_dropout", 0.1, 0.85)
    config['layer_norm_eps'] = trial.suggest_float("layer_norm_eps", 1e-6, 1e-3, log=True)

    config['learn_temperature'] = True

    return PeptideCLIP(config), config

def objective(trial, args, model_id):
    model, config = model_init(trial, args)
    print(config)

    callbacks = []
    if args.prune_optuna:
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_loss"))
    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=args.early_stopping_patience, min_delta=1e-1))
    
    save_dir = os.path.join("optuna_models", model_id, str(trial.number))
    callbacks.append(ModelCheckpoint(monitor="val_loss",  save_top_k=args.save_top_k, dirpath=os.path.join(save_dir, "top_models")))
    datamodule = PartnerPeptideDataModule(args.regions_csv_path, 
                                args.peptide_embeddings_path,
                                args.partner_embeddings_path, 
                                batch_size=args.batch_size,
                                use_interface=config["use_interface"])  

    wandb_logger = WandbLogger(project="peptide-clip", name=f"{model_id}_{trial.number}", log_model=True)
    trainer = pl.Trainer(logger=wandb_logger,
                         max_epochs=args.max_epochs,
                         gpus=args.n_gpus,
                         precision=args.precision,
                         enable_checkpointing=True,
                         callbacks=callbacks,
                         log_every_n_steps=5,
                         default_root_dir=save_dir,
                         detect_anomaly=args.anomaly_detection,
                         min_epochs=3)
    
    
    wandb_logger.log_hyperparams(config)
    wandb_logger.watch(model)
    wandb_logger.log_text(key="hyperparameters", columns=["key", "value"], data=[[str(k), str(v)] for k, v in config.items()])
    trainer.fit(model, datamodule=datamodule)  
    torch.cuda.empty_cache()
    
    print("    Trial {} validation loss: {}".format(trial.number, trainer.callback_metrics["val_loss"].item()))
    print("    Trial {} validation accuracy: {}".format(trial.number, trainer.validate(dataloaders=datamodule.val_dataloader())))

    wandb.finish()

    return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train peptide-clip')
    parser.add_argument('--regions_csv_path', dest='regions_csv_path', default="pepgen_dataset.csv")
    parser.add_argument('--peptide_embeddings_path', dest='peptide_embeddings_path', default="peptide_embeddings.pkl")
    parser.add_argument('--partner_embeddings_path', dest='partner_embeddings_path', default="partner_embeddings.pkl")
    parser.add_argument('--random_seed', dest='random_seed', type=int, default=42)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=256)
    parser.add_argument('--model_id', dest='model_id', default=None)
    parser.add_argument('--peptide_input_dim', dest='peptide_input_dim', type=int, default=1280)
    parser.add_argument('--partner_input_dim', dest='partner_input_dim', type=int, default=768)
    parser.add_argument('--use_interface', action='store_true')
    parser.add_argument('--n_gpus', dest='n_gpus', type=int, default=1)    
    parser.add_argument('--precision', dest='precision', type=int, default=32)  
    parser.add_argument('--max_epochs', dest='max_epochs', type=int, default=50)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--early_stopping_patience', dest='early_stopping_patience', type=int, default=10)

    parser.add_argument('--anomaly_detection', action='store_true')

    parser.add_argument('--optuna', dest='optuna', action="store_true")
    parser.add_argument('--optuna_trials', dest='optuna_trials', type=int, default=100)
    parser.add_argument('--prune_optuna', action='store_true')

    parser.add_argument('--lr', dest="lr", default=1e-3, type=float)

    parser.add_argument('--peptide_sequence_mlp_layers', dest="peptide_sequence_mlp_layers", type=int, default=2)
    parser.add_argument('--partner_sequence_mlp_layers', dest="partner_sequence_mlp_layers", type=int, default=2)

    parser.add_argument('--peptide_transformer_heads', dest="peptide_transformer_heads", type=int, default=2)
    parser.add_argument('--peptide_transformer_layers', dest="peptide_transformer_layers", type=int, default=0)
    parser.add_argument('--partner_transformer_heads', dest="partner_transformer_heads", type=int, default=2)
    parser.add_argument('--partner_transformer_layers', dest="partner_transformer_layers", type=int, default=0)

    parser.add_argument('--peptide_length_reducer', dest='peptide_length_reducer', default="avg")
    parser.add_argument('--partner_length_reducer', dest='partner_length_reducer', default="avg")

    parser.add_argument('--embedding_dim', dest="embedding_dim", type=int, default=512)
    parser.add_argument('--peptide_embedding_mlp_layers', dest="peptide_embedding_mlp_layers", type=int, default=2)
    parser.add_argument('--partner_embedding_mlp_layers', dest="partner_embedding_mlp_layers", type=int, default=2)

    parser.add_argument('--layer_norm_eps', dest="layer_norm_eps", type=float, default=1e-5)
    parser.add_argument('--sequence_dropout', dest="sequence_dropout", type=float, default=1e-1)
    parser.add_argument('--embedding_dropout', dest="embedding_dropout", type=float, default=1e-1)
    parser.add_argument('--temperature', dest="temperature", type=float, default=0.07)
    parser.add_argument('--learn_temperature', dest='learn_temperature', action='store_true')
    parser.add_argument('--fast_dev_run', dest='fast_dev_run', action='store_true')
    parser.add_argument('--resume_checkpoint', dest='resume_checkpoint', default=None)

    parser.add_argument('--save_top_k', dest='save_top_k', default=5, type=int)

    parser.add_argument('--submit', dest='submit', action='store_true')

    args, _ = parser.parse_known_args()
    config = vars(args)

    if args.model_id == None:
        config['model_id'] = random_name()
    else:
        config['model_id'] = args.model_id

    if args.submit:
        submit_script(os.path.realpath(__file__), os.getcwd(), config)
        exit()
    
    print(f"Training {config['model_id']}!")

    if args.use_interface:
        args.partner_input_dim += 1
    
    if args.n_gpus > 1:
        # super slow on supercloud
        torch.multiprocessing.set_sharing_strategy('file_system')

    if args.optuna:
        if args.prune_optuna:
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=10, n_startup_trials=10)
        else:
            pruner = None
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(lambda trial: objective(trial, args, config['model_id']),
                         n_trials=args.optuna_trials)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        datamodule = PartnerPeptideDataModule(args.regions_csv_path, 
                                        args.peptide_embeddings_path,
                                        args.partner_embeddings_path, 
                                        batch_size=args.batch_size,
                                        use_interface=args.use_interface)  

        if args.fast_dev_run:
            fast_dev_run = 5
        else:
            fast_dev_run = False

        root_dir = os.path.join("manual_models", config['model_id'])
        os.makedirs(root_dir, exist_ok=True)

        with open(os.path.join(root_dir, "config.pkl"), "wb") as f:
            pickle.dump(config, f)

        if args.early_stopping:
            callbacks = [EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=args.early_stopping_patience, min_delta=1e-1)]
        else:
            callbacks = []
        callbacks.append(ModelCheckpoint(monitor="val_loss", save_top_k=args.save_top_k, dirpath=os.path.join(root_dir, "top_models")))

        wandb_logger = WandbLogger(project="peptide-clip", name=args.model_id, log_model=True)
        print(config)
        trainer = pl.Trainer(logger=wandb_logger,
                            max_epochs=args.max_epochs,
                            precision=args.precision,
                            gpus=args.n_gpus,
                            enable_checkpointing=True,
                            callbacks=callbacks,
                            default_root_dir=root_dir,
                            fast_dev_run=args.fast_dev_run,
                            detect_anomaly=args.anomaly_detection,
                            log_every_n_steps=5,
                            check_val_every_n_epoch=1,
                            gradient_clip_val=0.5)


        model = PeptideCLIP(config)
        wandb_logger.log_hyperparams(config)
        wandb_logger.watch



        
        wandb_logger.log_text(key="hyperparameters", columns=["key", "value"], data=[[str(k), str(v)] for k, v in config.items()])

        trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_checkpoint)
