import sys
import os
from typing import Tuple, List

import pickle

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from rich.console import Console

from dataset import TFTDataset
from settings import CHECK_POINT_PATH, LOGS_PATH,\
    MAX_EPOCHS, LIMIT_TRAIN_BATCHES, LSTM_LAYERS, OUTPUT_SIZE, REDUCE_ON_PLATEAU_PATIENCE


# torch.backends.cudnn.enabled = False  # uncomment for monotonic constrains


def prep_callbacks(exp_n: str) -> Tuple[List, TensorBoardLogger]:
    """EarlyStopping, LearningRateMonitor, TensorBoardLogger, ModelCheckpoint"""

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=5,
        verbose=False,
        mode="min"
    )

    log_dir = os.path.join(LOGS_PATH, f'{exp_n}')
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger(log_dir)  # logging results to a tensorboard

    model_checkpoint_path = os.path.join(CHECK_POINT_PATH, f'{exp_n}')
    check_pointer = ModelCheckpoint(
        dirpath=model_checkpoint_path,
        filename='{epoch}-{val_loss:.4f}',
        save_top_k=2,
        save_on_train_epoch_end=False,
        monitor='val_loss',
        mode='min'
    )
    return [early_stop_callback, lr_logger, check_pointer], logger


def get_optuna_hp(exp_n: str) -> dict:
    result_file = os.path.join(
        CHECK_POINT_PATH, f"{exp_n}/optuna/results.pickle"
    )
    with open(os.path.join(result_file), 'rb') as handle:
        d_optuna = pickle.load(handle)
    best_params = d_optuna.best_trial.params
    return best_params


def init_trainer(callbacks, tb_logger, training, params):
    # configure network and trainer

    trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            weights_summary="top",
            gradient_clip_val=params['gradient_clip_val'],
            limit_train_batches=LIMIT_TRAIN_BATCHES,
            callbacks=callbacks,
            logger=tb_logger,
        )
    tft = TemporalFusionTransformer.from_dataset(
        training,
        lstm_layers=LSTM_LAYERS,
        learning_rate=params['learning_rate'],
        hidden_size=params['hidden_size'],
        attention_head_size=params['attention_head_size'],
        dropout=params['dropout'],
        hidden_continuous_size=params['hidden_continuous_size'],
        output_size=OUTPUT_SIZE,
        loss=QuantileLoss(),
        log_interval=0,
        reduce_on_plateau_patience=REDUCE_ON_PLATEAU_PATIENCE,
        # monotone_constaints=dict(  # uncomment for monotonic constrains
        #     onpromotion=1,
        #     transactions=1,
        #     dcoilwtico=1
        # )
    )

    return trainer, tft


def main(exp_n: str, use_optim_params: str):
    """Run training"""
    console = Console()

    tft_dataset = TFTDataset(data_pickle_filename='data_main_v3.pickle')
    start_msg = (
        f"[bold green]Start training ... exp = {exp_n}\n"
    )
    with console.status(start_msg, spinner="aesthetic") as status:

        status.update("[blue]Start prepare train/validate dataloaders[/blue]")
        train_dataloader, val_dataloader, training = tft_dataset.get_dataloaders()

        status.update("[blue]Prepare callbacks[/blue]")
        callbacks, tb_logger = prep_callbacks(exp_n)

        status.update("[blue]Prepare params[/blue]")
        if use_optim_params:
            params = get_optuna_hp(exp_n=exp_n)
        else:
            params = dict(
                gradient_clip_val=0.1,
                learning_rate=0.03,
                hidden_size=32,
                attention_head_size=7,
                dropout=0.2,
                hidden_continuous_size=32
            )
        status.update("[blue]Init trainer and tft model[/blue]")
        trainer, tft = init_trainer(callbacks, tb_logger, training, params)

        status.update("[blue]Start training[/blue]")
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    console.log("[green]End training[/green]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exp_n = sys.argv[1]
        use_optim_params = sys.argv[2]
    else:
        exp_n = "exp"
        use_optim_params = None
    main(exp_n=exp_n, use_optim_params=use_optim_params)



