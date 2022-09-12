import gc
import os
import pickle
import sys

import torch
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

from rich.console import Console

from dataset import TFTDataset
from settings import CHECK_POINT_PATH, LOGS_PATH,\
    N_TRIALS, MAX_EPOCHS_TUNE, LIMIT_TRAIN_BATCHES, LSTM_LAYERS, REDUCE_ON_PLATEAU_PATIENCE


def clean_cuda() -> bool:
    """Clean gpu before starting"""
    torch.cuda.empty_cache()
    gc.collect()
    return True


def tuning_params(train_dataloader, val_dataloader, exp_n, accelerator):
    """Optimize"""

    # create study
    model_path = os.path.join(
        CHECK_POINT_PATH, f"{exp_n}/optuna/model_check_points"
    )
    log_dir = os.path.join(LOGS_PATH, f"{exp_n}")

    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path=model_path,
        log_dir=log_dir,
        n_trials=N_TRIALS,
        max_epochs=MAX_EPOCHS_TUNE,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 8),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(
            limit_train_batches=LIMIT_TRAIN_BATCHES,
            accelerator=accelerator
        ),
        lstm_layers=LSTM_LAYERS,
        reduce_on_plateau_patience=REDUCE_ON_PLATEAU_PATIENCE,
        use_learning_rate_finder=False,
    )
    return study


def save_result(result_to_save, exp_n):
    """save study results - also we can resume tuning at a later point in time"""
    result_file = os.path.join(
        CHECK_POINT_PATH, f"{exp_n}/optuna/results.pickle"
    )
    with open(result_file, "wb") as fout:
        pickle.dump(result_to_save, fout)


def main(exp, accelerator):
    """Run tuning process"""
    console = Console()
    data_pickle_filename = 'data_main_v3.pickle'

    tft_dataset = TFTDataset(data_pickle_filename=data_pickle_filename)

    console.log(f"[blue]Tuning experiment witn file = {data_pickle_filename}[/blue]")
    start_msg = (
        f"[bold green]Tuning data by optuna to kaggle competition Favorita, exp = {exp}\n"
    )
    with console.status(start_msg, spinner="aesthetic") as status:

        clean_cuda()
        console.log("[green]Finish clean cuda[/green]")
        train_dataloader, val_dataloader, _ = tft_dataset.get_dataloaders()
        console.log("[green]Finish prepare train/validate datasets[/green]")

        status.update("[blue]Start tuning data[/blue]")
        result = tuning_params(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            exp_n=exp,
            accelerator=accelerator
        )
        console.log("[green]Finish tuning data[/green]")

        save_result(result_to_save=result, exp_n=exp)
        console.log("[green]Save result[/green]")
    console.log("[green]End tuning process![/green]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
        accelerator = sys.argv[2]
    else:
        exp_name = "exp"
        accelerator = 'cpu'
    main(exp=exp_name, accelerator=accelerator)
