import os
from typing import Tuple

import pickle
import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from settings import PROC_DATA_PATH,\
    MAX_PREDICTION_LENGTH, MAX_ENCODING_LENGTH, BATCH_SIZE, NUM_WORKERS


class TFTDataset:

    def __init__(self,
                 data_pickle_filename: str,
                 max_encoding_length: int = MAX_ENCODING_LENGTH,
                 batch_size: int = BATCH_SIZE,
                 num_workers: int = NUM_WORKERS
                 ):
        self.data_pickle_filename = data_pickle_filename
        self.max_encoding_length = max_encoding_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _get_data_from_pickle(self) -> pd.DataFrame:
        """read data from file"""
        full_data_file_name = os.path.join(PROC_DATA_PATH, self.data_pickle_filename)

        with open(full_data_file_name, "rb") as handle:
            data = pickle.load(handle)
        return data

    def _prepare_data_for_dataloader(self, data) -> Tuple[pd.DataFrame, TimeSeriesDataSet]:
        """Prepare optimization proces"""

        df = data["df_train"]
        group_ids = data["group_ids"]
        static_categoricals = data["static_categoricals"]
        time_varying_known_categoricals = data["time_varying_known_categoricals"]
        time_varying_known_reals = data["time_varying_known_reals"]
        time_varying_unknown_reals = data["time_varying_unknown_reals"]
        training_cutoff = df["time_idx"].max() - MAX_PREDICTION_LENGTH

        training = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="sales",
            group_ids=group_ids,
            min_encoder_length=self.max_encoding_length // 2,
            max_encoder_length=self.max_encoding_length,
            max_prediction_length=MAX_PREDICTION_LENGTH,
            static_categoricals=static_categoricals,
            static_reals=[],
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(
                groups=group_ids, transformation='softplus'
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

        return df, training

    def _prepare_train_val_dataset(self, df, training):
        """prepare train validate dataset"""
        validation = TimeSeriesDataSet.from_dataset(
            training, df, predict=True, stop_randomization=True
        )
        train_dataloader = training.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        val_dataloader = validation.to_dataloader(
            train=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        return train_dataloader, val_dataloader

    def get_dataloaders(self):
        data = self._get_data_from_pickle()
        df, training = self._prepare_data_for_dataloader(data)
        train_dataloader, val_dataloader = self._prepare_train_val_dataset(df, training)
        return train_dataloader, val_dataloader, training

