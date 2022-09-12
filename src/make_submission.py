import sys
import os
from typing import Tuple

from tqdm import tqdm
import pickle

import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer

from rich.console import Console

from settings import PROC_DATA_PATH, SUBMISSION_PATH, CHECK_POINT_PATH


def read_data_pickle(pickle_file_name: str) -> Tuple[dict, pd.DataFrame]:
    with open(os.path.join(
            PROC_DATA_PATH,
            pickle_file_name  # 'data_main_v3.pickle'
    ), 'rb') as handle:
        d_data = pickle.load(handle)

    df_raw_submission = d_data['df_submission']
    return d_data, df_raw_submission


def prep_data_for_prediction(d_data: dict, max_encoder_length: int = 90) -> pd.DataFrame:
    df_train = d_data['df_train']
    df_test = d_data['df_test']

    encoder_data = df_train[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]
    data_for_prediction = pd.concat([df_test, encoder_data]).sort_values('time_idx', ignore_index=True)

    return data_for_prediction


def get_raw_predictions(
        data_for_prediction: pd.DataFrame,
        check_point_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    best_tft = TemporalFusionTransformer.load_from_checkpoint(
        os.path.join(CHECK_POINT_PATH, check_point_name)
    )
    raw_preds, idx_s = best_tft.predict(data_for_prediction, mode="prediction", return_index=True)

    return raw_preds, idx_s


def get_predictions(
        raw_preds: pd.DataFrame,
        idx_s: pd.DataFrame,
        data_for_prediction: pd.DataFrame,
        prediction_horizon: int = 16
) -> pd.DataFrame:
    df_pred_list = []
    for idx, row in tqdm(idx_s.iterrows()):
        family_enc = row.family_enc
        store_nbr_enc = row.store_nbr_enc

        pred = raw_preds[idx, :]

        df_id_pred = (
            data_for_prediction
            .query('family_enc == @family_enc & store_nbr_enc == @store_nbr_enc')
            [['date', 'family_enc', 'store_nbr_enc']]
            .drop_duplicates()[-prediction_horizon:]
            .assign(pred=pred[-prediction_horizon:])  # last 16 values
        )
        df_pred_list.append(df_id_pred)

    return pd.concat(df_pred_list, ignore_index=True)


def get_submission(
        df_predictions: pd.DataFrame,
        d_data: dict,
        df_raw_submission: pd.DataFrame
) -> pd.DataFrame:

    df_submission = (
        df_raw_submission
        .assign(family_enc=lambda df_: df_['family'].map(d_data['d_family']))
        .assign(family_enc=lambda df_: df_['family_enc'].astype(str))
        .assign(store_nbr_enc=lambda df_: df_['store_nbr'].map(d_data['d_store']))
        .assign(store_nbr_enc=lambda df_: df_['store_nbr_enc'].astype(str))
        .merge(df_predictions, on=['date', 'family_enc', 'store_nbr_enc'], how='left')
        [['id', 'pred']]
        .rename(columns={'pred': 'sales'})
    )
    return df_submission


def save_submission(
        df_submission: pd.DataFrame,
        submision_file_name: str
):
    df_submission.to_csv(os.path.join(SUBMISSION_PATH, submision_file_name), index=False)


def main(check_point_name: str, submission_file_name: str):
    console = Console()
    console.log(f"[blue]Submission for {check_point_name}[/blue]")
    start_msg = (
        f"[bold green]Create {submission_file_name} file\n"
    )
    with console.status(start_msg, spinner="aesthetic") as status:
        d_data, df_raw_submission = read_data_pickle(pickle_file_name='data_main_v3.pickle')
        data_for_prediction = prep_data_for_prediction(d_data)
        console.log("[green]Finish prepare data for prediction")

        status.update("[blue]Start make raw predictions")
        raw_preds, idx_s = get_raw_predictions(data_for_prediction, check_point_name)
        df_predictions = get_predictions(raw_preds, idx_s, data_for_prediction)
        console.log("[green]Finish prepare predictions")

        df_submission = get_submission(df_predictions, d_data, df_raw_submission)
        save_submission(df_submission, submission_file_name)
        console.log("[green]Save result[/green]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_point_name = sys.argv[1]
        submission_file_name = sys.argv[2]
    else:
        check_point_name = 'please_specify_check_point'
        submission_file_name = 'submission.csv'
    main(check_point_name=check_point_name, submission_file_name=submission_file_name)

