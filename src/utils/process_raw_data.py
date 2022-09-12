from typing import Sequence

import pandas as pd


def encode_categorical_clmns(df: pd.DataFrame,
                             label_encoder,
                             encoding_cols: Sequence[str] = ('store_nbr', 'family')):
    encoding_cols = list(encoding_cols)
    for encoding_col in encoding_cols:
        df[encoding_col + '_enc'] = label_encoder.fit_transform(
            df[encoding_col]
        )
    return df


def create_date_features(df: pd.DataFrame,
                         date_col: str = 'date',
                         qty_col: str = 'sales',
                         transaction_col: str = 'transactions',
                         lags: int = 16,
                         encoding_cols: Sequence[str] = ('store_nbr', 'family')):
    """
    Given dataframe of time series, create features for modeling
    """
    df_copy = df.copy()
    encoding_cols = list(encoding_cols)
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], format='%Y-%m-%d')
    df_copy["is_weekend"] = df_copy[date_col].dt.dayofweek > 4
    df_copy['day'] = df_copy[date_col].dt.day
    df_copy['week'] = df_copy[date_col].dt.isocalendar().week.astype(int)
    df_copy['month'] = df_copy[date_col].dt.month
    df_copy['year'] = df_copy[date_col].dt.year

    df_copy.set_index(date_col, inplace=True)
    df_copy = df_copy.sort_index()

    for i in [7, 14, 30]:  # lags in days for rolling functions

        df_copy['rolling_perc10_' + str(i)] = df_copy.groupby(
            encoding_cols
        )[qty_col].transform(lambda x: x.shift(lags).rolling(i).quantile(0.1))

        df_copy['rolling_perc90_' + str(i)] = df_copy.groupby(
            encoding_cols
        )[qty_col].transform(lambda x: x.shift(lags).rolling(i).quantile(0.9))

        df_copy['rolling_median_' + str(i)] = df_copy.groupby(
            encoding_cols
        )[qty_col].transform(lambda x: x.shift(lags).rolling(i).median())

        df_copy['rolling_mean_' + str(i)] = df_copy.groupby(
            encoding_cols
        )[qty_col].transform(lambda x: x.shift(lags).rolling(i).mean())

        df_copy['rolling_mean_' + str(i)] = df_copy.groupby(
            encoding_cols
        )[qty_col].transform(lambda x: x.shift(lags).rolling(i).mean())

        df_copy['rolling_std_' + str(i)] = df_copy.groupby(
            encoding_cols
        )[qty_col].transform(lambda x: x.shift(lags).rolling(i).std())

        df_copy['year_rolling_mean_' + str(i)] = df_copy.groupby(
            encoding_cols
        )[qty_col].transform(lambda x: x.shift(365).rolling(i, center=True).mean())

        df_copy['year_rolling_std_' + str(i)] = df_copy.groupby(
            encoding_cols
        )[qty_col].transform(lambda x: x.shift(365).rolling(i, center=True).std())

    # create lag features
    lag_columns = []
    for lag in range(lags):
        df_copy[f'sales_lag{lags + lag}'] = df_copy.groupby(
            encoding_cols
        )[qty_col].shift(lags + lag)

        df_copy[f'transaction_lag{lags + lag}'] = df_copy.groupby(
            encoding_cols
        )[transaction_col].shift(lags + lag)

        lag_columns.append(f'sales_lag{lags + lag}')
    df_copy.dropna(subset=lag_columns, inplace=True)

    return df_copy, lag_columns
