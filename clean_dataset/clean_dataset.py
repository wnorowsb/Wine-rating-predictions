import click
import pandas as pd
from pathlib import Path
import logging


def _save_datasets(train, test, outdir: Path):
    """Save data sets into nice directory structure and write SUCCESS flag."""
    out_train = outdir / 'train_cleaned.csv'
    out_test = outdir / 'test_cleaned.csv'
    flag = outdir / '.SUCCESS-clean'

    train.to_csv(str(out_train))
    test.to_csv(str(out_test))

    flag.touch()


@click.command()
@click.option('--in-train')
@click.option('--in-test')
@click.option('--out-dir')
def clean_dataset(in_train, in_test, out_dir):
    """Handles None values in train and test datasets"""
    log = logging.getLogger('clean-data')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(in_train, index_col="Unnamed: 0")
    df_test = pd.read_csv(in_test, index_col="Unnamed: 0")
    log.info("File read")

    # drop rows with None values in selected columns
    df_train = df_train.dropna(
        subset=['country', 'province', 'price', 'description', 'winery'])
    df_test = df_test.dropna(
        subset=['country', 'province', 'price', 'description', 'winery'])

    df_train = df_train[df_train['price'] < 200]
    df_test = df_test[df_test['price'] < 200]

    # drop columns which does not bring enough value for model
    df_train = df_train.drop(
        ["designation", "region_2", "taster_twitter_handle"], axis=1)
    df_test = df_test.drop(
        ["designation", "region_2", "taster_twitter_handle"], axis=1)

    # replace None values witn Unknown in selected columns
    df_train[['region_1', 'taster_name']] = df_train[[
        'region_1', 'taster_name']].fillna('Unknown')
    df_test[['region_1', 'taster_name']] = df_test[[
        'region_1', 'taster_name']].fillna('Unknown')
    log.info("Data cleaned")

    _save_datasets(df_train, df_test, out_dir)
    log.info("Data saved")


if __name__ == '__main__':
    clean_dataset()
