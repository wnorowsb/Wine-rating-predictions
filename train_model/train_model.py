from pathlib import Path

import click
import logging
import pandas as pd
from sklearn.linear_model import Ridge
import pickle

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--in-csv')
@click.option('--filename')
@click.option('--out-dir')
def train_model(in_csv, filename, out_dir):
    """Trains model on prepared train dataset"""
    log = logging.getLogger('train-model')

    df = pd.read_csv(in_csv, index_col="Unnamed: 0")
    X_train = df.drop('points', axis=1)
    y_train = df['points']

    log.info("Data read")

    regression = Ridge()

    regression.fit(X_train, y_train)
    log.info("Model learned")

    out_path = Path(out_dir) / f'{filename}.pckl'
    pickle.dump(regression, open(out_path, 'wb'))
    log.info("Model saved")


if __name__ == '__main__':
    train_model()
