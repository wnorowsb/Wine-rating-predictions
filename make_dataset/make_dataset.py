import click
import pandas as pd
from pathlib import Path
import logging


def _save_datasets(train, test, outdir: Path):
    """Save data sets into nice directory structure and write SUCCESS flag."""
    out_train = outdir / 'train.csv'
    out_test = outdir / 'test.csv'
    flag = outdir / '.SUCCESS-split'

    train.to_csv(str(out_train))
    test.to_csv(str(out_test))

    flag.touch()


@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
def make_datasets(in_csv, out_dir):
    log = logging.getLogger('make_datasets')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv, index_col="Unnamed: 0")
    log.info("File read")

    # Shuffle dataset
    shuffle_df = df.sample(frac=1)
    log.info("Data shuffled")

    # Define a size for train set
    train_size = int(0.8 * len(df))

    # Split dataset
    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]
    log.info("Datasets split")

    _save_datasets(train_set, test_set, out_dir)
    log.info("Data saved")


if __name__ == '__main__':
    make_datasets()
