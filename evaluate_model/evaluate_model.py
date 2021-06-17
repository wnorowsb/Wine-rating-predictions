from pathlib import Path
import click
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import json

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--in-csv')
@click.option('--in-model')
@click.option('--out-dir')
def evaluate_model(in_csv, in_model, out_dir):
    """Outputs metrics and plots basen on a given model"""
    log = logging.getLogger('evaluate-model')
    out_dir = Path(out_dir)
    df = pd.read_csv(in_csv, index_col="Unnamed: 0")
    X = df.drop('points', axis=1)
    y = df['points']

    log.info("Data read")

    regression = pickle.load(open(in_model, 'rb'))

    y_pred = regression.predict(X)
    error = y - y_pred

    metrics = {}
    metrics['mean_absoulte_error'] = mean_absolute_error(y_pred, y)
    metrics['mean_squared_error'] = mean_squared_error(y_pred, y)
    metrics['root_mean_squared_error'] = mean_squared_error(
        y_pred, y, squared=False)

    metrics_file = open(out_dir / 'metrics.json', "w")
    json.dump(metrics, metrics_file)
    metrics_file.close()

    # Plot error distribution
    error_dist = sns.displot(error)
    error_dist.set_xticks(range(-7, 7, 1))
    error_dist.set(xlabel='error', ylabel='count')
    error_dist.get_figure().savefig(out_dir / 'error_dist.png')

    plt.clf()

    # Boxplot the error
    box = sns.boxplot(x=error.abs())
    box.set(xlabel='error')
    box.get_figure().savefig(out_dir / 'boxplot_error.png')

    plt.clf()

    # Compare predicted and true value distribution
    fig, ax = plt.subplots()
    sns.displot(y, bins=40, ax=ax)
    sns.displot(y_pred, bins=40, ax=ax)
    fig.legend(labels=['true', 'predicted'], bbox_to_anchor=(0.85, 0.75))
    ax.set(xlabel='points', ylabel='count')
    plt.savefig(out_dir / 'predicted_and_true.png')

    flag = out_dir / '.SUCCESS-evaluate'
    flag.touch()

    log.info("Evaluation saved")


if __name__ == '__main__':
    evaluate_model()
