import logging

import luigi
import os
from pathlib import Path

from util import DockerTask

VERSION = os.getenv('PIPELINE_VERSION', '0.1')


class Debug(DockerTask):
    """Use this task with appropriate image to debug things."""

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'sleep', '3600'
        ]


class DownloadData(DockerTask):
    """Initial pipeline task downloads dataset."""

    fname = luigi.Parameter(default='wine_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/raw/')
    url = luigi.Parameter(
        default='https://github.com/datarevenue-berlin/code-challenge-2019/'
                'releases/download/0.1.0/dataset_sampled.csv'
    )

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'download_data.py',
            '--name', self.fname,
            '--url', self.url,
            '--out-dir', self.out_dir
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return luigi.LocalTarget(
            path=str(out_dir/f'{self.fname}.csv')
        )


class MakeDatasets(DockerTask):

    in_csv = luigi.Parameter(default='/usr/share/data/raw/wine_dataset.csv')
    out_dir = luigi.Parameter(default='/usr/share/data/processed/')

    @property
    def image(self):
        return f'code-challenge/make-dataset:{VERSION}'

    def requires(self):
        return DownloadData()

    @property
    def command(self):
        return [
            'python', 'make_dataset.py',
            '--in-csv', self.in_csv,
            '--out-dir', self.out_dir,
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS-split')
        )


class CleanDataset(DockerTask):

    in_train = luigi.Parameter(default='/usr/share/data/processed/train.csv')
    in_test = luigi.Parameter(default='/usr/share/data/processed/test.csv')
    out_dir = luigi.Parameter(default='/usr/share/data/processed/')

    @property
    def image(self):
        return f'code-challenge/clean-dataset:{VERSION}'

    def requires(self):
        return MakeDatasets()

    @property
    def command(self):
        return [
            'python', 'clean_dataset.py',
            '--in-train', self.in_train,
            '--in-test', self.in_test,
            '--out-dir', self.out_dir,
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS-clean')
        )


class ExtractFeatures(DockerTask):

    in_train = luigi.Parameter(
        default='/usr/share/data/processed/train_cleaned.csv')
    in_test = luigi.Parameter(
        default='/usr/share/data/processed/test_cleaned.csv')
    out_dir = luigi.Parameter(default='/usr/share/data/processed/')

    @property
    def image(self):
        return f'code-challenge/extract-features:{VERSION}'

    def requires(self):
        return CleanDataset()

    @property
    def command(self):
        return [
            'python', 'extract_features.py',
            '--in-train', self.in_train,
            '--in-test', self.in_test,
            '--out-dir', self.out_dir,
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS-extract')
        )


class TrainModel(DockerTask):

    in_csv = luigi.Parameter(
        default='/usr/share/data/processed/train_ready.csv')
    filename = luigi.Parameter(default='model_pickle')
    out_dir = luigi.Parameter(default='/usr/share/data/output/')

    @property
    def image(self):
        return f'code-challenge/train-model:{VERSION}'

    def requires(self):
        return ExtractFeatures()

    @property
    def command(self):
        return [
            'python', 'train_model.py',
            '--in-csv', self.in_csv,
            '--filename', self.filename,
            '--out-dir', self.out_dir
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return luigi.LocalTarget(
            path=str(out_dir/f'{self.filename}.pckl')
        )


class EvaluateModel(DockerTask):

    in_csv = luigi.Parameter(
        default='/usr/share/data/processed/test_ready.csv')
    in_model = luigi.Parameter(
        default='/usr/share/data/output/model_pickle.pckl')
    out_dir = luigi.Parameter(default='/usr/share/data/output/')

    @property
    def image(self):
        return f'code-challenge/evaluate-model:{VERSION}'

    def requires(self):
        return TrainModel()

    @property
    def command(self):
        return [
            'python', 'evaluate_model.py',
            '--in-csv', self.in_csv,
            '--in-model', self.in_model,
            '--out-dir', self.out_dir,
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS-evaluate')
        )
