import click
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re
from nltk.stem.porter import PorterStemmer
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)


def _save_datasets(train, test, outdir: Path):
    """Save data sets into nice directory structure and write SUCCESS flag."""
    out_train = outdir / 'train_ready.csv'
    out_test = outdir / 'test_ready.csv'
    flag = outdir / '.SUCCESS-extract'

    train.to_csv(str(out_train))
    test.to_csv(str(out_test))

    flag.touch()


def extract_year(column):
    """Extracts year from every string in given Series

    Parameters
    ----------
    column: Series
        Pandas Series with strings

    Returns
    -------
    yearSearch: Series
        Pandas Series with years extracted from strings in according index
    """
    yearSearch = []
    for value in column:
        regexresult = re.search(r'19\d{2}|20\d{2}', value)
        if regexresult:
            yearSearch.append(int(regexresult.group()))
        else:
            yearSearch.append(None)
    return yearSearch


def encode_ordinal(train, test, features):
    """Appliest Ordinal Encoding to train and test datasets

    Parameters
    ----------
    train: DataFrame
    test: DataFrame
    features: List<string>
        list of column names which should be OrdinalEncoded

    Returns
    -------
    encoded_train: DataFrame
        Inputed Dataset with encoded values
    encoded_test: DataFrame
        Inputed Dataset with encoded values
    """
    ore = OrdinalEncoder(cols=features).fit(train)
    encoded_train = ore.transform(train)
    encoded_test = ore.transform(test)
    return encoded_train, encoded_test


def encode_onehot(train, test, features):
    """Appliest One Hot Encoding to train and test datasets

    Parameters
    ----------
    train: DataFrame
    test: DataFrame
    features: List<string>
        list of column names from which OneHot features should be extracted

    Returns
    -------
    encoded_train: DataFrame
        Inputed Dataset with encoded values
    encoded_test: DataFrame
        Inputed Dataset with encoded values
    """
    ohe = OneHotEncoder(cols=features, use_cat_names=False).fit(train)
    encoded_train = ohe.transform(train)
    encoded_test = ohe.transform(test)
    return encoded_train, encoded_test


def clean(text, stemmer):
    """Cleans text by eg. removing prefixes and suffixes, uses given stemmer"""
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    text = [stemmer.stem(word) for word in text.split()]
    text = ' '.join(text)
    return text


def extract_words(train, test):
    """Adds Tf-idf vectors to train and test datasets"""
    tfidf = TfidfVectorizer(stop_words=['english'])
    extracted_train = tfidf.fit_transform(
        train['description'].values.astype('U'))
    df1 = pd.DataFrame(extracted_train.toarray(),
                       columns=tfidf.get_feature_names())
    train.reset_index(drop=True, inplace=True)
    train = pd.concat([train, df1], axis=1)

    extracted_test = tfidf.transform(test['description'].values.astype('U'))
    df2 = pd.DataFrame(extracted_test.toarray(),
                       columns=tfidf.get_feature_names())
    test.reset_index(drop=True, inplace=True)
    test = pd.concat([test, df2], axis=1)
    return train, test


@click.command()
@click.option('--in-train')
@click.option('--in-test')
@click.option('--out-dir')
def extract_features(in_train, in_test, out_dir):
    """Extracts features from text columns in train and test datasets."""
    log = logging.getLogger('extract-features')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(in_train, index_col="Unnamed: 0")
    df_test = pd.read_csv(in_test, index_col="Unnamed: 0")

    log.info("File read")

    df_train['year'] = extract_year(df_train['title'])
    df_test['year'] = extract_year(df_test['title'])

    df_train = df_train.dropna(subset=['year'])
    df_test = df_test.dropna(subset=['year'])

    df_train = df_train[df_train['year'] >= 1990]
    df_test = df_test[df_test['year'] >= 1990]

    log.info("Year extracted")

    ordinal_features = ['province', 'region_1', 'variety', 'winery']
    one_hot_features = ['country', 'taster_name', 'year']
    df_train, df_test = encode_ordinal(df_train, df_test, ordinal_features)
    df_train, df_test = encode_onehot(df_train, df_test, one_hot_features)

    log.info("Categories encoded")

    ps = PorterStemmer()
    df_train['description'] = df_train['description'].apply(clean, args=(ps,))
    df_test['description'] = df_test['description'].apply(clean, args=(ps,))

    df_train, df_test = extract_words(df_train, df_test)

    log.info("Words extracted")

    df_train['price'] = np.sqrt(df_train['price'])
    df_test['price'] = np.sqrt(df_test['price'])
    df_train = df_train.rename(columns={'price': 'sqrt_price'})
    df_test = df_test.rename(columns={'price': 'sqrt_price'})

    # drop text Series after features were extracted from them
    df_train = df_train.drop(['description', 'title'], axis=1)
    df_test = df_test.drop(['description', 'title'], axis=1)

    _save_datasets(df_train, df_test, out_dir)
    log.info("Data saved")


if __name__ == '__main__':
    extract_features()
