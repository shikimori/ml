import argparse
import json
import sys
from pathlib import Path

import razdel as razdel
from jsonschema import validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

from json_schema import input_schema
from utils.clean import clean_txt
from utils.logger import get_logger


def process_x(items):
    result = []

    for _id, body, opinion in items:
        body = clean_txt(body)
        sentences = [s.text for s in razdel.sentenize(body)]
        sentences = [s for s in sentences if len(s) > 1]

        for sentence in sentences:
            if sentence is None or len(sentence) <= 5:
                continue

            result.append({'body': sentence, 'opinion': opinion, 'global_id': _id})

    return result


def main(args):
    logger = get_logger()

    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-in", "--input", required=True,
                    help="Input file with raw data")

    ap.add_argument("-s", "--seed", type=int,
                    help="Seed. Default: 42")
    ap.add_argument("-ss", "--splitsize", type=float,
                    help="Split size. Default: 0.8")
    ap.add_argument("-v", "--verbose", type=int,
                    help="Verbose")
    # ap.add_argument("-p4", "--param4",
    #                 help="Need to sort")

    ap.add_argument("--train", required=True,
                    help="Slice for training")

    ap.add_argument("--test", required=True,
                    help="Slice for testing")

    ap.add_argument("--sentences", required=True,
                    help="Sentences file")

    ap.add_argument("--labels", required=True,
                    help="Labels file")

    args, unknown = ap.parse_known_args(args[1:])
    args = vars(args)

    verbose = args['verbose']

    if verbose >= 52:
        logger.info(args)

    seed = args['seed']
    split_size = args['splitsize']

    if split_size is None:
        split_size = 0.8

    if seed is None:
        seed = 42

    np.random.seed(seed)

    input_path = Path(args['input'])

    train_path = Path(args['train'])
    test_path = Path(args['test'])
    sentences_path = Path(args['sentences'])
    labels_path = Path(args['labels'])

    if not input_path.exists() or not input_path.is_file():
        raise Exception('Input path is incorrect')

    with open(input_path, 'r', encoding='utf8') as f:
        df = json.load(f)

    validate(df, schema=input_schema)

    if verbose >= 2:
        logger.info('Input data is correct!')

    df = process_x(df)
    df = pd.DataFrame(df)

    if verbose >= 2:
        logger.info(df.head())

    df = df[df.body.notnull()]

    if verbose >= 2:
        logger.info(f'Save sentences file to {sentences_path}')

    df.to_json(sentences_path, orient='records', force_ascii=False)

    df = df[['body', 'opinion']]
    df.reset_index(inplace=True, drop=True)
    df = df.rename(columns={'body': 'text', 'opinion': 'label'})

    labels = df['label'].unique()

    if verbose >= 2:
        logger.info(f'Save labels file to {labels_path}')

    with open(labels_path, 'w', encoding='utf8') as f:
        json.dump(list(labels), f, ensure_ascii=False)

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    df['label'] = label_encoder.transform(df['label'])

    train = df.copy()
    train = train.reindex(np.random.permutation(train.index))

    train, val = train_test_split(train, test_size=1 - split_size, random_state=seed)
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)

    if verbose >= 52:
        logger.info(f'Train: {train.shape}')
        logger.info(f'Val: {val.shape}')

    if verbose >= 2:
        logger.info(f'Save train split to {train_path}')

    train.to_json(train_path, orient='records', force_ascii=False)

    if verbose >= 2:
        logger.info(f'Save test split to {test_path}')

    val.to_json(test_path, orient='records', force_ascii=False)

    logger.info('Done flawlessly!')


if __name__ == '__main__':
    main(sys.argv)
