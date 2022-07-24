import argparse
import json
import sys
from pathlib import Path

import razdel
from jsonschema import validate
from sklearn.preprocessing import LabelEncoder

from utils.clean import clean_txt
from utils.logger import get_logger
import numpy as np
import pandas as pd
import tensorflow as tf
from catboost import CatBoostClassifier

validation_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "array",
    "items": [
        {
            "type": "integer"
        },
        {
            "type": "string"
        }
    ]
}


def process_x(items):
    result = []

    for _id, body in items:
        body = clean_txt(body)
        sentences = [s.text for s in razdel.sentenize(body)]
        sentences = [s for s in sentences if len(s) > 1]

        for sentence in sentences:
            if sentence is None or len(sentence) <= 5:
                continue

            result.append({'body': sentence, 'global_id': _id})

    return result


def main(args):
    logger = get_logger()

    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("--prod", required=True,
                    help="Prod data")
    ap.add_argument("--labels", required=True,
                    help="Labels")
    ap.add_argument("--model", required=True,
                    help="RNN model")
    ap.add_argument('--cbmodel', required=True,
                    help="Catboost model")

    ap.add_argument("-s", "--seed", type=int, default=42,
                    help="Seed. Default: 42")
    ap.add_argument("-v", "--verbose", type=int, default=0,
                    help="Verbose")

    ap.add_argument("--positive", type=str, default='positive',
                    help="Positive label. Default 'positive'")
    ap.add_argument("--negative", type=str, default='negative',
                    help="Negative label. Default 'negative'")
    ap.add_argument("--neutral", type=str, default='neutral',
                    help="Neutral label. Default 'neutral'")
    ap.add_argument('--result', required=True,
                    help="Predictions result")

    args, unknown = ap.parse_known_args(args[1:])
    args = vars(args)

    verbose = args['verbose']
    if verbose >= 52:
        logger.info(args)

    seed = args['seed']

    np.random.seed(seed)

    model_path = Path(args['model'])
    cb_model_path = Path(args['cbmodel'])
    labels_path = Path(args['labels'])
    result_path = Path(args['result'])

    positive_label = str(args['positive'])
    negative_label = str(args['negative'])
    neutral_label = str(args['neutral'])

    prod_path = Path(args['prod'])

    if not prod_path.exists() or not prod_path.is_file():
        raise Exception('Prod path is incorrect')

    if not labels_path.exists() or not labels_path.is_file():
        raise Exception('Labels path is incorrect')

    with open(prod_path, 'r') as f:
        prod = json.load(f)

    for item in prod:
        validate(item, schema=validation_schema)
    model = tf.keras.models.load_model(model_path)
    cmodel = CatBoostClassifier()
    cmodel.load_model(str(cb_model_path))

    df = process_x(prod)
    df = pd.DataFrame(df)

    with open(labels_path, 'r') as f:
        labels = json.load(f)

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    if verbose >= 2:
        logger.info(df.head())

    df = df[df.body.notnull()]
    df = df[['global_id', 'body']]
    df.reset_index(inplace=True, drop=True)

    df['decision'] = [np.argmax(p) for p in model.predict(df['body'].values)]
    df['decision_label'] = df['decision'].apply(lambda x: label_encoder.inverse_transform([x])[0])
    df['pp'] = df['decision_label'].apply(lambda x: 1 if x == positive_label else 0)
    df['nn'] = df['decision_label'].apply(lambda x: 1 if x == negative_label else 0)
    df['ne'] = df['decision_label'].apply(lambda x: 1 if x == neutral_label else 0)

    df = df[['global_id', 'decision', 'pp', 'nn', 'ne']]

    gb = df.groupby(['global_id'])
    counts = gb.size().to_frame(name="amount")

    counts = counts \
        .join(
        gb['pp'].sum().to_frame(name="positive")
    ) \
        .join(
        gb['nn'].sum().to_frame(name="negative")
    ) \
        .join(
        gb['ne'].sum().to_frame(name="neutral")
    ) \
        .reset_index()

    counts['pos_percent'] = counts['positive'] / counts['amount']
    x_test = counts[
        [
            'amount',
            'positive',
            'negative',
            'neutral',
            'pos_percent'
        ]
    ].astype(dtype=np.float)

    counts['p_target'] = label_encoder.inverse_transform(cmodel.predict(x_test))

    if verbose > 0:
        logger.info(f'Total: {counts.shape}')

    if verbose >= 52:
        logger.info(counts.head(10))

    counts[['global_id', 'p_target']].to_json(result_path, orient='records', force_ascii=False)


if __name__ == '__main__':
    main(sys.argv)
