import argparse
import json
import pickle
import sys
from pathlib import Path

from jsonschema import validate
from sklearn.preprocessing import LabelEncoder

from utils.logger import get_logger
import numpy as np
import pandas as pd
import tensorflow as tf

validation_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "properties": {
        "text": {
            "type": "string"
        },
        "label": {
            "type": "integer"
        }
    },
    "required": [
        "text",
        "label"
    ]
}


def main(args):
    logger = get_logger()

    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("--sentences", required=True,
                    help="Sentences file")
    ap.add_argument("--model", required=True,
                    help="Pretrained model")
    ap.add_argument("--labels", required=True,
                    help="Labels file")

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

    ap.add_argument("--predictions", required=True,
                    help="Predictions file")
    ap.add_argument("--cctrain", required=True,
                    help="Train set for Catboost Classifier")

    args, unknown = ap.parse_known_args(args[1:])
    args = vars(args)

    verbose = args['verbose']
    if verbose >= 52:
        logger.info(args)

    seed = args['seed']

    np.random.seed(seed)

    sentences_path = Path(args['sentences'])
    model_path = Path(args['model'])
    labels_path = Path(args['labels'])

    predictions_path = Path(args['predictions'])
    cctrain_path = Path(args['cctrain'])

    positive_label = str(args['positive'])
    negative_label = str(args['negative'])
    neutral_label = str(args['neutral'])

    if not sentences_path.exists() or not sentences_path.is_file():
        raise Exception('Sentences path is incorrect')

    if not labels_path.exists() or not labels_path.is_file():
        raise Exception('Labels path is incorrect')

    if not model_path.exists() or not model_path.is_dir():
        raise Exception('Models path is incorrect')

    sentences = pd.read_json(sentences_path)
    model = tf.keras.models.load_model(model_path)

    with open(labels_path, 'r') as f:
        labels = json.load(f)

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    source = sentences.copy()
    source.reset_index(inplace=True, drop=True)
    source = source.rename(columns={'body': 'text', 'opinion': 'target'})
    source['target'] = label_encoder.transform(source['target'])

    source['decision'] = [np.argmax(p) for p in model.predict(source['text'].values)]
    source['decision_label'] = source['decision'].apply(lambda x: label_encoder.inverse_transform([x])[0])
    source['pp'] = source['decision_label'].apply(lambda x: 1 if x == positive_label else 0)
    source['nn'] = source['decision_label'].apply(lambda x: 1 if x == negative_label else 0)
    source['ne'] = source['decision_label'].apply(lambda x: 1 if x == neutral_label else 0)

    source[['global_id', 'text', 'target', 'decision']].to_json(predictions_path, orient='records', force_ascii=False)

    source = source[['global_id', 'decision', 'target', 'pp', 'nn', 'ne']]

    gb = source.groupby(['global_id', 'target'])
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

    counts[[
        'target',
        'amount',
        'positive',
        'negative',
        'neutral',
        'pos_percent'
    ]].to_json(cctrain_path, orient='records', force_ascii=False)


if __name__ == '__main__':
    main(sys.argv)
