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
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

validation_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "properties": {
        "target": {
            "type": "integer"
        },
        "amount": {
            "type": "integer"
        },
        "positive": {
            "type": "integer"
        },
        "negative": {
            "type": "integer"
        },
        "neutral": {
            "type": "integer"
        },
        "pos_percent": {
            "type": "number"
        }
    },
    "required": [
        "target",
        "amount",
        "positive",
        "negative",
        "neutral",
        "pos_percent"
    ]
}


def main(args):
    logger = get_logger()

    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("--train", required=True,
                    help="Train data")

    ap.add_argument("-s", "--seed", type=int, default=42,
                    help="Seed. Default: 42")
    ap.add_argument("-ss", "--splitsize", type=float, default=0.8,
                    help="Split size. Default: 0.8")
    ap.add_argument("-v", "--verbose", type=int, default=0,
                    help="Verbose")

    ap.add_argument("--model", required=True,
                    help="Train Catboost Classifier")

    args, unknown = ap.parse_known_args(args[1:])
    args = vars(args)

    verbose = args['verbose']
    if verbose >= 52:
        logger.info(args)

    logging_level = 'Silent' if verbose < 52 else 'Verbose'

    seed = args['seed']
    split_size = args['splitsize']

    np.random.seed(seed)

    train_path = Path(args['train'])
    model_path = Path(args['model'])

    if not train_path.exists() or not train_path.is_file():
        raise Exception('Train path is incorrect')

    df = pd.read_json(train_path)

    y = df['target']
    x = df[
        [
            'amount',
            'positive',
            'negative',
            'neutral',
            'pos_percent'
        ]
    ].astype(dtype=np.float)

    x_train, x_validation, y_train, y_validation = train_test_split(x, y, train_size=split_size, random_state=seed)

    cmodel = CatBoostClassifier(
        custom_loss=['MultiClass'],
        random_seed=seed,
        logging_level=logging_level
    )

    cmodel.fit(
        x_train, y_train,
        cat_features=[],
        eval_set=(x_validation, y_validation),
        logging_level=logging_level
    )

    cmodel.save_model(str(model_path))


if __name__ == '__main__':
    main(sys.argv)
