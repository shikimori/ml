import argparse
import json
import pickle
import sys
from pathlib import Path

from jsonschema import validate

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
    ap.add_argument("--train", required=True,
                    help="Train data")
    ap.add_argument("--test", required=True,
                    help="Train data")
    ap.add_argument("--labels", required=True,
                    help="Labels file")

    ap.add_argument("-s", "--seed", type=int, default=42,
                    help="Seed. Default: 42")

    ap.add_argument("--vocab", type=int, default=10000,
                    help="Vocabulary size. Default: 10000")

    ap.add_argument("-v", "--verbose", type=int, default=0,
                    help="Verbose")

    ap.add_argument("-e", "--epochs", type=int, default=10,
                    help="Number of epochs")
    ap.add_argument("--validation_steps", type=int, default=30,
                    help="Number of validation steps")

    ap.add_argument("--model", required=True,
                    help="Path to save TF model")
    ap.add_argument("--history",
                    help="[Optional] Path to save training history")

    args, unknown = ap.parse_known_args(args[1:])
    args = vars(args)

    verbose = args['verbose']
    vocab = args['vocab']
    epochs = args['epochs']
    validation_steps = args['validation_steps']

    if verbose >= 52:
        logger.info(args)

    seed = args['seed']

    np.random.seed(seed)

    train_path = Path(args['train'])
    test_path = Path(args['test'])
    model_path = Path(args['model'])
    labels_path = Path(args['labels'])
    history_path = args['history']

    if history_path is not None:
        history_path = Path(history_path)

    if not train_path.exists() or not train_path.is_file():
        raise Exception('Train path is incorrect')

    if not test_path.exists() or not test_path.is_file():
        raise Exception('Test path is incorrect')

    if not labels_path.exists() or not labels_path.is_file():
        raise Exception('Labels path is incorrect')

    with open(train_path, 'r') as f:
        train_raw = json.load(f)

    with open(labels_path, 'r') as f:
        labels = json.load(f)

    for item in train_raw:
        validate(item, schema=validation_schema)

    with open(test_path, 'r') as f:
        test_raw = json.load(f)

    for item in test_raw:
        validate(item, schema=validation_schema)

    train = pd.DataFrame(train_raw)
    val = pd.DataFrame(test_raw)

    num_classes = len(labels)
    x_field = 'text'
    y_field = 'label'

    if verbose >= 52:
        logger.info(train.head())

    encoder = tf.keras.layers.TextVectorization(
        max_tokens=vocab)
    encoder.adapt(train[x_field].values)

    actual_vocab_size = np.array(encoder.get_vocabulary()).size

    if verbose >= 52:
        logger.info(f'Actual vocab size: {actual_vocab_size}')

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=actual_vocab_size,
            output_dim=64,
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=['accuracy']
    )

    history = model.fit(
        tf.convert_to_tensor(train[x_field].values, dtype=tf.string),
        tf.keras.utils.to_categorical(train[y_field].values, num_classes=num_classes),
        epochs=epochs,
        validation_data=(
            tf.convert_to_tensor(val[x_field].values, dtype=tf.string),
            tf.keras.utils.to_categorical(val[y_field].values, num_classes=num_classes),
        ),
        validation_steps=validation_steps,
    )

    if history_path is not None:
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)

    model.save(model_path, overwrite=True)


if __name__ == '__main__':
    main(sys.argv)
