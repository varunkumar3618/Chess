import csv
import argparse

import numpy as np
import tensorflow as tf
import chess

from src.features import ALL_FEATURES

def _result_to_int(result):
    if result == "1-0":
        return 1
    elif result == "0-1":
        return -1
    else:
        return 0

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _all_feature_extractor(epd):
    board = chess.Board()
    board.set_epd(epd)
    vectors = []
    for feature in ALL_FEATURES:
        vector = feature.value(board).flatten()
        vectors.append(vector)
    featureVector = np.concatenate(vectors)
    return featureVector

def _get_feature_extractor(features_name):
    if features_name == "all":
        return _all_feature_extractor
    else:
        raise ValueError("Unrecognized feature extractor: %s" % features_name)

def main(args):
    infile = args.infile
    outfile = args.outfile
    keep_draws = args.keep_draws
    features_name = args.features

    feature_extractor = _get_feature_extractor(features_name)

    with tf.python_io.TFRecordWriter(outfile) as writer:
        with open(infile, "rb") as inf:
            reader = csv.reader(inf, delimiter=",", quotechar="\"")
            i = 0
            for epd, result, steps in reader:
                if i % 1000 == 0:
                    print "At line %s" % (i + 1)
                i += 1
                if result == "1/2-1/2" and not keep_draws:
                    continue
                example = tf.train.Example(
                    features=tf.train.Features(feature={
                        "board": _bytes_feature(feature_extractor(epd).tobytes()),
                        "steps": _int64_feature(int(steps)),
                        "label": _int64_feature(_result_to_int(result))
                    })
                )
                writer.write(example.SerializeToString())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert the board positions dataset from CSV to TensorFlow format.")
    parser.add_argument("--infile", type=str, help="The CSV file to convert", required=True)
    parser.add_argument("--outfile", type=str, help="Where to store the tensorflow file", required=True)
    parser.add_argument("--keep_draws", type=bool, default=False, help="Whether to retain boards from drawn games.")
    parser.add_argument("--features", type=str, required=True, help="The feature extractor to use. Choose from: [all]")

    main(parser.parse_args())
