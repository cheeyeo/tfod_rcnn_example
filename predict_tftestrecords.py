# https://www.tensorflow.org/tutorials/load_data/tfrecord#reading_a_tfrecord_file_2

import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    testset = "lisa/records/testing.record"

    dataset = tf.data.TFRecordDataset(testset)
    print(dataset)

    # result = {}

    filenames = []
    for rec in dataset.take(20):
        example = tf.train.Example()
        example.ParseFromString(rec.numpy())
        print(example.features.feature["image/filename"])

        for key, feature in example.features.feature.items(): 
            kind = feature.WhichOneof("kind")
            val = getattr(feature, kind).value
            # result[key] = np.array(val)
            if key == "image/filename":
                filenames.append(val[0].decode("utf-8"))

    # print(result.keys())
    # print(result["image/filename"][0].decode("utf-8"))
    print(filenames)