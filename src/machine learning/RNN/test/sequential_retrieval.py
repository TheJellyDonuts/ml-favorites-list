import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs


# !wget -nc https://raw.githubusercontent.com/tensorflow/examples/master/lite/examples/recommendation/ml/data/example_generation_movielens.py
# !python -m example_generation_movielens  --data_dir=data/raw  --output_dir=data/examples  --min_timeline_length=3  --max_context_length=10  --max_context_movie_genre_length=10  --min_rating=2  --train_data_fraction=0.9  --build_vocabs=False

TFRECORD_NAME = "apps" # Old: train_movielens_1m
DATASET_NAME = "apps" # Old: movielens/1m-movies


train_filename = f"./data/examples/train_{TFRECORD_NAME}.tfrecord"
train = tf.data.TFRecordDataset(train_filename)

test_filename = f"./data/examples/test_{TFRECORD_NAME}.tfrecord"
test = tf.data.TFRecordDataset(test_filename)

# feature_description = {
#     'context_movie_id': tf.io.FixedLenFeature([10], tf.int64, default_value=np.repeat(0, 10)),
#     'context_movie_rating': tf.io.FixedLenFeature([10], tf.float32, default_value=np.repeat(0, 10)),
#     'context_movie_year': tf.io.FixedLenFeature([10], tf.int64, default_value=np.repeat(1980, 10)),
#     'context_movie_genre': tf.io.FixedLenFeature([10], tf.string, default_value=np.repeat("Drama", 10)),
#     'label_movie_id': tf.io.FixedLenFeature([1], tf.int64, default_value=0),
# }

feature_description = {
    'context_app_id':         tf.io.FixedLenFeature([10], tf.int64, default_value=np.repeat(0, 10)),
    'context_app_category':   tf.io.FixedLenFeature([10], tf.string, default_value=np.repeat("None", 10)),
    'context_app_score':      tf.io.FixedLenFeature([10], tf.int64, default_value=np.repeat(0, 10)),
    'label_app_id':           tf.io.FixedLenFeature([1],  tf.int64, default_value=0),
}

def _parse_function(example_proto):
  return tf.io.parse_single_example(example_proto, feature_description)

train_ds = train.map(_parse_function).map(lambda x: {
    "context_app_id": tf.strings.as_string(x["context_app_id"]),
    "label_app_id": tf.strings.as_string(x["label_app_id"])
})

test_ds = test.map(_parse_function).map(lambda x: {
    "context_app_id": tf.strings.as_string(x["context_app_id"]),
    "label_app_id": tf.strings.as_string(x["label_app_id"])
})

for x in train_ds.take(1).as_numpy_iterator():
  pprint.pprint(x)


# change to apps
apps = tfds.load(DATASET_NAME, split='train')
apps = apps.map(lambda x: x["app_id"])
app_ids = apps.batch(1_000)
unique_app_ids = np.unique(np.concatenate(list(app_ids)))


embedding_dimension = 32

query_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
      vocabulary=unique_app_ids, mask_token=None),
    tf.keras.layers.Embedding(len(unique_app_ids) + 1, embedding_dimension),
    tf.keras.layers.GRU(embedding_dimension),
])

candidate_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_app_ids, mask_token=None),
  tf.keras.layers.Embedding(len(unique_app_ids) + 1, embedding_dimension)
])

metrics = tfrs.metrics.FactorizedTopK(
  candidates=apps.batch(128).map(candidate_model)
)

task = tfrs.tasks.Retrieval(
  metrics=metrics
)

class Model(tfrs.Model):

    def __init__(self, query_model, candidate_model):
        super().__init__()
        self._query_model = query_model
        self._candidate_model = candidate_model

        self._task = task

    def compute_loss(self, features, training=False):
        use_history = features["context_app_id"]
        use_next_label = features["label_app_id"]

        query_embedding = self._query_model(use_history)
        candidate_embedding = self._candidate_model(use_next_label)

        return self._task(query_embedding, candidate_embedding, compute_metrics=not training)


model = Model(query_model, candidate_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train_ds.shuffle(10_000).batch(12800).cache()
cached_test = test_ds.batch(2560).cache()

model.fit(cached_train, epochs=3)

model.evaluate(cached_test, return_dict=True)