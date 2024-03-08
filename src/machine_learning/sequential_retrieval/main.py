import os
import pprint
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds
import pandas as pd

TFRECORD_NAME = "app"

# Get Current Directory
cur_dir = os.path.dirname(__file__)

# 
train_filename = os.path.join(cur_dir, f"data/records/train_{TFRECORD_NAME}.tfrecord")
train = tf.data.TFRecordDataset(train_filename)

test_filename = os.path.join(cur_dir, f"data/records/test_{TFRECORD_NAME}.tfrecord")
test = tf.data.TFRecordDataset(test_filename)

feature_description = {
    'context_app_id':         tf.io.FixedLenFeature([10], tf.int64, default_value=np.repeat(0, 10)),
    'context_app_category':   tf.io.FixedLenFeature([10], tf.string, default_value=np.repeat("None", 10)),
    'context_app_rating':     tf.io.FixedLenFeature([10], tf.float32, default_value=np.repeat(0, 10)),
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

# Get Current Directory
cur_dir = os.path.dirname(__file__)

app_csv_path = os.path.join(cur_dir, 'data', 'csv', 'apps.csv')
# !ratings_csv_path = os.path.join(cur_dir, 'data', 'csv', 'ratings.csv')
# !users_csv_path = os.path.join(cur_dir, 'data', 'csv', 'users.csv')

# Load your CSV files into pandas DataFrames
apps_df = pd.read_csv(app_csv_path, header=None, names=['app_id', 'app_name', 'app_category'])

# Convert apps_df['app_id'] to string
apps_df['app_id'] = apps_df['app_id'].astype(str)

# Make apps_df a TensorFlow Dataset
apps_df = tf.data.Dataset.from_tensor_slices(dict(apps_df))

# Make app a list of only App IDs
apps = apps_df.map(lambda x: x['app_id'])

# Batch the app_ids
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
    
    def call(self, inputs, training=False):
        return self._query_model(inputs)



model = Model(query_model, candidate_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train_ds.shuffle(10_000).batch(12800).cache()
cached_test = test_ds.batch(2560).cache()

model.fit(cached_train, epochs=20000)

model.evaluate(cached_test, return_dict=True)


model.build(input_shape=(None, 10))

# Make predictions
user_id = "3"
context_app_id = "29"

# I want to use this model. HOW?
x_new = np.array([user_id, context_app_id])

x_new = tf.reshape(x_new, (1, -1, 1))

# Preproces x_new to match what I did earlier
x_new = tf.data.Dataset.from_tensor_slices(x_new)
x_new = x_new.batch(1)

# Reshape x_new

predictions = model.predict(x_new)

# Return the top 10 recommendations
top_10_indices = np.argsort(predictions[0])[-10:][::-1]

# Use these indices to get the app_ids
top_10_app_ids = unique_app_ids[top_10_indices]

# Print the top 10 app_ids
print(top_10_app_ids)

# print(predictions)