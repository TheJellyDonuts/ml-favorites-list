import tensorflow as tf
import os

# Get Cur Dir
cur_dir = os.path.dirname(__file__)

# Input Path
input_path = os.path.join(cur_dir, 'data/records/train_app.tfrecord')

raw_dataset = tf.data.TFRecordDataset(input_path)

for raw_record in raw_dataset.take(2):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)