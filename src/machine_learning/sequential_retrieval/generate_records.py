'''
Stephen Venable
2024-03-08

This script is used to generate the train_app.tfrecord and test_app.tfrecord files from the apps.csv and ratings.csv files.
'''
import collections
import os
import random

from absl import app
from absl import flags
from absl import logging
import pandas as pd
import tensorflow as tf

FLAGS = flags.FLAGS

RATINGS_FILE_NAME = "ratings.csv"
APPS_FILE_NAME = "apps.csv"
RATINGS_DATA_COLUMNS = ["UserID", "AppID", "Rating", "Timestamp"]
APPS_DATA_COLUMNS = ["AppID", "Name", "Categories"]
OUTPUT_TRAINING_DATA_FILENAME = "train_app.tfrecord"
OUTPUT_TESTING_DATA_FILENAME = "test_app.tfrecord"
PAD_APP_ID = 0
PAD_RATING = 0.0
UNKNOWN_STR = "UNK"

cur_dir = os.path.dirname(__file__)
input_path = os.path.join(cur_dir, '../../data/generated/')
output_path = os.path.join(cur_dir, './data/records/')

def define_flags():
  """Define flags."""
  flags.DEFINE_string("data_dir", input_path,
                      "Path to download and store apps data.")
  flags.DEFINE_string("output_dir", output_path,
                      "Path to the directory of output files.")
  flags.DEFINE_integer("min_timeline_length", 3,
                       "The minimum timeline length to construct examples.")
  flags.DEFINE_integer("max_context_length", 10,
                       "The maximum length of user context history.")
  flags.DEFINE_integer("max_context_app_category_length", 10,
                       "The maximum length of user context history.")
  flags.DEFINE_integer(
      "min_rating", None, "Minimum rating of app that will be used to in "
      "training data")
  flags.DEFINE_float("train_data_fraction", 0.9, "Fraction of training data.")


class AppInfo(
    collections.namedtuple(
        "AppInfo", ["app_id", "timestamp", "rating", "title", "categories"])):
  """Data holder of basic information of an app."""
  __slots__ = ()

  def __new__(cls,
              app_id=PAD_APP_ID,
              timestamp=0,
              rating=PAD_RATING,
              title="",
              categories=""):
    return super(AppInfo, cls).__new__(cls, app_id, timestamp, rating,
                                         title, categories)

def read_data(data_directory, min_rating=None):
  """Read ratings.csv and apps.csv file into dataframe."""
  ratings_df = pd.read_csv(
      os.path.join(data_directory, RATINGS_FILE_NAME),
      sep=",",
      names=RATINGS_DATA_COLUMNS,
      encoding="unicode_escape")  # May contain unicode. Need to escape.
  ratings_df["Timestamp"] = ratings_df["Timestamp"].apply(int)
  if min_rating is not None:
    ratings_df = ratings_df[ratings_df["Rating"] >= min_rating]
  apps_df = pd.read_csv(
      os.path.join(data_directory, APPS_FILE_NAME),
      sep=",",
      names=APPS_DATA_COLUMNS,
      encoding="unicode_escape")  # May contain unicode. Need to escape.
  return ratings_df, apps_df


def convert_to_timelines(ratings_df):
  """Convert ratings data to user."""
  timelines = collections.defaultdict(list)
  app_counts = collections.Counter()
  for user_id, app_id, rating, timestamp in ratings_df.values:
    timelines[user_id].append(
        AppInfo(app_id=app_id, timestamp=int(timestamp), rating=rating))
    app_counts[app_id] += 1
  # Sort per-user timeline by timestamp
  for (user_id, context) in timelines.items():
    context.sort(key=lambda x: x.timestamp)
    timelines[user_id] = context
  return timelines, app_counts


def generate_apps_dict(apps_df):
  """Generates apps dictionary from apps dataframe."""
  apps_dict = {
      app_id: AppInfo(app_id=app_id, title=title, categories=categories)
      for app_id, title, categories in apps_df.values
  }
  apps_dict[0] = AppInfo()
  return apps_dict



def generate_app_categories(apps_dict, apps):
  """Create a feature of the category of each app.

  Save category as a feature for the apps.

  Args:
    apps_dict: Dict of apps, keyed by app_id with value of (name, category)
    apps: list of apps to extract categories.

  Returns:
    app_categories: list of categories of all input apps.
  """
  app_categories = []
  for app in apps:
    if not apps_dict[app.app_id].categories:
      continue
    categories = [
        tf.compat.as_bytes(category)
        for category in apps_dict[app.app_id].categories.split("|")
    ]
    app_categories.extend(categories)

  return app_categories


def _pad_or_truncate_app_feature(feature, max_len, pad_value):
  feature.extend([pad_value for _ in range(max_len - len(feature))])
  return feature[:max_len]


def generate_examples_from_single_timeline(timeline,
                                           apps_dict,
                                           max_context_len=100,
                                           max_context_app_category_len=320):
  """Generate TF examples from a single user timeline.

  Generate TF examples from a single user timeline. Timeline with length less
  than minimum timeline length will be skipped. And if context user history
  length is shorter than max_context_len, features will be padded with default
  values.

  Args:
    timeline: The timeline to generate TF examples from.
    apps_dict: Dictionary of all AppInfos.
    max_context_len: The maximum length of the context. If the context history
      length is less than max_context_length, features will be padded with
      default values.
    max_context_app_category_len: The length of app category feature.

  Returns:
    examples: Generated examples from this single timeline.
  """
  examples = []
  for label_idx in range(1, len(timeline)):
    start_idx = max(0, label_idx - max_context_len)
    context = timeline[start_idx:label_idx]
    while len(context) < max_context_len:
      context.append(AppInfo())
    label_app_id = int(timeline[label_idx].app_id)
    context_app_id = [int(app.app_id) for app in context]
    context_app_rating = [app.rating for app in context]
    context_app_categories = generate_app_categories(apps_dict, context)
    context_app_categories = _pad_or_truncate_app_feature(
        context_app_categories, max_context_app_category_len,
        tf.compat.as_bytes(UNKNOWN_STR))
    feature = {
        "context_app_id":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_app_id)),
        "context_app_rating":
            tf.train.Feature(
                float_list=tf.train.FloatList(value=context_app_rating)),
        "context_app_category":
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=context_app_categories)),
        "label_app_id":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=[label_app_id]))
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    examples.append(tf_example)

  return examples


def generate_examples_from_timelines(timelines,
                                     apps_df,
                                     min_timeline_len=3,
                                     max_context_len=100,
                                     max_context_app_category_len=320,
                                     train_data_fraction=0.9,
                                     random_seed=None,
                                     shuffle=True):
  """Convert user timelines to tf examples.

  Convert user timelines to tf examples by adding all possible context-label
  pairs in the examples pool.

  Args:
    timelines: The user timelines to process.
    apps_df: The dataframe of all apps.
    min_timeline_len: The minimum length of timeline. If the timeline length is
      less than min_timeline_len, empty examples list will be returned.
    max_context_len: The maximum length of the context. If the context history
      length is less than max_context_length, features will be padded with
      default values.
    max_context_app_category_len: The length of app category feature.
    train_data_fraction: Fraction of training data.
    random_seed: Seed for randomization.
    shuffle: Whether to shuffle the examples before splitting train and test
      data.

  Returns:
    train_examples: TF example list for training.
    test_examples: TF example list for testing.
  """
  examples = []
  apps_dict = generate_apps_dict(apps_df)
  progress_bar = tf.keras.utils.Progbar(len(timelines))
  for timeline in timelines.values():
    if len(timeline) < min_timeline_len:
      progress_bar.add(1)
      continue
    single_timeline_examples = generate_examples_from_single_timeline(
        timeline=timeline,
        apps_dict=apps_dict,
        max_context_len=max_context_len,
        max_context_app_category_len=max_context_app_category_len)
    examples.extend(single_timeline_examples)
    progress_bar.add(1)
  # Split the examples into train, test sets.
  if shuffle:
    random.seed(random_seed)
    random.shuffle(examples)
  last_train_index = round(len(examples) * train_data_fraction)

  train_examples = examples[:last_train_index]
  test_examples = examples[last_train_index:]
  return train_examples, test_examples

def write_tfrecords(tf_examples, filename):
  """Writes tf examples to tfrecord file, and returns the count."""
  with tf.io.TFRecordWriter(filename) as file_writer:
    length = len(tf_examples)
    progress_bar = tf.keras.utils.Progbar(length)
    for example in tf_examples:
      file_writer.write(example.SerializeToString())
      progress_bar.add(1)
    return length


def generate_datasets(extracted_data_dir,
                      output_dir,
                      min_timeline_length,
                      max_context_length,
                      max_context_app_category_length,
                      min_rating=None,
                      train_data_fraction=0.9,
                      train_filename=OUTPUT_TRAINING_DATA_FILENAME,
                      test_filename=OUTPUT_TESTING_DATA_FILENAME):
  """Generates train and test datasets as TFRecord, and returns stats."""
  logging.info("Reading data to dataframes.")
  ratings_df, apps_df = read_data(extracted_data_dir, min_rating=min_rating)
  logging.info("Generating app rating user timelines.")
  timelines, app_counts = convert_to_timelines(ratings_df)
  logging.info("Generating train and test examples.")
  train_examples, test_examples = generate_examples_from_timelines(
      timelines=timelines,
      apps_df=apps_df,
      min_timeline_len=min_timeline_length,
      max_context_len=max_context_length,
      max_context_app_category_len=max_context_app_category_length,
      train_data_fraction=train_data_fraction)

  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)
  logging.info("Writing generated training examples.")
  train_file = os.path.join(output_dir, train_filename)
  train_size = write_tfrecords(tf_examples=train_examples, filename=train_file)
  logging.info("Writing generated testing examples.")
  test_file = os.path.join(output_dir, test_filename)
  test_size = write_tfrecords(tf_examples=test_examples, filename=test_file)
  stats = {
      "train_size": train_size,
      "test_size": test_size,
      "train_file": train_file,
      "test_file": test_file,
  }
  return stats


def main(_):
  logging.info("Downloading and extracting data.")
#   extracted_data_dir = download_and_extract_data(data_directory=FLAGS.data_dir)
   
  extracted_data_dir = FLAGS.data_dir
  stats = generate_datasets(
      extracted_data_dir=extracted_data_dir,
      output_dir=FLAGS.output_dir,
      min_timeline_length=FLAGS.min_timeline_length,
      max_context_length=FLAGS.max_context_length,
      max_context_app_category_length=FLAGS.max_context_app_category_length,
      min_rating=FLAGS.min_rating,
      train_data_fraction=FLAGS.train_data_fraction,
  )
  logging.info("Generated dataset: %s", stats)


if __name__ == "__main__":
  define_flags()
  app.run(main)