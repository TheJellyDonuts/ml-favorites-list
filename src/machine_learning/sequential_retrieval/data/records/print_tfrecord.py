import tensorflow as tf
import os
import inquirer

# Get Cur Dir
cur_dir = os.path.dirname(__file__)

# Get all the files that end in tfrecord from the current direcotry
tfrecord_files = [f for f in os.listdir(cur_dir) if f.endswith('.tfrecord')]

# Ask the user which file they want to print
questions = [
  inquirer.List('file',
                message="Which file do you want to print?",
                choices=tfrecord_files,
            ),
]
answers = inquirer.prompt(questions)
input_file = answers['file']

# Ask the user how many records they want to print
questions = [
    inquirer.Text('records',
                    message="How many records do you want to print?",
                    default=1,
    ),
]
answers = inquirer.prompt(questions)
num_of_records = answers['records']

# Input Path
input_path = os.path.join(cur_dir, input_file)

raw_dataset = tf.data.TFRecordDataset(input_path)

for num in range(int(num_of_records)):
    for raw_record in raw_dataset.take(num+1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)