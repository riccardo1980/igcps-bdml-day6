# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example implementation of code to run on the Cloud ML service.
"""

import argparse
import model
import json
import os

import tensorflow as tf

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--traindata',
      help='Training data can have wildcards',
      required=True
  )
  parser.add_argument(
      '--evaldata',
      help='Training data can have wildcards',
      required=True
  )
  parser.add_argument(
      '--origin_file',
      help='File containing keys for origin column',
      required=True
  )
  parser.add_argument(
      '--dest_file',
      help='File containing keys for dest column',
      required=True
  )
  parser.add_argument(
      '--job-dir',
      help='this model ignores this field, but it is required by gcloud',
      default='./junk'
  )
  parser.add_argument(
      '--output_dir',
      help='Output directory',
      required=True
  )
  parser.add_argument(
      '--num_training_epochs',
      help='Number of passes through training dataset',
      type=int,
      default=10
  )

  # for hyper-parameter tuning
  parser.add_argument(
      '--batch_size',
      help='Number of examples to compute gradient on',
      type=int,
      default=100
  )
  parser.add_argument(
      '--nbuckets',
      help='Number of bins into which to discretize lats and lons',
      type=int,
      default=5
  )
  parser.add_argument(
      '--hidden_units',
      help='Architecture of DNN part of wide-and-deep network',
      default='64,64,64,16,4'
  )
  parser.add_argument(
      '--learning_rate',
      help='Controls size of step in gradient descent.',
      type=float,
      default=0.0606
  )

  # parse args
  args = parser.parse_args()
  arguments = args.__dict__

  # unused args provided by service
  arguments.pop('job-dir', None)
  arguments.pop('job_dir', None)
  output_dir = arguments.pop('output_dir')

  # when hp-tuning, we need to use different output directories for different runs
  output_dir = os.path.join(
      output_dir,
      json.loads(
          os.environ.get('TF_CONFIG', '{}')
      ).get('task', {}).get('trial', '')
  )
 

  # run
  tf.logging.set_verbosity(tf.logging.INFO)
  # create estimator
  estimator = model.wide_and_deep_model(output_dir,
                                  arguments['origin_file'],
                                  arguments['dest_file'], 
                                  arguments['nbuckets'],
                                  arguments['hidden_units'],
                                  arguments['learning_rate'])

  estimator = tf.contrib.estimator.add_metrics(estimator, model.my_rmse)

  train_spec = tf.estimator.TrainSpec(input_fn=model.read_dataset(arguments['traindata'], 
                                                          mode=tf.estimator.ModeKeys.TRAIN, 
                                                          batch_size=arguments['batch_size'], 
                                                          num_training_epochs=arguments['num_training_epochs']))

  eval_spec = tf.estimator.EvalSpec(input_fn=model.read_dataset(arguments['evaldata']),
                                    steps = None,
                                    start_delay_secs = 20 * 60, # start evaluating after N seconds
                                    throttle_secs = 10 * 60)  # evaluate every N seconds

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  estimator.export_savedmodel(os.path.join(output_dir,'Servo'),
                              serving_input_receiver_fn=model.serving_input_fn())
    
