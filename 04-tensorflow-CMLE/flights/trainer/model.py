from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
import tensorflow.contrib.metrics as tfmetrics
import tensorflow as tf
import numpy as np
import os

CSV_COLUMNS  = ('ontime,dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay' + \
                ',carrier,dep_lat,dep_lon,arr_lat,arr_lon,origin,dest').split(',')
LABEL_COLUMN = 'ontime'
DEFAULTS     = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],\
                ['na'],[0.0],[0.0],[0.0],[0.0],['na'],['na']]

def read_dataset(filename, mode=tf.contrib.learn.ModeKeys.EVAL, batch_size=512, num_training_epochs=10):

  # the actual input function passed to TensorFlow
  def _input_fn():
    num_epochs = num_training_epochs if mode == tf.contrib.learn.ModeKeys.TRAIN else 1

    # could be a path to one file or a file pattern.
    input_file_names = tf.train.match_filenames_once(filename)
    filename_queue = tf.train.string_input_producer(
        input_file_names, num_epochs=num_epochs, shuffle=True)
 
    # read CSV
    reader = tf.TextLineReader()
    _, value = reader.read_up_to(filename_queue, num_records=batch_size)
    value_column = tf.expand_dims(value, -1)
    columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
    features = dict(zip(CSV_COLUMNS, columns))
    label = features.pop(LABEL_COLUMN)
    return features, label
  
  return _input_fn

def get_features_raw(origin_file, dest_file):
    real = {
      colname : tf.feature_column.numeric_column(colname) \
          for colname in \
            ('dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay' + 
             ',dep_lat,dep_lon,arr_lat,arr_lon').split(',')
    }
    sparse = {
      'carrier': tf.feature_column.categorical_column_with_vocabulary_list('carrier',
                  vocabulary_list='AS,B6,WN,HA,OO,F9,NK,EV,DL,UA,US,AA,MQ,VX'.split(','),
                  dtype=tf.string)
      , 'origin': tf.feature_column.categorical_column_with_vocabulary_file('origin',origin_file)
      , 'dest'   : tf.feature_column.categorical_column_with_vocabulary_file('dest',dest_file)
    }
    return real, sparse

def get_features(origin_file, dest_file):
    return get_features_raw(origin_file, dest_file)

def parse_hidden_units(s):
    return [int(item) for item in s.split(',')]

def wide_and_deep_model(output_dir,  origin_file, dest_file, nbuckets=5, hidden_units='64,32', learning_rate=0.01):
    real, sparse = get_features(origin_file, dest_file)

    # the lat/lon columns can be discretized to yield "air traffic corridors"
    latbuckets = np.linspace(20.0, 50.0, nbuckets).tolist()  # USA
    lonbuckets = np.linspace(-120.0, -70.0, nbuckets).tolist() # USA
    disc = {}
    disc.update({
       'd_{}'.format(key) : tf.feature_column.bucketized_column(real[key], latbuckets) \
          for key in ['dep_lat', 'arr_lat']
    })
    disc.update({
       'd_{}'.format(key) : tf.feature_column.bucketized_column(real[key], lonbuckets) \
          for key in ['dep_lon', 'arr_lon']
    })

    # cross columns that make sense in combination
    sparse['dep_loc'] = tf.feature_column.crossed_column([disc['d_dep_lat'], disc['d_dep_lon']],\
                                                nbuckets*nbuckets)
    sparse['dep_loc'] = tf.feature_column.crossed_column([disc['d_dep_lat'], disc['d_dep_lon']],\
                                                nbuckets*nbuckets)
    sparse['arr_loc'] = tf.feature_column.crossed_column([disc['d_arr_lat'], disc['d_arr_lon']],\
                                                nbuckets*nbuckets)
    sparse['dep_arr'] = tf.feature_column.crossed_column([sparse['dep_loc'], sparse['arr_loc']],\
                                                nbuckets ** 4)
    sparse['ori_dest'] = tf.feature_column.crossed_column([sparse['origin'], sparse['dest']], \
                                                hash_bucket_size=1000)
    
    # create embeddings of all the sparse columns
    embed = {
       colname : create_embed(col) \
          for colname, col in sparse.items()
    }
    real.update(embed)
    
    estimator = \
        tf.estimator.DNNLinearCombinedClassifier(model_dir=output_dir,
                                           linear_feature_columns=sparse.values(),
                                           dnn_feature_columns=real.values(),
                                           dnn_hidden_units=parse_hidden_units(hidden_units),
                                           loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE,
                                           linear_optimizer=tf.train.FtrlOptimizer(learning_rate=learning_rate),
                                           dnn_optimizer=tf.train.AdagradOptimizer(learning_rate=learning_rate*0.25))
    
    # estimator.params["head"]._thresholds = [0.7]  # FIXME: hack (seems it's not a valid member)
    return estimator

def create_embed(sparse_col):
    dim = 10 # default
    if hasattr(sparse_col, 'bucket_size'):
       nbins = sparse_col.bucket_size
       if nbins is not None:
          dim = 1 + int(round(np.log2(nbins)))
    return tf.feature_column.embedding_column(sparse_col, dimension=dim)

def serving_input_fn():
    feature_placeholders = {
      key : tf.placeholder(tf.float32, [None]) \
        for key in ('dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay' +
             ',dep_lat,dep_lon,arr_lat,arr_lon').split(',')
    }
    feature_placeholders.update( {
      key : tf.placeholder(tf.string, [None]) \
        for key in 'carrier,origin,dest'.split(',')
    } )

    features = {
      key: tf.expand_dims(tensor, -1)
      for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.build_raw_serving_input_receiver_fn(feature_placeholders)

def my_rmse(predictions, labels, **args):
  prob_ontime = predictions['probabilities'][:,1]

  return {'rmse': tf.metrics.root_mean_squared_error(prob_ontime, labels)}