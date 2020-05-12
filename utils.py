# utils.py

import pickle

def load_dataset(path):
  with open(path, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

def load_train_data(folder_path):
  fn = '/data_batch_'
  train_data = []
  train_labels = []
  for i in range(1, 6):
    data_dict = load_dataset(folder_path+fn+str(i))
    for ex in data_dict[b'data']:
      train_data.append(ex)
    for label in data_dict[b'labels']:
      train_labels.append(label)
  return train_data, train_labels

def load_test_data(folder_path):
  fn = '/test_batch'
  test_data = []
  test_labels = []
  data_dict = load_dataset(folder_path+fn)
  for ex in data_dict[b'data']:
    test_data.append(ex)
  for label in data_dict[b'labels']:
    test_labels.append(label)
  return test_data, test_labels