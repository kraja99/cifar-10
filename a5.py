import argparse
import time
import numpy as np
from utils import *
from tqdm import tqdm
from sklearn.decomposition import PCA

from sklearn import metrics
# from skimage.feature import haar_like_feature, haar_like_feature_coord
from sklearn.ensemble import AdaBoostClassifier


class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class CifarClassifier(object):
  def predict(self, image):
    """
    :param image
    :return: predicted class label
    """
    raise Exception("Only implemented in subclasses")

  def train(self, data, labels):
    """
    :param data: training data
    :param labels: training labels
    """
    raise Exception("Only implemented in subclasses")

  def eval(self, do_print, test_data=None, test_labels=None):
    if do_print:
      print("Evaluating " + self.name)
    if test_data is None or test_labels is None:
      test_data, test_labels = load_test_data(args.data_path)
    acc = 0
    pred_labels = []
    it = range(len(test_data))
    if do_print:
      it = tqdm(it)
    for i in it:
      image = test_data[i]
      label = test_labels[i]
      pred = self.predict(image)
      pred_labels.append(pred)
      if pred == label:
        acc += 1
    acc /= len(test_data)
    if do_print:
      print("Test Accuracy:", acc)
      print("Confusion Matrix: (Cols are Predicted Values, Rows are Actual Values)")
      print(metrics.confusion_matrix(test_labels, pred_labels))
      print("Other Classification Metrics:")
      print(metrics.classification_report(test_labels, pred_labels, digits=3))
    return acc


class KNN(CifarClassifier):
  def __init__(self, k):
    super().__init__()
    self.name = 'KNN Classifier'
    self.k = k

  def predict(self, image):
    image = np.expand_dims(image, axis=0)
    image = self.pca.transform(image)
    dists = np.sum(np.abs(image - self.data), axis=1)
    indices = np.argpartition(dists, args.k)[:args.k]
    label_count = np.zeros(10)
    for i in indices:
      label_count[self.labels[i]] += 1
    return np.argmax(label_count)
  
  def train(self, data, labels):
    self.pca = PCA(n_components=30)
    self.data = self.pca.fit_transform(np.array(data))
    self.labels = labels


class DecisionStump():
  def __init__(self, ):
    super().__init__()
    self.polarity = 1
    self.feature_idx = None
    self.threshold = None
    self.alpha = None


class AdaboostBinary():
  def __init__(self, t):
    super().__init__()
    self.name = 'Binary Adaboost Classifier'
    self.t = t
  
  def predict(self, image):
    pred_label = 0
    sum_alpha = 0
    sum_alphah = 0
    for c in self.weak_classifiers:
      if image[c.feature_idx] >= c.threshold:
        sum_alphah += c.alpha 
      sum_alpha += c.alpha
    if sum_alphah > .5 * sum_alpha:
      pred_label = 1
    return pred_label, sum_alphah/sum_alpha

  def train(self, data, labels):
    m = labels.count(0)
    l = labels.count(1)
    self.weights = np.array([1/(2 * m) if label == 0 else 1/(2 * l) for label in labels])
    self.weak_classifiers = []
    for i in range(self.t):
      weak_classifier = DecisionStump()
      min_error = float('inf')
      self.weights /= np.sum(self.weights)
      for feature_idx in range(data.shape[1]):
        possible_thresholds = np.unique(data[:, feature_idx])
        translation = np.min(possible_thresholds)
        scale = np.max(possible_thresholds - translation)
        for b in range(1000):
          threshold = (b/1000) * scale + translation
          polarity = 1
          preds = np.ones(len(labels))
          preds[data[:, feature_idx] < threshold] = 0
          error = np.sum(self.weights[labels != preds])
          if error > 0.5:
            error = 1 - error
            polarity = -1
          if error < min_error:
            weak_classifier.polarity = polarity
            weak_classifier.feature_idx = feature_idx
            weak_classifier.threshold = threshold
            min_error = error
      beta = min_error / (1 - min_error)
      # print(min_error, beta, weak_classifier.threshold, np.sum(self.weights))
      weak_classifier.alpha = np.log(1/(beta))
      self.weak_classifiers.append(weak_classifier)
      preds = np.ones(len(labels))
      preds[weak_classifier.polarity * data[:, weak_classifier.feature_idx] < weak_classifier.polarity * weak_classifier.threshold] = 0
      for w_idx in range(len(self.weights)):
        if preds[w_idx] == labels[w_idx]:
          self.weights[w_idx] *= beta
      

class Adaboost(CifarClassifier):
  def __init__(self, t, num_haar):
    super().__init__()
    self.name = 'Adaboost Classifier'
    self.t = t
    self.num_haar = num_haar

  def predict(self, image):
    probs = []
    for i in range(10):
      _, score = self.strong_classifiers[i].predict(self.extract_haar_features(image))
      probs.append(score)
    return np.argmax(probs)
  
  def train(self, data, labels):
    print("Training Adaboost")
    start_time = time.time()
    data = data[:10000]
    labels = labels[:10000]
    print("Extracting Features")
    self.generate_haar_coords(32, 32)
    self.data = np.array([self.extract_haar_features(image) for image in tqdm(data)])
    self.strong_classifiers = []
    print("Training Binary Classifiers")
    for i in tqdm(range(0, 10)):
      binary_labels = [1 if l == i else 0 for l in labels]
      classifier = AdaboostBinary(self.t)
      classifier.train(self.data, binary_labels)
      self.strong_classifiers.append(classifier)
      # break
    print("Trained in", time.time() - start_time)
  
  def generate_haar_coords(self, width, height):
    two_rect = []
    three_rect = []
    four_rect = []
    for x in range(width):
      for y in range(height):
        for window_width in range(1, width):
          for window_height in range(1, height):
            # Coordinates for 2-rectangle features (split across x or y)
            if y + window_height <= height and x + 2 * window_width <= width:
              two_rect.append((y, x, y + window_height - 1, x + window_width - 1, 
                                  y, x + window_width, y + window_height - 1, x + 2 * window_width - 1))
            if y + 2 * window_height <= height and x + window_width <= width:
              two_rect.append((y, x, y + window_height - 1, x + window_width - 1, 
                                  y + window_height, x, y + 2 * window_height - 1, x + window_width - 1))
            # Coordinates for 3-rectangle features (split across x or y)
            if y + window_height <= height and x + 3 * window_width <= width:
              three_rect.append((y, x, y + window_height - 1, x + window_width - 1, 
                                  y, x + window_width, y + window_height - 1, x + 2 * window_width - 1,
                                  y, x + 2 * window_width, y + window_height - 1, x + 3 * window_width - 1))
            if y + 3 * window_height <= height and x + window_width <= width:
              three_rect.append((y, x, y + window_height - 1, x + window_width - 1, 
                                  y + window_height, x, y + 2 * window_height - 1, x + window_width - 1,
                                  y + 2 * window_height, x, y + 3 * window_height - 1, x + window_width - 1))
            # Coordinates for 4-rectangle features
            if y + 2 * window_height <= height and x + 2 * window_width <= width:
              four_rect.append((y, x, y + window_height - 1, x + window_width - 1, 
                                  y, x + window_width, y + window_height - 1, x + 2 * window_width - 1,
                                  y + window_height, x, y + 2 * window_height - 1, x + window_width - 1,
                                  y + window_height, x + window_width, y + 2 * window_height - 1, x + 2 * window_width - 1))
    self.haar_coords = np.random.choice(two_rect + three_rect + four_rect, self.num_haar)

  def extract_haar_features(self, image):
    haar_features = []
    gray_im = .2989 * image[:1024] + .5870 * image[1024:2048] + .1140 * image[2048:]
    ii = self.integral_image(np.array(gray_im).reshape(32, 32))
    for coord_set in self.haar_coords:
      if len(coord_set) == 8:
        rect1 = self.integrate_rect(ii, coord_set[0], coord_set[1], coord_set[2], coord_set[3])
        rect2 = self.integrate_rect(ii, coord_set[4], coord_set[5], coord_set[6], coord_set[7])
        haar_feat = rect2 - rect1
      elif len(coord_set) == 12:
        rect1 = self.integrate_rect(ii, coord_set[0], coord_set[1], coord_set[2], coord_set[3])
        rect2 = self.integrate_rect(ii, coord_set[4], coord_set[5], coord_set[6], coord_set[7])
        rect3 = self.integrate_rect(ii, coord_set[8], coord_set[9], coord_set[10], coord_set[11])
        haar_feat = rect2 - rect1 - rect3
      else:
        rect1 = self.integrate_rect(ii, coord_set[0], coord_set[1], coord_set[2], coord_set[3])
        rect2 = self.integrate_rect(ii, coord_set[4], coord_set[5], coord_set[6], coord_set[7])
        rect3 = self.integrate_rect(ii, coord_set[8], coord_set[9], coord_set[10], coord_set[11])
        rect4 = self.integrate_rect(ii, coord_set[12], coord_set[13], coord_set[14], coord_set[15])
        haar_feat = rect2 + rect4 - rect1 - rect3
      haar_features.append(haar_feat)
    return haar_features

  def integrate_rect(self, ii, top_left_y, top_left_x, bottom_right_y, bottom_right_x):
    return ii[bottom_right_y][bottom_right_x] + ii[top_left_y][top_left_x] - (ii[top_left_y][bottom_right_x] + ii[bottom_right_y][top_left_x])
  
  # Computes integral image
  def integral_image(self, image):
    ii = image
    ii = ii.cumsum(axis=0)
    ii = ii.cumsum(axis=1)
    return ii


def n_fold_cross_validation(n, k, data, labels):
    avg_acc = 0
    for i in range(1, n+1):
      train_data = data[:int(len(data) * (i - 1) / n)] + data[int(len(data) * i / n):]
      train_labels = labels[:int(len(labels) * (i - 1) / n)] + labels[int(len(labels) * i / n):]
      test_data = data[int(len(data) * (i - 1) / n):int(len(data) * i / n)]
      test_labels = labels[int(len(labels) * (i - 1) / n):int(len(labels) * i / n)]
      if args.model == 'KNN':
        model = KNN(k)
      else:
        model = Adaboost()
      model.train(train_data, train_labels)
      avg_acc += model.eval(False, test_data=test_data, test_labels=test_labels)
    return avg_acc / n

def find_best_k(args, data, labels):
  print("Finding Best K... (This could take some time)")
  best_k = 1
  best_k_acc = 0
  for i in range(3, 12, 2):
    acc = n_fold_cross_validation(args.n, i, data, labels)
    if acc > best_k_acc:
      best_k = i
      best_k_acc = acc
  print("Found Best k =", best_k)
  return best_k

def _parse_args():
    """
    Command-line arguments to the system. --model switches between KNN and Adaboost models
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='a5.py')
    parser.add_argument('--model', type=str, default='KNN', choices=['KNN', 'Adaboost'], help='model to run (KNN or Adaboost)')
    parser.add_argument('--k', type=int, default=10, help='number of nearest neighbors')
    parser.add_argument('--data_path', type=str, default='cifar-10-batches-py', help='path to cifar folder')
    parser.add_argument('--do_train', action='store_true', help='Whether to train Adaboost')
    parser.add_argument('--do_eval', action='store_true', help='Whether to evaluate model')
    parser.add_argument('--do_print', action='store_true', help='Whether to print eval output or not')
    parser.add_argument('--do_best_k', action='store_true', help='Whether to use n-fold cross validation to find best k')
    parser.add_argument('--n', type=int, default=5, help='number of folds for cross validation')
    parser.add_argument('--t', type=int, default=100, help='number of weak classifiers')
    parser.add_argument('--num_haar', type=int, default=2000, help='number of haar-like features')
    args = parser.parse_args()
    return args


def to_g(image):
  return .2989 * image[:1024] + .5870 * image[1024:2048] + .1140 * image[2048:]

if __name__ == '__main__':
    args = _parse_args()
    print(args)
    data, labels = load_train_data(args.data_path)

    # model = Adaboost(args.t, args.num_haar)
    # two, three, four = model.generate_haar_coords(32, 32)
    # print(len(two[0]), len(three[0]), len(four[0]))
    # all_coords = two + three + four
    # all_coords = np.random.choice(all_coords, 100)
    # print(all_coords)
    # print(two)
    # print(three)
    # print(four)

    # data = np.array([to_g(image) for image in data])
    # image = data[0]
    # print(image)
    # ii = integral_image(image.reshape(32, 32))
    # print(ii)
    # haar_coord, haar_feature_type = haar_like_feature_coord(32, 32)
    # print(len(haar_coord))
    # randomh = np.random.choice(haar_coord, 2000)
    # print(randomh)
    # haar_features = haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1])
    # print(len(np.random.choice(haar_features, 2000)))

    if args.model == 'Adaboost':
      model = Adaboost(args.t, args.num_haar)
    else:
      if args.do_best_k:
        args.k = find_best_k(args, data, labels)
      model = KNN(args.k)
    if args.do_train:
      model.train(data, labels)
    if args.do_eval:
      model.eval(args.do_print)

    
