import logging
import math

import slidingwindow as sw

import cv2
import numpy as np
import tensorflow as tf
import time

from tf_pose import common
from tf_pose.common import CocoPart
from tf_pose.tensblur.smoother import Smoother
import datetime


## Initialize variables starts here
sess = None

def initialize_variables():
    global x,y,LABELS, n_steps, n_input, n_hidden, n_classes, pred,cost, pred, accuracy, optimizer,sess
    LABELS = ["NO ACTION", "JUMPING_JACKS", "BOXING", "waving 2 hands", "waving 1 hand", "clapping"]
    n_steps = 32
    n_input = 36
    n_hidden = 34
    n_classes = 6
    decaying_learning_rate = True
    learning_rate = 0.0025 #used if decaying_learning_rate set to False
    init_learning_rate = 0.005
    decay_rate = 0.96 #the base of the exponential in the decay
    decay_steps = 100000 #used in decay every 60000 steps with a base of 0.96

    global_step = tf.Variable(0, trainable=False)
    lambda_loss_amount = 0.0015
    batch_size = 512
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    pred = LSTM_RNN(x, weights, biases)
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    )
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2
    if decaying_learning_rate:
        learning_rate = tf.train.exponential_decay(init_learning_rate, global_step*batch_size, decay_steps, decay_rate, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step) # Adam Optimizer
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    test_losses = []
    test_accuracies = []
    train_losses = []
    train_accuracies = []
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, "./new_model/new_model.ckpt")
    print("Variables and model initialized")
    return (sess, accuracy, pred, optimizer) 


def getPose(X):
    one_hot_predictions, accuracy1, final_loss = sess.run(
        [pred, accuracy, cost],
        feed_dict={
                x: X,
                y: np.array([[0. , 0., 0., 0., 0., 2.]])
        }
    )
    #label = LABELS[np.argmax(one_hot_predictions[0])]
    score = np.amax(one_hot_predictions[0])
    
    if score > 4.5:
        label = label = LABELS[np.argmax(one_hot_predictions[0])]
    else:
        label = "No Action"
    #print("[", datetime.datetime.now(), "]: ",one_hot_predictions[0], label)
    return label

def load_X(X):
    X_ = np.asarray(X, dtype=np.float32)
    X_ = X_[np.newaxis, :, :]
    return X_

# Load the networks outputs

def load_y(y_path):
    file = open(y_path, 'r')
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # for 0-based indexing
    return y_ - 1

def LSTM_RNN(_X, _weights, _biases):
    # model architecture based on "guillaume-chevalier" and "aymericdamien" under the MIT license.

    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, n_input])   
    # Rectifies Linear Unit activation function used
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # A single output is produced, in style of "many to one" classifier, refer to http://karpathy.github.io/2015/05/21/rnn-effectiveness/ for details
    lstm_last_output = outputs[-1]
    
    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def extract_batch_size(_train, _labels, _unsampled, batch_size):
    # Fetch a "batch_size" amount of data and labels from "(X|y)_train" data. 
    # Elements of each batch are chosen randomly, without replacement, from X_train with corresponding label from Y_train
    # unsampled_indices keeps track of sampled data ensuring non-replacement. Resets when remaining datapoints < batch_size    
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)
    batch_labels = np.empty((batch_size,1)) 
    #print("#######",batch_s, batch_labels)
    for i in range(batch_size):
        # Loop index
        # index = random sample from _unsampled (indices)
        index = random.choice(_unsampled)
        #print("i",index)
        batch_s[i] = _train[index] 
        batch_labels[i] = _labels[index]
        _unsampled.remove(index)


    return batch_s, batch_labels, _unsampled


def one_hot(y_):
    # One hot encoding of the network outputs
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

## Code  ends here

try:
    from tf_pose.pafprocess import pafprocess
except ModuleNotFoundError as e:
    print(e)
    print('you need to build c++ library for pafprocess. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess')
    exit(-1)

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def _round(v):
    return int(round(v))


def _include_part(part_list, part_idx):
    for part in part_list:
        if part_idx == part.part_idx:
            return True, part
    return False, None


class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def get_face_box(self, img_w, img_h, mode=0):
        """
        Get Face box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :param mode:
        :return:
        """
        # SEE : https://github.com/ildoonet/tf-pose-estimation/blob/master/tf_pose/common.py#L13
        _NOSE = CocoPart.Nose.value
        _NECK = CocoPart.Neck.value
        _REye = CocoPart.REye.value
        _LEye = CocoPart.LEye.value
        _REar = CocoPart.REar.value
        _LEar = CocoPart.LEar.value

        _THRESHOLD_PART_CONFIDENCE = 0.2
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]

        is_nose, part_nose = _include_part(parts, _NOSE)
        if not is_nose:
            return None

        size = 0
        is_neck, part_neck = _include_part(parts, _NECK)
        if is_neck:
            size = max(size, img_h * (part_neck.y - part_nose.y) * 0.8)

        is_reye, part_reye = _include_part(parts, _REye)
        is_leye, part_leye = _include_part(parts, _LEye)
        if is_reye and is_leye:
            size = max(size, img_w * (part_reye.x - part_leye.x) * 2.0)
            size = max(size,
                       img_w * math.sqrt((part_reye.x - part_leye.x) ** 2 + (part_reye.y - part_leye.y) ** 2) * 2.0)

        if mode == 1:
            if not is_reye and not is_leye:
                return None

        is_rear, part_rear = _include_part(parts, _REar)
        is_lear, part_lear = _include_part(parts, _LEar)
        if is_rear and is_lear:
            size = max(size, img_w * (part_rear.x - part_lear.x) * 1.6)

        if size <= 0:
            return None

        if not is_reye and is_leye:
            x = part_nose.x * img_w - (size // 3 * 2)
        elif is_reye and not is_leye:
            x = part_nose.x * img_w - (size // 3)
        else:  # is_reye and is_leye:
            x = part_nose.x * img_w - size // 2

        x2 = x + size
        if mode == 0:
            y = part_nose.y * img_h - size // 3
        else:
            y = part_nose.y * img_h - _round(size / 2 * 1.2)
        y2 = y + size

        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if _round(x2 - x) == 0.0 or _round(y2 - y) == 0.0:
            return None
        if mode == 0:
            return {"x": _round((x + x2) / 2),
                    "y": _round((y + y2) / 2),
                    "w": _round(x2 - x),
                    "h": _round(y2 - y)}
        else:
            return {"x": _round(x),
                    "y": _round(y),
                    "w": _round(x2 - x),
                    "h": _round(y2 - y)}

    def get_upper_body_box(self, img_w, img_h):
        """
        Get Upper body box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :return:
        """

        if not (img_w > 0 and img_h > 0):
            raise Exception("img size should be positive")

        _NOSE = CocoPart.Nose.value
        _NECK = CocoPart.Neck.value
        _RSHOULDER = CocoPart.RShoulder.value
        _LSHOULDER = CocoPart.LShoulder.value
        _THRESHOLD_PART_CONFIDENCE = 0.3
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]
        part_coords = [(img_w * part.x, img_h * part.y) for part in parts if
                       part.part_idx in [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]]

        if len(part_coords) < 5:
            return None

        # Initial Bounding Box
        x = min([part[0] for part in part_coords])
        y = min([part[1] for part in part_coords])
        x2 = max([part[0] for part in part_coords])
        y2 = max([part[1] for part in part_coords])

        # # ------ Adjust heuristically +
        # if face points are detcted, adjust y value

        is_nose, part_nose = _include_part(parts, _NOSE)
        is_neck, part_neck = _include_part(parts, _NECK)
        torso_height = 0
        if is_nose and is_neck:
            y -= (part_neck.y * img_h - y) * 0.8
            torso_height = max(0, (part_neck.y - part_nose.y) * img_h * 2.5)
        #
        # # by using shoulder position, adjust width
        is_rshoulder, part_rshoulder = _include_part(parts, _RSHOULDER)
        is_lshoulder, part_lshoulder = _include_part(parts, _LSHOULDER)
        if is_rshoulder and is_lshoulder:
            half_w = x2 - x
            dx = half_w * 0.15
            x -= dx
            x2 += dx
        elif is_neck:
            if is_lshoulder and not is_rshoulder:
                half_w = abs(part_lshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)
            elif not is_lshoulder and is_rshoulder:
                half_w = abs(part_rshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)

        # ------ Adjust heuristically -

        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if _round(x2 - x) == 0.0 or _round(y2 - y) == 0.0:
            return None
        return {"x": _round((x + x2) / 2),
                "y": _round((y + y2) / 2),
                "w": _round(x2 - x),
                "h": _round(y2 - y)}

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()


class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()


class PoseEstimator:
    def __init__(self):
        pass

    @staticmethod
    def estimate_paf(peaks, heat_mat, paf_mat):
        pafprocess.process_paf(peaks, heat_mat, paf_mat)

        humans = []
        for human_id in range(pafprocess.get_num_humans()):
            human = Human([])
            is_added = False

            for part_idx in range(18):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue

                is_added = True
                human.body_parts[part_idx] = BodyPart(
                    '%d-%d' % (human_id, part_idx), part_idx,
                    float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[1],
                    float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[0],
                    pafprocess.get_part_score(c_idx)
                )

            if is_added:
                score = pafprocess.get_score(human_id)
                human.score = score
                humans.append(human)

        return humans


class TfPoseEstimator:
    # TODO : multi-scale

    def __init__(self, graph_path, target_size=(320, 240), tf_config=None):
        self.target_size = target_size

        # load graph
        logger.info('loading graph from %s(default size=%dx%d)' % (graph_path, target_size[0], target_size[1]))
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name='TfPoseEstimator')
        self.persistent_sess = tf.Session(graph=self.graph, config=tf_config)
        self.frame_number=0
        # for op in self.graph.get_operations():
        #     print(op.name)
        # for ts in [n.name for n in tf.get_default_graph().as_graph_def().node]:
        #     print(ts)

        self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator/image:0')
        self.tensor_output = self.graph.get_tensor_by_name('TfPoseEstimator/Openpose/concat_stage7:0')
        self.tensor_heatMat = self.tensor_output[:, :, :, :19]
        self.tensor_pafMat = self.tensor_output[:, :, :, 19:]
        self.upsample_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='upsample_size')
        self.tensor_heatMat_up = tf.image.resize_area(self.tensor_output[:, :, :, :19], self.upsample_size,
                                                      align_corners=False, name='upsample_heatmat')
        self.tensor_pafMat_up = tf.image.resize_area(self.tensor_output[:, :, :, 19:], self.upsample_size,
                                                     align_corners=False, name='upsample_pafmat')
        smoother = Smoother({'data': self.tensor_heatMat_up}, 25, 3.0)
        gaussian_heatMat = smoother.get_output()

        max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
        self.tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat,
                                     tf.zeros_like(gaussian_heatMat))

        self.heatMat = self.pafMat = None

        # warm-up
        self.persistent_sess.run(tf.variables_initializer(
            [v for v in tf.global_variables() if
             v.name.split(':')[0] in [x.decode('utf-8') for x in
                                      self.persistent_sess.run(tf.report_uninitialized_variables())]
             ])
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1], target_size[0]]
            }
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1] // 2, target_size[0] // 2]
            }
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1] // 4, target_size[0] // 4]
            }
        )

    def __del__(self):
        # self.persistent_sess.close()
        pass

    @staticmethod
    def _quantize_img(npimg):
        npimg_q = npimg + 1.0
        npimg_q /= (2.0 / 2 ** 8)
        # npimg_q += 0.5
        npimg_q = npimg_q.astype(np.uint8)
        return npimg_q

    @staticmethod
    def draw_humans(npimg, humans, frame_number,sequence_arr,imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        centers = {}
        row = []
        label = ''
        for human in humans:
            # draw point
            #print("HUMAN score::", human.score)
            if(human.score > 0.8):
                for i in range(common.CocoPart.Background.value):
                    if i not in human.body_parts.keys():
                       row.append(0)
                       row.append(0)  
                       continue

                    body_part = human.body_parts[i]
                    center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                    centers[i] = center
                    if (int(body_part.x * image_w + 0.5) > 0 and  int(body_part.y * image_h + 0.5) > 0):
                        row.append(int(body_part.x * image_w + 0.5))
                        row.append(int(body_part.y * image_h + 0.5))

                    cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=2, lineType=8, shift=0)

                # draw line
                for pair_order, pair in enumerate(common.CocoPairsRender):
                    if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                        continue

                # npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
                    cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 1)
            break
        if len(sequence_arr)<32 and len(row)>0:
               sequence_arr.append(row)
        elif len(row)>0:
               sequence_arr[frame_number%32]=row
               X=load_X(sequence_arr)
                        
               label=getPose(X)
               #print("", label) 
               #cv2.putText(npimg,label,(50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
        
        #print(row,frame_number,len(humans))      
        return sequence_arr,npimg, label

    def _get_scaled_img(self, npimg, scale):
        get_base_scale = lambda s, w, h: max(self.target_size[0] / float(h), self.target_size[1] / float(w)) * s
        img_h, img_w = npimg.shape[:2]

        if scale is None:
            if npimg.shape[:2] != (self.target_size[1], self.target_size[0]):
                # resize
                npimg = cv2.resize(npimg, self.target_size, interpolation=cv2.INTER_CUBIC)
            return [npimg], [(0.0, 0.0, 1.0, 1.0)]
        elif isinstance(scale, float):
            # scaling with center crop
            base_scale = get_base_scale(scale, img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)

            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.target_size[1] or npimg.shape[1] < self.target_size[0]:
                newimg = np.zeros(
                    (max(self.target_size[1], npimg.shape[0]), max(self.target_size[0], npimg.shape[1]), 3),
                    dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.target_size[0], self.target_size[1], 0.2)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios
        elif isinstance(scale, tuple) and len(scale) == 2:
            # scaling with sliding window : (scale, step)
            base_scale = get_base_scale(scale[0], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)
            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.target_size[1] or npimg.shape[1] < self.target_size[0]:
                newimg = np.zeros(
                    (max(self.target_size[1], npimg.shape[0]), max(self.target_size[0], npimg.shape[1]), 3),
                    dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            window_step = scale[1]

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.target_size[0], self.target_size[1],
                                  window_step)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios
        elif isinstance(scale, tuple) and len(scale) == 3:
            # scaling with ROI : (want_x, want_y, scale_ratio)
            base_scale = get_base_scale(scale[2], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)
            ratio_w = self.target_size[0] / float(npimg.shape[1])
            ratio_h = self.target_size[1] / float(npimg.shape[0])

            want_x, want_y = scale[:2]
            ratio_x = want_x - ratio_w / 2.
            ratio_y = want_y - ratio_h / 2.
            ratio_x = max(ratio_x, 0.0)
            ratio_y = max(ratio_y, 0.0)
            if ratio_x + ratio_w > 1.0:
                ratio_x = 1. - ratio_w
            if ratio_y + ratio_h > 1.0:
                ratio_y = 1. - ratio_h

            roi = self._crop_roi(npimg, ratio_x, ratio_y)
            return [roi], [(ratio_x, ratio_y, ratio_w, ratio_h)]

    def _crop_roi(self, npimg, ratio_x, ratio_y):
        target_w, target_h = self.target_size
        h, w = npimg.shape[:2]
        x = max(int(w * ratio_x - .5), 0)
        y = max(int(h * ratio_y - .5), 0)
        cropped = npimg[y:y + target_h, x:x + target_w]

        cropped_h, cropped_w = cropped.shape[:2]
        if cropped_w < target_w or cropped_h < target_h:
            npblank = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)

            copy_x, copy_y = (target_w - cropped_w) // 2, (target_h - cropped_h) // 2
            npblank[copy_y:copy_y + cropped_h, copy_x:copy_x + cropped_w] = cropped
        else:
            return cropped

    def inference(self, npimg, resize_to_default=True, upsample_size=1.0):
        if npimg is None:
            raise Exception('The image is not valid. Please check your image exists.')

        if resize_to_default:
            upsample_size = [int(self.target_size[1] / 8 * upsample_size), int(self.target_size[0] / 8 * upsample_size)]
        else:
            upsample_size = [int(npimg.shape[0] / 8 * upsample_size), int(npimg.shape[1] / 8 * upsample_size)]

        if self.tensor_image.dtype == tf.quint8:
            # quantize input image
            npimg = TfPoseEstimator._quantize_img(npimg)
            pass

        logger.debug('inference+ original shape=%dx%d' % (npimg.shape[1], npimg.shape[0]))
        img = npimg
        if resize_to_default:
            img = self._get_scaled_img(npimg, None)[0][0]
        peaks, heatMat_up, pafMat_up = self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up], feed_dict={
                self.tensor_image: [img], self.upsample_size: upsample_size
            })
        peaks = peaks[0]
        self.heatMat = heatMat_up[0]
        self.pafMat = pafMat_up[0]
        logger.debug('inference- heatMat=%dx%d pafMat=%dx%d' % (
            self.heatMat.shape[1], self.heatMat.shape[0], self.pafMat.shape[1], self.pafMat.shape[0]))

        t = time.time()
        humans = PoseEstimator.estimate_paf(peaks, self.heatMat, self.pafMat)
        logger.debug('estimate time=%.5f' % (time.time() - t))
        return humans


if __name__ == '__main__':
    import pickle

    f = open('./etcs/heatpaf1.pkl', 'rb')
    data = pickle.load(f)
    logger.info('size={}'.format(data['heatMat'].shape))
    f.close()

    t = time.time()
    humans = PoseEstimator.estimate_paf(data['peaks'], data['heatMat'], data['pafMat'])
    dt = time.time() - t;
    t = time.time()
    logger.info('elapsed #humans=%d time=%.8f' % (len(humans), dt))
