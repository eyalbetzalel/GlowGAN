import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

from imagegpt.model import model
from imagegpt.utils import count_parameters, color_quantize
# from model import model
# from utils import count_parameters, color_quantize


class ImageGPT:

    def __init__(
            self,
            batch_size,
            devices: list,
            # n_gpu=1,
            n_class=10,
            ckpt_path="../artifacts/model.ckpt-1000000",
            color_cluster_path="../artifacts/kmeans_centers.npy"
    ):
        self.hps = HParams(
            n_ctx=32*32,  # todo: pass image size
            n_embd=512,
            n_head=8,
            n_layer=24,
            n_vocab=512,  # todo: verify
            bert=False,
            bert_mask_prob=0.15,
            clf=False,
        )
        n_gpu = len(devices)
        self.devices = devices

        self.clusters = np.load(color_cluster_path)

        self.X = tf.placeholder(tf.int32, [batch_size, self.hps.n_ctx])
        self.Y = tf.placeholder(tf.float32, [batch_size, n_class])

        x = tf.split(self.X, n_gpu, 0)
        y = tf.split(self.Y, n_gpu, 0)

        self.trainable_params, self.gen_logits, self.gen_loss, self.clf_loss, self.tot_loss, self.accuracy = \
            self.create_model(x, y, devices=self.devices, hparams=self.hps)
        # self.reduce_mean(self.gen_loss, self.clf_loss, self.tot_loss, self.accuracy, n_gpu)

        self.saver = tf.train.Saver(var_list=[tp for tp in self.trainable_params if not 'clf' in tp.name])
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, ckpt_path)

    def eval_model(self, x: np.array, y=None, quantize=False):
        if quantize:
            x = self.color_quantize(x)
        return self.evaluate(self.sess, x, self.X, self.gen_loss)

    def color_quantize(self, x):
        """expect batch of images with channels last"""
        return self._color_quantize(x, np_clusters=self.clusters)

    @staticmethod
    def squared_euclidean_distance(a, b):
        b = np.transpose(b)
        a2 = np.sum(np.square(a), axis=2, keepdims=True)
        b2 = np.sum(np.square(b), axis=0, keepdims=True)
        ab = np.matmul(a, b)
        d = a2 - 2*ab + b2
        return d

    def _color_quantize(self, x, np_clusters):
        x = np.reshape(x, [x.shape[0], -1, 3])
        d = self.squared_euclidean_distance(x, np_clusters)
        return np.argmin(d, 2)

    @staticmethod
    def evaluate(sess, evX, X, gen_loss):
        # out = sess.run(gen_loss[0], {X: evX})
        out = sess.run(gen_loss, {X: evX})
        return out

    @staticmethod
    def reduce_mean(gen_loss, clf_loss, tot_loss, accuracy, n_gpu):
        with tf.device("/gpu:0"):
            for i in range(1, n_gpu):
                gen_loss[0] += gen_loss[i]
                clf_loss[0] += clf_loss[i]
                tot_loss[0] += tot_loss[i]
                accuracy[0] += accuracy[i]
            gen_loss[0] /= n_gpu
            clf_loss[0] /= n_gpu
            tot_loss[0] /= n_gpu
            accuracy[0] /= n_gpu

    @staticmethod
    def create_model(x, y, devices, hparams):
        gen_logits = []
        gen_loss = []
        clf_loss = []
        tot_loss = []
        accuracy = []

        trainable_params = None
        for i, device in enumerate(devices):
            with tf.device("/gpu:%d" % device):
                results = model(hparams, x[i], y[i], reuse=(i != 0))

                gen_logits.append(results["gen_logits"])
                gen_loss.append(results["gen_loss"])
                clf_loss.append(results["clf_loss"])

                if hparams.clf:
                    tot_loss.append(results["gen_loss"] + results["clf_loss"])
                else:
                    tot_loss.append(results["gen_loss"])

                accuracy.append(results["accuracy"])

                if i == 0:
                    trainable_params = tf.trainable_variables()
                    print("trainable parameters:", count_parameters())

        return trainable_params, gen_logits, gen_loss, clf_loss, tot_loss, accuracy


if __name__ == "__main__":
    bs = 32
    # image_gpt = ImageGPT(batch_size=bs, devices=[0], ckpt_path='../image-gpt/artifacts/model.ckpt-1000000', color_cluster_path='../image-gpt/artifacts/kmeans_centers.npy')
    image_gpt = ImageGPT(batch_size=bs, devices=[0], ckpt_path='../../image-gpt/artifacts/model.ckpt-1000000', color_cluster_path='../../image-gpt/artifacts/kmeans_centers.npy')
    # x_test = np.load("../../image-gpt/artifacts/cifar10_teX.npy")
    # # x = x_test[:bs]
    # for i in range(5):
    #     print(i)
    #     out = image_gpt.eval_model(x_test[i*bs:(i+1)*bs])

    X = np.random.randint(0, 255, size=bs*32*32*3).reshape((bs, 32, 32, 3))
    X = ((X / 255.) - .5) * 2.
    X = image_gpt.color_quantize(X)
    out = image_gpt.eval_model(X)
    print('debug')
