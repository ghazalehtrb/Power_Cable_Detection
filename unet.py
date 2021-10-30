import tensorflow as tf
import layers
import os
import matplotlib.image as mp
import numpy as np
import random
import termcolor
from scipy.misc import imsave
import time
from losses import *
import matplotlib.pyplot as plt
from skimage.transform import resize

# def threshold(image,t):
#     for i in range(224):
#         for j in range(224):
#             if image[i][j] < t:
#                 image[i][j] = 0.
#     return image

class Unet:
    graph_save = ''
    weight_save = ''
    conditional_detection = True
    DIM = 224
    # block = None
    detection_mode = 0
    EPOCH = 50000000
    sd_lr = 1e-4
    BATCH_SIZE = 4
    INIT_LEARNING_RATE = 1e-4

    mask_path = 'all_data1/train/mask/'
    image_path = 'all_data1/train/image/'
    test_image_path = 'all_data1/test/image/'
    test_mask_path = 'all_data1/test/mask/'

    def __init__(self, graph_save='./graph_save/', weight_save='./results/weight/', activation='prelu',
                 block='conv', agg='sum', EPOCH=50000000, BATCH_SIZE=3,
                 conditional_detection=True, color_space='HSV', main_input='RGB',
                 use_gpu=True, DIM=224, detection_mode=0,
                 sd_lr=1e-2, beta1=0.9, beta2=0.999, optimizer='Adam'):

        tf.set_random_seed(1111)  ##todo

        tf.reset_default_graph()

        self.summary_path = './results/summary/'
        self.agg = agg
        self.activation = activation
        self.block_type = block
        self.conditional_detection = conditional_detection
        self.color_space = color_space
        self.main_input = main_input
        self.use_gpu = use_gpu
        self.weight_save = weight_save
        self.graph_save = graph_save
        self.DIM = DIM
        self.optimizer = optimizer
        self.sd_lr = sd_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.BATCH_SIZE = BATCH_SIZE

        self.image = tf.placeholder(dtype='float', shape=[None, self.DIM, self.DIM, 3],
                                    name='Input_1')
        self.mask = tf.placeholder(dtype='float', shape=[None, self.DIM, self.DIM, 2],
                                   name='Mask')
        # self.detection_mode = detection_mode  ##todo
        self.EPOCH = EPOCH

    @staticmethod
    def lr_decay(lr_input, decay_rate, num_epoch):  ##todo
        return lr_input / (1 + decay_rate * num_epoch)

    @staticmethod
    def _process_labels(label):
        labels = np.zeros((224, 224, 2), dtype=np.float32)
        for i in range(224):
            for j in range(224):
                if label[i][j] == 0:
                    labels[i, j, 0] = 1
                else:
                    labels[i, j, 1] = 1
        return labels

    @staticmethod
    def normalize(image):
        image = np.array(image, dtype=np.float32) / 255
        return image

    def build_net(self):
        with tf.name_scope('IlluminatorNet'):
            l1 = layers.conv_layer(self.image, [3, 3, 3, 64], 1, 'conv_1', 'relu')
            l2 = layers.conv_layer(l1, [3, 3, 64, 64], 1, 'conv_2', 'relu')
            p1 = tf.layers.max_pooling2d(inputs=l2, pool_size=[2, 2], strides=2)

            l3 = layers.conv_layer(p1, [3, 3, 64, 128], 1, 'conv_3', 'relu')
            l4 = layers.conv_layer(l3, [3, 3, 128, 128], 1, 'conv_4', 'relu')
            p2 = tf.layers.max_pooling2d(inputs=l4, pool_size=[2, 2], strides=2)

            l5 = layers.conv_layer(p2, [3, 3, 128, 256], 1, 'conv_5', 'relu')
            l6 = layers.conv_layer(l5, [3, 3, 256, 256], 1, 'conv_6', 'relu')
            p3 = tf.layers.max_pooling2d(inputs=l6, pool_size=[2, 2], strides=2)

            l7 = layers.conv_layer(p3, [3, 3, 256, 512], 1, 'conv_7', 'relu')
            l8 = layers.conv_layer(l7, [3, 3, 512, 512], 1, 'conv_8', 'relu')
            d1 = tf.layers.dropout(l8)
            p4 = tf.layers.max_pooling2d(inputs=d1, pool_size=[2, 2], strides=2)
            ##todo
            l9 = layers.conv_layer(p4, [3, 3, 512, 1024], 1, 'conv_9', 'relu')
            l10 = layers.conv_layer(l9, [3, 3, 1024, 1024], 1, 'conv_10', 'relu')
            d2 = tf.layers.dropout(l10)

            up1 = tf.layers.conv2d_transpose(d2, 1024, (1, 1), 2, padding='SAME', name='up1')
            l11 = layers.conv_layer(up1, [2, 2, 1024, 512], 1, 'conv_11', 'relu')
            merge1 = tf.concat([d1, l11], axis=-1, name='concat1')
            l12 = layers.conv_layer(merge1, [3, 3, 1024, 512], 1, 'conv_12', 'relu')
            l13 = layers.conv_layer(l12, [3, 3, 512, 512], 1, 'conv_13', 'relu')
            up2 = tf.layers.conv2d_transpose(l13, 512, (1, 1), 2, padding='SAME', name='up2')
            l14 = layers.conv_layer(up2, [2, 2, 512, 256], 1, 'conv_14', 'relu')
            merge2 = tf.concat([l6, l14], axis=-1, name='concat2')
            l15 = layers.conv_layer(merge2, [3, 3, 512, 256], 1, 'conv_15', 'relu')
            l16 = layers.conv_layer(l15, [3, 3, 256, 256], 1, 'conv_16', 'relu')
            up3 = tf.layers.conv2d_transpose(l16, 256, (1, 1), 2, padding='SAME', name='up3')
            l17 = layers.conv_layer(up3, [2, 2, 256, 128], 1, 'conv_17', 'relu')
            merge3 = tf.concat([l4, l17], axis=-1, name='concat3')
            l18 = layers.conv_layer(merge3, [3, 3, 256, 128], 1, 'conv_18', 'relu')
            l19 = layers.conv_layer(l18, [3, 3, 128, 128], 1, 'conv_19', 'relu')
            up4 = tf.layers.conv2d_transpose(l19, 128, (1, 1), 2, padding='SAME', name='up4')
            l20 = layers.conv_layer(up4, [2, 2, 128, 64], 1, 'conv_20', 'relu')
            merge4 = tf.concat([l2, l20], axis=-1, name='concat4')
            l21 = layers.conv_layer(merge4, [3, 3, 128, 64], 1, 'conv_21', 'relu')
            l22 = layers.conv_layer(l21, [3, 3, 64, 64], 1, 'conv_22', 'relu')
            l23 = layers.conv_layer(l22, [3, 3, 64, 2], 1, 'conv_23', 'linear')

            return l23

    def train(self):

        with tf.name_scope('Model'):

            logits = self.build_net()

            tf.summary.image('wire Detection 1', tf.reshape(logits[..., 1], [-1, 224, 224, 1]),self.BATCH_SIZE)
            tf.summary.image('wire Detection', tf.reshape(self.mask[..., 1], [-1, 224, 224, 1]),self.BATCH_SIZE)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.mask, logits=logits))
            tf.summary.scalar('Cost', loss)
            with tf.name_scope('Optimization'):
                sd_lr = tf.placeholder(tf.float32, name='learn_rate_sd')
                tf.summary.scalar('Learning Rate', sd_lr)
                optr = tf.train.AdamOptimizer(learning_rate=sd_lr) ##todo
                train_opt = optr.minimize(loss)

            saver = tf.train.Saver()
            global_init = tf.global_variables_initializer()
            merged_summary = tf.summary.merge_all()

        epoch = 0
        is_train = False
        with tf.Session() as sess:

            writer = tf.summary.FileWriter(self.summary_path, graph=tf.get_default_graph())

            if is_train:
                sess.run([global_init])
                #saver.restore(sess, self.weight_save + 'Model-' + str(30000) + '.ckpt')
                names = os.listdir(self.image_path)
                random.shuffle(names)
                for ep in range(self.EPOCH):
                    print(termcolor.colored('Epoch:', 'green'), ep)
                    random.shuffle(names)

                    for bn in range(int(len(names) / self.BATCH_SIZE)):
                        masks = [self._process_labels(mp.imread(self.mask_path + name[:-4] + '.png')) for name in
                                 names[bn * self.BATCH_SIZE:(bn + 1) * self.BATCH_SIZE]]
                        ##todo
                        img = [self.normalize(mp.imread(self.image_path + name)) for name in
                               names[bn * self.BATCH_SIZE:(bn + 1) * self.BATCH_SIZE]]

                        print('Batch', bn + 1, 'loaded.')
                        _, sd = sess.run([train_opt, loss], feed_dict={
                            self.image: img,
                            self.mask: masks,
                            sd_lr: self.sd_lr
                        })
                        print('Epoch:', ep, 'Batch:', bn, 'Learning Rate:', self.sd_lr, 'Cost:', sd)
                        ##todo
                        if epoch < 30000:
                            self.sd_lr = 0.0001
                        if 60000 > epoch >= 30000:
                            self.sd_lr = 0.00001
                        if 60000 <= epoch < 90000:
                            self.sd_lr = 0.00001 / 3
                        if ep >= 90000:
                            self.sd_lr = 0.000001

                        if (epoch % 30) == 0:
                            test_names = os.listdir(self.test_image_path)
                            random.shuffle(test_names)
                            test_masks = [self._process_labels(mp.imread(self.test_mask_path + name[:-4] + '.png')) for
                                          name in
                                          test_names[:self.BATCH_SIZE]]
                            test_hsv = [self.normalize(mp.imread(self.test_image_path + name)) for name in
                                        test_names[:self.BATCH_SIZE]]
                            summary, tl = sess.run([merged_summary, loss], feed_dict={
                                self.image: test_hsv,
                                self.mask: test_masks,
                                sd_lr: self.sd_lr})
                            writer.add_summary(summary, epoch)
                            print('Epoch:', ep, 'Batch:', bn, termcolor.colored('Test Error:',
                                                                                'blue'), tl, 'Learning Rate:',
                                  self.sd_lr)
                            print('summary was updated.')

                        if (epoch % 1000) == 0:
                            saver.save(sess, self.weight_save + 'Model-' + str(epoch) + '.ckpt')
                            print('Model was saved successfully in epoch ' + str(epoch))
                        epoch += 1
            else:
                saver.restore(sess, self.weight_save + 'Model-' + str(98000) + '.ckpt')
                bd = 'all_data1/test/image/'
                images = [(name,mp.imread(bd + name)) for name in os.listdir(bd)]
                for image in images:
                    tic = time.clock()
                    pics = sess.run(logits, feed_dict={self.image: [self.normalize(image[1])]})[0]
                    p = 1/(1+np.exp(-pics))
                    mask = np.argmax(p,2)
                    toc = time.clock()
                    print(tic-toc)
                    imsave('all_data1/test/preds/'+ image[0][:-4]+'.png',mask,format='png')
                    print(image[0])


if __name__ == '__main__':
    model = Unet(BATCH_SIZE=3, sd_lr=0.00001)
    model.train()
