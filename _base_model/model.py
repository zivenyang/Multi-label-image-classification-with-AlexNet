# coding=utf-8:
from __future__ import absolute_import, division, print_function
import sys
from _base_model.AlexNet import *
from _base_model.mini_batch import *
import os


class BaseModel(object):
    def __init__(self, images_file, labels_file, val_images_file, val_labels_file, log_dir, is_train, save_dir, save_path, val_result_file):
        self.images_file = images_file
        self.labels_file = labels_file
        self.val_images_file = val_images_file
        self.val_labels_file = val_labels_file
        self.batch_size = 48 if is_train is True else 1
        self.img_size = 128
        self.num_channel = 3
        self.num_epochs = 10
        self.learning_rate = 10e-5
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.decay = 0.9
        self.cell_size = 512
        self.num_class = 81
        self.is_train = is_train
        self.top_k = 4
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.save_path = save_path
        self.val_result_file = val_result_file
        self.build()
        self.saver = tf.train.Saver()

    def build(self):
        raise NotImplementedError()

    def get_feed_dict(self, batch, is_train):
        raise NotImplementedError()

    def train(self):
        print("Training...")
        _train_data = train_data(images_file=self.images_file, labels_file=self.labels_file, batch_size=self.batch_size)
        sess = tf.Session()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        sess.run(self.init)
        num_epochs = self.num_epochs
        for epoch in range(num_epochs):
            ave_cost = 0
            ave_precision = 0
            ave_recall = 0
            ave_F1 = 0
            ave_precision_k = 0
            ave_nDCG = 0
            for idx in  range(_train_data.num_batches):
                batch = _train_data.next_batch()
                _, train_labels = batch

                feed_dict = self.get_feed_dict(batch=batch, is_train=self.is_train)

                _, cost, topk_idx = sess.run([self.optimizer, self.cost, self.topk_idx], feed_dict)

                pridects = np.zeros(shape=train_labels.shape, dtype=float)
                precision = 0
                recall = 0
                precision_k = 0
                GT_topk = []
                for i in range(topk_idx.shape[0]):
                    for j in range(topk_idx.shape[1]):
                        pridects[i][topk_idx[i][j]] = 1.
                        GT_topk.append(train_labels[i][topk_idx[i][j]])
                nDCG = self.nDCG(topk_idx=topk_idx, labels_data=train_labels)
                _GT_topk = np.array(GT_topk).reshape([self.batch_size, -1])

                for i in range(self.batch_size):
                    precision += np.sum(np.multiply(pridects[i], train_labels[i])) / np.sum(pridects[i])
                    recall += np.sum(np.multiply(pridects[i], train_labels[i])) / np.sum(train_labels[i])
                    precision_k += np.sum(_GT_topk[i]) / min(topk_idx.shape[1], np.sum(train_labels[i]))
                precision /= topk_idx.shape[0]
                recall /= topk_idx.shape[0]
                precision_k /= topk_idx.shape[0]

                if precision + recall == 0:
                    F1 = 0
                else:
                    F1 = (2 * precision * recall) / (precision + recall)

                print("======> epoch:%dth, batch:%dth <=======" % ((epoch + 1), (idx + 1)))
                print("cost=%.3f" % cost)
                print("precision=%.3f" % precision)
                print("recall=%.3f" % recall)
                print("nDCG=%.3f" % nDCG)
                print("precision_k=%.3f" % precision_k)
                print("F1=%.3F" % F1)
                ave_cost += cost
                ave_precision += precision
                ave_recall += recall
                ave_F1 += F1
                ave_nDCG += nDCG
                ave_precision_k += precision_k
                if (idx + 1) % 10 == 0:
                    result = sess.run(merged, feed_dict)
                    writer.add_summary(summary=result, global_step=idx + 1)
                    print("+-----------------------------------+")
                    print("|--------> ave_cost=%.3f <--------|" % (ave_cost / (idx + 1)))
                    print("|------> ave_precision=%.3f <------|" % (ave_precision / (idx + 1)))
                    print("|-------> ave_recall=%.3f <--------|" % (ave_recall / (idx + 1)))
                    print("|---------> ave_F1=%.3f <----------|" % (ave_F1 / (idx + 1)))
                    print("|-----> ave_precision_k=%.3f <-----|" % (ave_precision_k / (idx + 1)))
                    print("|-------> ave_nDCG@%d=%.3f <-------|" % (topk_idx.shape[1], (ave_nDCG / (idx + 1))))
                    print("+-----------------------------------+")

            _train_data.reset()

        self.save(sess)
        sess.close()

        print("Training complete.")

    def val(self):
        """ Validate the model. """
        print("Validating the model ...")
        _val_data = val_data(images_file=self.val_images_file, labels_file=self.val_labels_file)
        sess = tf.Session()
        self.load(sess)
        ave_cost = 0
        ave_precision = 0
        ave_recall = 0
        ave_F1 = 0
        ave_precision_k = 0
        ave_nDCG = 0
        save_result = []
        result_file = self.val_result_file
        for idx in range(_val_data.num_batches):
            batch = _val_data.next_batch()
            images_data, labels_data = batch
            feed_dict = self.get_feed_dict(batch=batch, is_train=False)
            cost = sess.run(self.cost, feed_dict)
            topk_idx = sess.run(self.topk_idx, feed_dict)
            pridects = np.zeros(shape=labels_data.shape, dtype=float)
            precision = 0
            recall = 0
            precision_k = 0
            GT_topk = []
            for i in range(topk_idx.shape[0]):
                for j in range(topk_idx.shape[1]):
                    pridects[i][topk_idx[i][j]] = 1.
                    GT_topk.append(labels_data[i][topk_idx[i][j]])
                save_result.append(pridects[i])

            nDCG = self.nDCG(topk_idx=topk_idx, labels_data=labels_data)
            _GT_topk = np.array(GT_topk).reshape([self.batch_size, -1])

            for i in range(self.batch_size):
                precision += np.sum(np.multiply(pridects[i], labels_data[i])) / np.sum(pridects[i])
                recall += np.sum(np.multiply(pridects[i], labels_data[i])) / np.sum(labels_data[i])
                precision_k += np.sum(_GT_topk[i]) / min(topk_idx.shape[1], np.sum(labels_data[i]))
            precision /= topk_idx.shape[0]
            recall /= topk_idx.shape[0]
            precision_k /= topk_idx.shape[0]

            if precision + recall == 0:
                F1 = 0
            else:
                F1 = (2 * precision * recall) / (precision + recall)


            print("======> batch:%dth <=======" % (idx + 1))
            print("cost=%.3f" % cost)
            print("precision=%.3f" % precision)
            print("recall=%.3f" % recall)
            print("nDCG=%.3f" % nDCG)
            print("precision_k=%.3f" % precision_k)
            print("F1=%.3F" % F1)
            ave_cost += cost
            ave_precision += precision
            ave_recall += recall
            ave_F1 += F1
            ave_nDCG += nDCG
            ave_precision_k += precision_k
            if (idx + 1) % 10 == 0:
                print("+-----------------------------------+")
                print("|--------> ave_cost=%.3f <--------|" % (ave_cost / (idx + 1)))
                print("|------> ave_precision=%.3f <------|" % (ave_precision / (idx + 1)))
                print("|-------> ave_recall=%.3f <--------|" % (ave_recall / (idx + 1)))
                print("|---------> ave_F1=%.3f <----------|" % (ave_F1 / (idx + 1)))
                print("|-----> ave_precision_k=%.3f <-----|" % (ave_precision_k / (idx + 1)))
                print("|-------> ave_nDCG@%d=%.3f <-------|" % (topk_idx.shape[1], (ave_nDCG / (idx + 1))))
                print("+-----------------------------------+")
        save_result = np.array(save_result, dtype=int)

        print("Saving %s..." % result_file)
        np.save(file=result_file, arr=save_result)
        print("%s saved" % result_file)
        _val_data.reset()
        sess.close()

        print("Validation complete.")


    def save(self, sess):
        """ Save the model """
        save_path = os.path.join(self.save_dir, self.save_path)
        print(("Saving model to %s") % save_path)
        self.saver.save(sess, save_path)

    def load(self, sess):
        """ Load the model. """
        print("Loading model...")
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        if checkpoint is None:
            print("Error: No saved model found. Please train first.")
            sys.exit(0)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)


    def nDCG(self, topk_idx, labels_data):
        k = topk_idx.shape[1]
        DCG = []
        _iDCG = []
        iDCG = []
        for i in range(topk_idx.shape[0]):
            num_positive = 0
            for j in range(topk_idx.shape[1]):
                rel_i = labels_data[i][topk_idx[i][j]]
                DCG.append(rel_i / np.log2(1+j+1))
                if int(rel_i) is 1:
                    num_positive += 1
            if num_positive == 0:
                _iDCG.append(0.)
            else:
                for min_k in range(num_positive):
                    _iDCG.append(1 / np.log2(1+min_k+1))
            iDCG.append(_iDCG)
        DCG = np.array(DCG).reshape([topk_idx.shape[0], k])
        DCG = np.sum(DCG, 1)
        iDCG_sum = []
        for i in iDCG:
            iDCG_sum.append(np.sum(i))
        iDCG_result = []
        for j in iDCG_sum:
            if j == 0:
                iDCG_result.append(j)
            else:
                iDCG_result.append(1/j)
        _nDCG = np.multiply(DCG, iDCG_result)
        result = np.sum(_nDCG) / topk_idx.shape[0]
        return result


if __name__ == '__main__':
    is_train = False

    model = cnn(images_file='../train_images_no0.npy',
                labels_file='../train_labels_no0.npy',
                val_images_file='../test_images_no0.npy',
                val_labels_file='../test_labels_no0.npy',
                log_dir='./log',
                is_train=is_train,
                save_dir='./models/models',
                save_path='checkpoint.ckpt',
                val_result_file='./val_result.npy')
    if is_train is True:
        model.train()
    else:
        model.val()