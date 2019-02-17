"""
Created on 2018/10/9 by Chun-hui Yin(yinchunhui.ahu@gmail.com).
Description: Script file for running our experiments on response-time QoS data.
"""
import multiprocessing
import os
import sys
from time import time
import argparse
from DataSet import DataSet
from Evaluator import evaluate, saveResult
import numpy as np
from keras import initializers
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten, concatenate, dot, Lambda
from keras.optimizers import Adam, Adamax
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.utils import plot_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    parser = argparse.ArgumentParser(description="Parameter Settings")
    parser.add_argument('--dataType', default='tp', type=str, help='Type of data:rt|tp.')
    parser.add_argument('--parallel', default=True, type=bool, help='Whether to use multi-process.')
    parser.add_argument('--density', default=list(np.arange(0.05, 0.31, 0.05)), type=list, help='Density of matrix.')
    parser.add_argument('--epochNum', default=50, type=int, help='Numbers of epochs per run.')
    parser.add_argument('--batchSize', default=256, type=int, help='Size of a batch.')
    parser.add_argument('--layers', default=[64, 32, 16, 8, 1], type=list, help='Layers of MLP.')
    parser.add_argument('--regLayers', default=[0, 0, 0, 0, 0], type=list, help='Regularizers.')
    parser.add_argument('--optimizer', default=Adam, type=str, help='The optimizer:Adam|Adamax|Nadam|Adagrad.')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate of the model.')
    parser.add_argument('--decay', default=0.0, type=float, help='Decay factor of learning rate.')
    parser.add_argument('--verbose', default=1, type=int, help='Iterations per evaluation.')
    parser.add_argument('--store', default=True, type=bool, help='Whether to store the model and result.')
    parser.add_argument('--modelPath', default='./Model', type=str, help='Path to save the model.')
    parser.add_argument('--resultPath', default='./Result', type=str, help='Path to save the result.')
    args = parser.parse_args()

    if args.parallel:
        pool = multiprocessing.Pool()
        for density in args.density:
            pool.apply_async(LDCF, (args, density))
        pool.close()
        pool.join()
    else:
        for density in args.density:
            LDCF(args, density)


class LDCF:

    def __init__(self, args, density):

        self.dataset = DataSet(args.dataType, density)
        self.dataType = self.dataset.dataType
        self.density = self.dataset.density
        self.shape = self.dataset.shape

        self.train = self.dataset.train
        self.test = self.dataset.test

        self.epochNum = args.epochNum
        self.batchSize = args.batchSize
        self.layers = args.layers
        self.regLayers = args.regLayers
        self.lr = args.lr
        self.decay = args.decay
        self.optimizer = args.optimizer
        self.verbose = args.verbose

        self.store = args.store
        self.modelPath = args.modelPath
        self.resultPath = args.resultPath

        self.model = self.compile_model()

        self.run()

    def run(self):
        # Initialization
        x_test, y_test = self.dataset.getTestInstance(self.test)
        sys.stdout.write('\rInitializing...')
        mae, rmse = evaluate(self.model, x_test, y_test)
        sys.stdout.write('\rInitializing completes.MAE = %.4f|RMSE = %.4f.\n' % (mae, rmse))
        best_mae, best_rmse, best_epoch = mae, rmse, -1
        evalResults = np.zeros((self.epochNum, 2))
        # Training model
        print('=' * 14 + 'Start Training' + '=' * 22)
        for epoch in range(self.epochNum):
            sys.stdout.write('\rEpoch %d starts...' % epoch)
            start = time()
            x_train, y_train = self.dataset.getTrainInstance(self.train)
            # Training
            history = self.model.fit(x_train, y_train, batch_size=self.batchSize, epochs=1, verbose=0, shuffle=True)
                                     #, callbacks=[TensorBoard(log_dir='./Log')])
            end = time()
            sys.stdout.write('\rEpoch %d ends.[%.1fs]' % (epoch, end - start))
            # Evaluation
            if epoch % self.verbose == 0:
                sys.stdout.write('\rEvaluating Epoch %d...' % epoch)
                mae, rmse = evaluate(self.model, x_test, y_test)
                loss = history.history['loss'][0]
                sys.stdout.write('\rEvaluating completes.[%.1fs] ' % (time() - end))
                if mae < best_mae:
                    best_mae, best_rmse, best_epoch = mae, rmse, epoch
                    if self.store:
                        self.saveModel(self.model)
                evalResults[epoch, :] = [mae, rmse]
                sys.stdout.write('\rEpoch %d : MAE = %.4f|RMSE = %.4f|Loss = %.4f\n' % (epoch, mae, rmse, loss))
        print('=' * 14 + 'Training Complete!' + '=' * 18)
        print('The best is at epoch %d : MAE = %.4f|RMSE = %.4f.' % (best_epoch, best_mae, best_rmse))
        if self.store:
            saveResult(self.resultPath, self.dataType, self.density, evalResults, ['MAE', 'RMSE'])
            print('The model is stored in %s.' % self.modelPath)
            print('The result is stored in %s.' % self.resultPath)

    def compile_model(self):

        _model = self.build_model(self.shape[0], self.shape[1], self.layers, self.regLayers)
        _model.compile(optimizer=self.optimizer(lr=self.lr, decay=self.decay), loss=[self.huber_loss])
        return _model

    def build_model(self, num_users, num_item, layers, reg_layers):

        assert len(layers) == len(reg_layers)

        # Input Layer
        user_id_input = Input(shape=(1,), dtype='int64', name='user_id_input')
        user_lc_input = Input(shape=(2,), dtype='int64', name='user_lc_input')

        item_id_input = Input(shape=(1,), dtype='int64', name='item_id_input')
        item_lc_input = Input(shape=(2,), dtype='int64', name='item_lc_input')

        user_id_embedding = self.getEmbedding(num_users, int(layers[0] / 4), 1, reg_layers[0], 'user_id_embedding')
        user_lc_embedding = self.getEmbedding(num_users, int(layers[0] / 4), 2, reg_layers[0], 'user_lc_embedding')

        item_id_embedding = self.getEmbedding(num_item, int(layers[0] / 4), 1, reg_layers[0], 'item_id_embedding')
        item_lc_embedding = self.getEmbedding(num_item, int(layers[0] / 4), 2, reg_layers[0], 'item_lc_embedding')

        user_id_latent = Flatten()(user_id_embedding(user_id_input))
        user_lc_latent = Flatten()(user_lc_embedding(user_lc_input))

        item_id_latent = Flatten()(item_id_embedding(item_id_input))
        item_lc_latent = Flatten()(item_lc_embedding(item_lc_input))

        # concatenate
        predict_user_vector = concatenate([user_id_latent, user_lc_latent])
        predict_item_vector = concatenate([item_id_latent, item_lc_latent])

        mlp_vector = concatenate([predict_user_vector, predict_item_vector])

        # AC-COS
        cosine_vector = dot([user_lc_latent, item_lc_latent], axes=1, normalize=True)

        # AC_EUC
        #euclidean_vector = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([user_lc_latent, item_lc_latent])

        # Middle Layer
        for index in range(1, len(layers) - 1):
            layer = Dense(units=layers[index], kernel_initializer=initializers.random_normal(),
                          kernel_regularizer=l2(reg_layers[index]), activation='relu', name='mlpLayer%d' % index)
            mlp_vector = layer(mlp_vector)

        predict_vector = concatenate([mlp_vector, cosine_vector])

        # Output layer
        prediction = Dense(units=layers[-1], activation='linear', kernel_initializer=initializers.lecun_normal(),
                           kernel_regularizer=l2(reg_layers[-1]), name='prediction')(predict_vector)

        _model = Model(inputs=[user_id_input, user_lc_input, item_id_input, item_lc_input], outputs=prediction)
        plot_model(_model, to_file='model.png')
        return _model

    def huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) < clip_delta
        squared_loss = 0.5 * K.square(error)
        linear_loss = clip_delta * (K.abs(error) - 0.5 * clip_delta)
        return K.tf.where(cond, squared_loss, linear_loss)

    # One-hot encoding + 0-layer mlp
    def getEmbedding(self, input_dim, output_dim, input_length, reg_layers, name):
        _Embedding = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length,
                               embeddings_initializer=initializers.random_normal(),
                               embeddings_regularizer=l2(reg_layers), name=name)
        return _Embedding

    def euclidean_distance(self, vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return shape1[0], 1

    def saveModel(self, _model):
        _model.save_weights(self.modelPath + '/%s_%.2f_%s.h5'
                            % (self.dataType, self.density, self.layers), overwrite=True)


if __name__ == '__main__':
    main()
