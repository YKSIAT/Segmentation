from model_bn import *
from models import *
from data_preprocessing_3 import *
from loss import dice_coefficient_loss, dice_coefficient
import argparse
from keras.utils import multi_gpu_model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.layers import *
from keras.callbacks import ReduceLROnPlateau
import warnings
from utils import LossHistory
from BiseNet import bisenet_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

warnings.filterwarnings('ignore')
K.clear_session()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def plot_model_():
    # model_ = unet()
    model_ = unet_scSE()
    plot_model(model_, to_file='./structure/net.pdf', show_shapes=True)


def train(args):

    train_set = \
        generator_image_mask(args.train_root_path, args.batch_size)
    val_set = \
        generator_image_mask(args.val_root_path, args.batch_size)

    # model = multi_gpu_model(unet_scSE(), gpus=2)
    model = multi_gpu_model(unet_scSE_gap(), gpus=2)
    model.load_weights("unet_scSE_gap_300.h5")
    # plot_model(model, to_file='./structure/feature_attention_net.pdf', show_shapes=True)

    # model.compile(optimizer=optimizers.Adam(lr=1e-4), loss=dice_coefficient_loss,
    #               metrics=['accuracy', dice_coefficient])"Nadam"
    model.compile(optimizer=optimizers.Nadam(lr=1e-3), loss=dice_coefficient_loss,
                  metrics=['accuracy', dice_coefficient])

    history = LossHistory()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', min_lr=1e-5)
    model_checkpoint = ModelCheckpoint("unet_scSE_gap_300.h5", monitor='loss', verbose=1, save_best_only=True)
    # model_checkpoint = ModelCheckpoint("brats_se.h5", monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(train_set, steps_per_epoch=165, epochs=300, callbacks=[model_checkpoint, history, reduce_lr],
                        use_multiprocessing=True, validation_data=val_set, validation_steps=20, workers=20, verbose=1)

    # plot_model(model, to_file='./structure/model_test.png', show_shapes=True)

    history.loss_plot('epoch')


def statistics(slice_data):
    shape = slice_data.shape
    print("slice_data's shape {}".format(shape))
    print("value_0 {}".format(np.sum(slice_data == 0)))
    print("value_1 {}".format(np.sum(slice_data == 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--train_root_path', type=str, default="/home/yk/Project/keras/Unet/data_processed_norm/HGG",
                        help='the root directory containing the train dataset.')
    parser.add_argument('--val_root_path', type=str, default="/home/yk/Project/keras/Unet/data_processed_norm/LGG",
                        help='the root directory containing the test dataset.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size.')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Each volume have n_classes segmentation labels.')
    parser.add_argument('--output_h', type=int, default=240,
                        help='')
    parser.add_argument('--output_w', type=int, default=240,
                        help='')

    args = parser.parse_args()

    train(args)



