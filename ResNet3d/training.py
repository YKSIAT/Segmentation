from ResNet3D import DataGen3D
from ResNet3D.ResUNet3D import ResUNet
from utils import LossHistory
from keras.models import Model
from keras import optimizers
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from loss import dice_coefficient_loss, dice_coefficient, binary_focal_loss
from keras.callbacks import ReduceLROnPlateau
import argparse
import os

# from ResNet3D.loss_3d import binary_focal_loss


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def train(arg):
    train_set =\
        DataGen3D.Gen3D(root_path=arg.root_path, patch_shape=[160, 160, 128], batch_size=10)
    input_layer, output = ResUNet()
    model = Model(inputs=input_layer, outputs=output)
    paralleled_model = multi_gpu_model(model, gpus=2)
    paralleled_model.compile(optimizer=optimizers.adam(lr=args.learning_rate), loss=args.loss, metrics=args.metrics)
    history = LossHistory()
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, mode='auto', min_lr=1e-6)
    # early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None,
    #                            restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(args.parallel_model_parameter_save_path,
                                       monitor='loss', verbose=1, save_best_only=True)
    paralleled_model.fit_generator(train_set, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs,
                                   callbacks=[model_checkpoint, history, reduce_lr], verbose=1,
                                   workers=args.workers, use_multiprocessing=True)
    paralleled_model.load_weights(args.parallel_model_parameter_save_path)
    model.save(args.model_parameter_save_path)
    history.loss_plot('epoch')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--root_path', type=str,
                        default="/home/yk/Project/keras/dataset/BraTs19DataSet"
                                "/BraTs19Mixture_N4_HM_Norm/all/*",
                        help='the root directory containing the train dataset.')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size.')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--steps_per_epoch', type=int, default=36)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--workers', type=int, default=30)
    parser.add_argument('--metrics', default=[dice_coefficient])
    parser.add_argument('--loss', default=dice_coefficient_loss)

    parser.add_argument('--model_parameter_save_path', type=str,
                        # default="./checkpoint_new2/dfn_norm_wt_710_001.h5",
                        default="./checkpoint_813/model3d_825_01.h5",
                        help='The path to the model save')
    parser.add_argument('--parallel_model_parameter_save_path', type=str,
                        default="./checkpoint_813/parallel_model3d_825_01.h5",
                        help='The path to the parallel model save')

    parser.add_argument('--load_model_parameters_path', type=str,
                        default="./checkpoint_813/model3d_825_01.h5",
                        help='The path to the model save')
    args = parser.parse_args()
    train(args)



