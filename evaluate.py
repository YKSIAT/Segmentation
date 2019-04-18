from model_bn import *
from models import *
from data_preprocessing_3 import *
# from loss import calculate_metrics
from utils import show_image_color, plot_fusion_image
from loss_1 import calculate_metrics, dice_coefficient
import argparse
from keras.utils import multi_gpu_model
from utils import show_img_multiplex, transform
from keras.layers import *
import warnings
warnings.filterwarnings('ignore')
K.clear_session()


def evaluate():

    # model = multi_gpu_model(feature_attention_net(), gpus=2)
    model = multi_gpu_model(unet_scSE_gap(), gpus=2)
    # model.load_weights('brats_se_2.h5')

    model.load_weights("unet_scSE_gap_300.h5")

    # model.load_weights("feature_attention_net_2_300.h5")
    # colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(4)]
    val_data = generator_image_mask_val("/home/yk/Project/keras/Unet/data_processed_norm/LGG/image",
                                        "/home/yk/Project/keras/Unet/data_processed_norm/LGG/mask")
    counter = 0
    dice = 0
    ave_dice = 0
    sens = 0
    spec = 0
    ave_sens = 0
    ave_spec = 0

    for index, (img, true_mask) in enumerate(val_data):
        counter += 1
        predict_mask = model.predict(img, batch_size=1, verbose=0, steps=None)[0]

        print("img.shape", img.shape)
        # print("true_mask.shape", true_mask.shape)
        # print("predict_mask.shape", predict_mask.shape)
        true_mask_ = true_mask[0, :, :]
        predict_mask_ = predict_mask[:, :, 0]
        # np.save("true_mask.npy", true_mask_)
        # np.save("pre_mask.npy", true_mask_)

        # show_img_multiplex(predict_mask_, true_mask_)
        # show_img_multiplex((predict_mask_ >= 0.238), true_mask_)
        show_img_multiplex(predict_mask_, true_mask_)
        dice_ = dice_coefficient(true_mask_.astype(np.float32), predict_mask_.astype(np.float32))
        # dice_ = dice_coefficient(true_mask_.astype(np.float32), (predict_mask_ >= 0.238).astype(np.float32))
        print(counter)
        print("*******counter{}'dice is {}******".format(counter, dice_))
        dice = dice + dice_
        print("*******counter{}'average dice is {}******".format(counter, dice/counter))
        # sens_ = sensitivity_2(true_mask_.astype(np.float32), predict_mask_.astype(np.float32))
        # print("sens_", sens_)
        # spec_ = spec_2(predict_mask_.astype(np.float32), true_mask_.astype(np.float32))
        # print("spec_", spec_)

        specificity, sensitivity_, dice_coefficient_, hausdorff_ = \
            calculate_metrics(predict_mask_, true_mask_, optimal_threshold=0.23)
        sens = sens + sensitivity_
        spec = spec + specificity
        # show_img_multiplex(predict_mask_, true_mask_)
        # predict_mask_ = transform(predict_mask_, 0.23)
        # true_mask_show = show_image_color(true_mask_)
        # pre_mask_show = show_image_color(predict_mask_)
        # show_img_multiplex(pre_mask_show, true_mask_show)
        # plot_fusion_image(img[0, :, :, 0], true_mask_show, 1, 0.5)
        # plot_fusion_image(img[0, :, :, 0], pre_mask_show, 1, 0.5)
        print("specificity is {}\n"
              "sensitivity_ is {}\n"
              "dice_coefficient_ is {} \n"
              "hausdorff_ is {}\n"
              .format(specificity, sensitivity_, dice_coefficient_, hausdorff_))
        statistical = 5
        if counter == statistical:
            ave_dice = dice / statistical
            ave_sens = sens / statistical
            ave_spec = spec / statistical
            show_img_multiplex(predict_mask_, true_mask_)
            break
    print("ave_dice", ave_dice)
    print("ave_sens", ave_sens)
    print("ave_spec", ave_spec)

    return 0


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

    evaluate()
