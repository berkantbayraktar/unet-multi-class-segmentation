from model import *
from data import *

import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


'''

    Obstacles and environment = 0 (value zero)
    Water = 1 (value one)
    Sky = 2 (value two)
    Ignore region / unknown category = 4 (value four)

'''

number_of_class = 4
epochs = 10
batch_size = 16
network_size = (256, 256)


def train():
    data_gen_args = dict(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05,
                         zoom_range=0.05, horizontal_flip=True, fill_mode='nearest')

    train_gen = train_generator(batch_size=batch_size, train_path='data/MaSTr/train', image_folder='image', mask_folder='label',
                                aug_dict=data_gen_args, image_color_mode="rgb", mask_color_mode="grayscale",
                                num_class=number_of_class, save_to_dir=None)

    model = unet(input_size=network_size + (3,), pretrained_weights=None)
    model_checkpoint = ModelCheckpoint('unet_mastr.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(train_gen, steps_per_epoch=50, epochs=epochs, callbacks=[model_checkpoint])


def test():
    test_gen = test_generator(test_path="data/MaSTr/test", num_image=30, target_size=network_size, as_gray=False)

    model = unet(input_size=network_size + (3,), pretrained_weights=None)
    model.load_weights("unet_mastr.hdf5")

    results = model.predict_generator(test_gen, 30, verbose=1)

    test_gen = test_generator(test_path="data/MaSTr/test", num_image=30, target_size=network_size, as_gray=False)
    save_result("data/MaSTr/test", results, test_gen, num_class=4)


def main(argv):
    command = argv[0]
    if command == 'train':
        train()
    elif command == 'test':
        test()
    else:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
