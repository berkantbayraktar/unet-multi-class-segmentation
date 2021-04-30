from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2
'''

    Obstacles and environment = 0 (value zero)
    Water = 1 (value one)
    Sky = 2 (value two)
    Ignore region / unknown category = 4 (value four)

'''

Obstacles = [225, 109, 50]
Water = [10, 100, 152]
Sky = [223, 218, 214]
Ignore = [0, 0, 0]
COLOR_DICT = np.array([Obstacles, Water, Sky, Ignore])


def adjust_data(img, mask, num_class):

    img = img / 255
    mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
    new_mask = np.zeros(mask.shape + (num_class,))
    new_mask[mask == 0, 0] = 1
    new_mask[mask == 1, 1] = 1
    new_mask[mask == 2, 2] = 1
    new_mask[mask == 4, 3] = 1
    mask = new_mask

    return img, mask


def train_generator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                    mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix ="mask",
                    num_class=2, save_to_dir=None, target_size=(256,256), seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjust_data(img, mask, num_class)
        yield img, mask


def test_generator(test_path, num_image=30, target_size=(256, 256), as_gray=True):
    for i in range(0, num_image, 1):
        img = io.imread(os.path.join(test_path, format(i,"04") + ".jpg"),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,(1,)+img.shape)
        yield img


def geneTrainNpy(image_path, mask_path, flag_multi_class=False,num_class=2, image_prefix="image", mask_prefix="mask", image_as_gray=True, mask_as_gray=True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray = image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjust_data(img, mask, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


def label_visualize(num_class, color_dict, img):
    img = np.argmax(img, axis=2)
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def save_result(save_path, npyfile, test_gen, num_class=2):
    for i, item in enumerate(npyfile):
        orj = next(test_gen)[0]
        img = label_visualize(num_class, COLOR_DICT, item)
        blended = cv2.addWeighted(np.float32(orj), 0.5, np.float32(img), 0.5, 0)

        io.imsave(os.path.join(save_path, "%d_predict.png"%i), blended)