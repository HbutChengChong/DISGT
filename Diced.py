import numpy as np
import scipy.io as scio
import h5py
from __init__ import *

size = 32  # 想要裁剪的尺寸
stride = 20  # 滑动步长

# dataname = 'Indian_pines'
# dataname = 'Salinas'
dataname = 'Hanchuan'
# dataname = 'PaviaU'
# dataname = 'Dioni'


if dataname == 'indian':
    cropimg_savepath = PATH_block + 'Indian_pines_32/img_slide/_{}-{}_.npy'
    croplabel_savepath = PATH_block + 'Indian_pines_32/label_slide/_{}-{}_.npy'
    cropimg1_savepath = PATH_block + 'Indian_pines_32/img_mean/_{}+{}_.npy'
    croplabel1_savepath = PATH_block + 'Indian_pines_32/label_mean/_{}+{}_.npy'
elif dataname == 'Salinas':
    cropimg_savepath = PATH_block + 'Salinas_32/img_slide/_{}-{}_.npy'
    croplabel_savepath = PATH_block + 'Salinas_32/label_slide/_{}-{}_.npy'
    cropimg1_savepath = PATH_block + 'Salinas_32/img_mean/_{}+{}_.npy'
    croplabel1_savepath = PATH_block + 'Salinas_32/label_mean/_{}+{}_.npy'
elif dataname == 'Hanchuan':
    cropimg_savepath = PATH_block + 'Hanchuan_32_nonormed/img_slide/_{}-{}_.npy'
    croplabel_savepath = PATH_block + 'Hanchuan_32_nonormed/label_slide/_{}-{}_.npy'
    # cropimg1_savepath = PATH_block + 'Hanchuan_32/img_mean/_{}+{}_.npy'
    # croplabel1_savepath = PATH_block + 'Hanchuan_32/label_mean/_{}+{}_.npy'
elif dataname == 'PaviaU':
    cropimg_savepath = PATH_block + 'PaviaU_32/img_slide/_{}-{}_.npy'
    croplabel_savepath = PATH_block + 'PaviaU_32/label_slide/_{}-{}_.npy'
    cropimg1_savepath = PATH_block + 'PaviaU_32/img_mean/_{}+{}_.npy'
    croplabel1_savepath = PATH_block + 'PaviaU_32/label_mean/_{}+{}_.npy'
elif dataname == 'Dioni':
    cropimg_savepath = PATH_block + 'Dioni_32/img_slide/_{}-{}_.npy'
    croplabel_savepath = PATH_block + 'Dioni_32/label_slide/_{}-{}_.npy'
    # cropimg1_savepath = PATH_block + 'Dioni_32/img_mean/_{}+{}_.npy'
    # croplabel1_savepath = PATH_block + 'Dioni_32/label_mean/_{}+{}_.npy'

def normalize(image):
    mean = np.mean(image)
    var = np.mean(np.square(image - mean))
    image = (image - mean) / np.sqrt(var)
    return image


def crop(img, label):
    (width, height, channel) = img.shape  # [H,W,C]
    num_width = int((width - size) / stride)
    num_height = int((height - size) / stride)
    for i in range(num_width):
        for j in range(num_height):
            crop_img = img[i * stride:i * stride + size, j * stride:j * stride + size, :]
            crop_label = label[i * stride:i * stride + size, j * stride:j * stride + size]
            # print(crop_img.shape)
            # print(crop_label.shape)
            np.save(cropimg_savepath.format(i, j), crop_img)
            np.save(croplabel_savepath.format(i, j), crop_label)


def crop1(img, label):
    (width, height, channel) = img.shape  # [H,W,C]
    num_width = int(width / size)
    num_height = int(height / size)
    for i in range(0, num_width):
        for j in range(0, num_height):
            crop_img = img[i * size:(i + 1) * size, j * size:(j + 1) * size, :]
            crop_label = label[i * size:(i + 1) * size, j * size:(j + 1) * size]
            np.save(cropimg1_savepath.format(i + 1, j + 1), crop_img)
            np.save(croplabel1_savepath.format(i + 1, j + 1), crop_label)

    for i in range(0, num_width):
        w_end_img = img[i * size:(i + 1) * size, height - size:height, :]
        w_end_label = label[i * size:(i + 1) * size, height - size:height]
        np.save(cropimg1_savepath.format(i + 1, num_height + 1), w_end_img)
        np.save(croplabel1_savepath.format(i + 1, num_height + 1), w_end_label)

    for j in range(0, num_height):
        h_end_img = img[width - size:width, j * size:(j + 1) * size, :]
        h_end_label = label[width - size:width, j * size:(j + 1) * size]
        np.save(cropimg1_savepath.format(num_width + 1, j + 1), h_end_img)
        np.save(croplabel1_savepath.format(num_width + 1, j + 1), h_end_label)

    end_img = img[width - size:width, height - size:height, :]
    end_label = label[width - size:width, height - size:height]
    np.save(cropimg1_savepath.format(num_width + 1, num_height + 1), end_img)
    np.save(croplabel1_savepath.format(num_width + 1, num_height + 1), end_label)


if __name__ == '__main__':
    # 读取原始数据集
    # data = scio.loadmat(PATH_ori+'Indian_pines.mat')['indian_pines_corrected']
    # matlab v7.3数据读取
    # data = h5py.File('D:/tca/dataset/Xiongan/Xiongan.mat')['Xiongan'][:]
    data = scio.loadmat(PATH_ori+'Hanchuan.mat')['Hanchuan']
    # data = tiff.imread(r'C:\Users\HP\PycharmProjects\Transformer\data\ori_data\Dioni.tif')

    # data = np.transpose(data, (2, 1, 0))

    # norm = normalize(data)
    # scio.savemat(PATH_ori+'Salinas_normed.mat', {'salinas_normed': norm})
    # print(norm)
    # print('norm mat saved')

    # img = norm
    img = data
    # img = scio.loadmat('D:/tca/dataset/HC/Hanchuan_normed.mat')['hanchuan_normed']
    label = sio.loadmat(PATH_ori+'Hanchuan_gt.mat')['Hanchuan_gt']
    # label = tiff.imread(r'C:\Users\HP\PycharmProjects\Transformer\data\ori_data\Dioni_GT.tif')
    crop(img, label)

    # print('norm mat cropped')
    print('unnorm mat cropped')
    # crop1(img, label)
    # print('sequential mat cropped')
