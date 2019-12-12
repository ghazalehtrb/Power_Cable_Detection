import os
import matplotlib.pyplot as plt
import matplotlib.image as mp
from skimage.transform import resize
import cv2
import numpy as np
from scipy.misc import imsave
import PIL
from PIL import ImageDraw
def get_mask(ground, size):
    mask224 = PIL.Image.fromarray(np.zeros((224, 224), dtype=np.uint8))
    d2 = ImageDraw.Draw(mask224)
    for line in open(ground):
        if len(line.split()) > 2:
            p1x, p1y, p2x, p2y = list(map(float, line.split()))
            p1x_ = p1x * 224 / size[1]
            p2x_ = p2x * 224 / size[1]
            p1y_ = p1y * 224 / size[0]
            p2y_ = p2y * 224 / size[0]

            if p1x + p2x + p2y + p1y > 0:
                d2.line([(p1x_, p1y_), (p2x_, p2y_)], 255, 2)
    return mask224


base = './DATASET/USF/'
base_size = 224
names = []
for sd1 in os.listdir(base):
    for sd2 in os.listdir(base + sd1 + '/'):
        for image_name in os.listdir(base + sd1 + '/' + sd2):
            name = sd1 + '/' + sd2 + '/' + image_name
            names.append(name)

set_names = list(set([name.replace('.bmp','').replace('.ground','') for name in names]))

for name in set_names:
    print(name)
    image_name = name + '.bmp'
    image = mp.imread(base + image_name)
    init_size = image.shape
    image = resize(image, (base_size, base_size))

    if 'No_Wires' not in name:
        mask_name = name + '.ground'
        mask = get_mask(base+mask_name, init_size)
    else:
        mask = np.zeros((base_size, base_size))
    print(np.shape(np.array(mask)))
    # plt.imshow(image)
    # plt.show()
    # plt.imshow(mask)
    # plt.show()
    # plt.hist(np.array(mask).ravel())
    # plt.show()
    mp.imsave('./all_data1/' + image_name.replace('/','-')[:-4] + '.JPG', image, format='JPG')
    imsave('./all_data1/' + image_name.replace('/','-')[:-4] + '.png', np.array(mask), 'png')