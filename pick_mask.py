

from glob import glob
import cv2
import shutil
import os 

print(os.getcwd())

cwd = os.getcwd()


mask_list = []

img_path = cwd + '/data/patches/image_patches/'
mask_path = cwd + '/data/patches/mask_patches/'

nonzero_img_path = cwd + '/data/patches/nonzero_image_patches/'
nonzero_mask_path = cwd + '/data/patches/nonzero_mask_patches/'


mask_images = glob(mask_path + '*.jpg')

print(mask_images[0])

for img in range(len(mask_images)):
    
    mask = cv2.imread(mask_images[img], 0)
    print(mask.sum())
    
    if mask.sum() != 0 :
        origin_mask = mask_images[img].split('/')[-1]
        
        nonzero_mask = mask_path + origin_mask
        nonzero_img = img_path + origin_mask
        shutil.copy(nonzero_mask, nonzero_mask_path)
        shutil.copy(nonzero_img, nonzero_img_path)
        