import os
import time

import cv2
from tqdm import tqdm

from norm_tools import MacenkoNormalizer, ReinhardNormalizer, VahadaneNormalizer
from utils import read_image


if __name__ == '__main__':
    target_dir = r'F:\Data\endometrium_pathology\all'
    norm_dir = r'F:\Data\endometrium_pathology\all_norm'
    norm_method = 'vahadane'
    target_img = r'target.jpg'

    # copy the file structure of target_dir to norm_dir
    for root, dirs, files in os.walk(target_dir):
        for dir in dirs:
            os.makedirs(os.path.join(norm_dir, dir), exist_ok=True)

    # normalization
    if norm_method == 'macenko':
        normalizer = MacenkoNormalizer()
    elif norm_method == 'reinhard':
        normalizer = ReinhardNormalizer()
    elif norm_method == 'vahadane':
        normalizer = VahadaneNormalizer()
    else:
        print(f'Unsupported normalization method {norm_method}')
        exit(-1)
    target_img = read_image(target_img)
    normalizer.fit(target_img)
    for root, dirs, files in os.walk(target_dir):
        for dir in tqdm(dirs):
            print(f'Processing ')
            idx = 0
            length = len(os.listdir(os.path.join(root, dir)))
            for file in os.listdir(os.path.join(root, dir)):
                if not dir.startswith('TCGA'):
                    continue
                file = os.path.join(dir, file)
                if file.endswith('.jpg') or file.endswith('.png'):
                    tic = time.time()
                    img_path = os.path.join(root, file)
                    norm_img = read_image(img_path)
                    norm_img = normalizer.transform(norm_img)
                    norm_img_path = os.path.join(norm_dir, os.path.relpath(img_path, target_dir))
                    norm_img_path = norm_img_path.replace('.png', '.jpg')
                    cv2.imwrite(norm_img_path, norm_img)
                    info = 'Processing {} / {} Time elapse {:.2f} s'.format(idx, length, time.time()-tic)
                    print(info)
                    idx += 1
    print('Done')


