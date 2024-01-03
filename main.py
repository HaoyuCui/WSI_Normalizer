import os
import time

import cv2
import argparse

from tqdm import tqdm

from src.norm_tools import MacenkoNormalizer, ReinhardNormalizer, VahadaneNormalizer


def read_image(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--method', type=str, default='vahadane', choices=['macenko', 'reinhard', 'vahadane'])
    parser.add_argument('--target_img', type=str, default='target.jpg')

    args = parser.parse_args()
    target_img = args.target_img

    # copy the file structure of target_dir to output_dir
    for root, dirs, files in os.walk(args.target_dir):
        for dir in dirs:
            os.makedirs(os.path.join(args.output_dir, dir), exist_ok=True)

    # normalization
    method = args.method
    if args.method == 'macenko':
        normalizer = MacenkoNormalizer()
    elif args.method == 'reinhard':
        normalizer = ReinhardNormalizer()
    elif args.method == 'vahadane':
        normalizer = VahadaneNormalizer()
    else:
        print(f'Unsupported normalization method {args.method}')
        exit(-1)

    if not os.path.exists(target_img):
        print(f'Target image {target_img} not found')
        exit(-1)

    target_img = read_image(target_img)
    normalizer.fit(target_img)
    # iter through the target_dir
    for root, dirs, files in os.walk(args.target_dir):
        for dir in tqdm(dirs):
            print(f'Processing folder: {dir}')
            idx = 0
            abs_path = os.path.join(root, dir)
            length = len(os.listdir(abs_path))
            for file in os.listdir(abs_path):
                file = os.path.join(dir, file)
                if file.endswith('.jpg') or file.endswith('.png'):
                    tic = time.time()
                    img_path = os.path.join(root, file)
                    norm_img = read_image(img_path)
                    norm_img = normalizer.transform(norm_img)
                    norm_img_path = os.path.join(args.output_dir, os.path.relpath(img_path, args.target_dir))
                    # replace the extension, jpg for efficient compression, png for lossless compression
                    norm_img_path = norm_img_path.replace('.png', '.jpg')
                    cv2.imwrite(norm_img_path, norm_img)
                    info = 'Processing {} / {} Time elapse {:.2f} s'.format(idx, length, time.time()-tic)
                    print(info)
                    idx += 1
    print('Done')


