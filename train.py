import os
import argparse
import time
import numpy as np
import cv2


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--ground_truth_dir", required=True)
    parser.add_argument("--window_size", required=True, type=int, choices=[x for x in range(1, 36) if x % 2 != 0])

    return parser.parse_args()


def get_input_data(images_dir, ground_truth_dir, window_size):
    images = []
    labels = []

    for file in os.listdir(images_dir):
        original_img = cv2.imread(images_dir + '/' + file)
        if original_img is None:
            raise FileNotFoundError("Não foi possivel abrir a imagem " + images_dir + '/' + file)

        ground_truth = cv2.imread(ground_truth_dir + '/' + file, cv2.IMREAD_GRAYSCALE)
        if ground_truth is None:
            raise FileNotFoundError("Não foi possivel abrir o ground truth  " + ground_truth_dir + '/' + file)

        # Binarize the ground truth
        ground_truth = (ground_truth > 10).astype(np.int32)

        # Select the pixels from the 4 diagonals starting at the corners of the image
        half_window = window_size // 2
        smaller_dimension = min(original_img.shape[0], original_img.shape[1])
        main_diagonal = list(range(half_window, smaller_dimension - half_window, window_size))

        for i in main_diagonal:
            # Repeat for the 4 diagonals
            for coordinate in [[ i - half_window,  i + half_window+1,  i - half_window,  i + half_window+1],
                               [-i - half_window, -i + half_window+1,  i - half_window,  i + half_window+1],
                               [ i - half_window,  i + half_window+1, -i - half_window, -i + half_window+1],
                               [-i - half_window, -i + half_window+1, -i - half_window, -i + half_window+1]]:
                images.append(original_img[coordinate[0] : coordinate[1], coordinate[2] : coordinate[3]])

                # The label is set to 1 if more than half of the pixels in the window is skin color
                labels.append(sum(ground_truth[coordinate[0] : coordinate[1], coordinate[2] : coordinate[3]].ravel()) > (window_size ** 2) / 2)

                # Paint window on the image
                # if labels[-1]:
                #     original_img[coordinate[0] : coordinate[1], coordinate[2] : coordinate[3], 1] = 255
                # else:
                #     original_img[coordinate[0] : coordinate[1], coordinate[2] : coordinate[3], 2] = 255

    return images, labels


def main():
    args = get_args()

    start = time.time()

    images, labels = get_input_data(args.images_dir, args.ground_truth_dir, args.window_size)

    print("Dados de entrada: {} imagens de {}x{} pixels".format(len(images), args.window_size, args.window_size))

    end = time.time()

    print("Time: {}".format(end - start))


if __name__ == "__main__":
    main()