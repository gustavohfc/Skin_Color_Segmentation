# from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
import cv2

def get_args():
    parser = argparse.ArgumentParser(description='.')

    parser.add_argument("--input_images_dir", required=True)
    parser.add_argument("--ground_truths_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    return parser.parse_args()


def read_images(dir):
    input_images = []

    for file in os.listdir(dir):
        # Ignore hidden files
        if file[0] == '.':
            continue

        image = cv2.imread(dir + '/' + file)
        if image is None:
            raise FileNotFoundError("Não foi possivel abrir a imagem " + dir + '/' + file)

        # Convert the image from BGR to HSV for better skin detection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        input_images.append(image)

    # np.stack(input_images, axis=0)

    return input_images


def calculate_histogram(ground_truths):
    # Resize all images to 512x768
    for i in range(len(ground_truths)):
        ground_truths[i] = np.resize(ground_truths[i], (768, 512, 3))

    # Convert the list to a numpy array of shape (N_IMAGES, 768, 512, 3)
    ground_truths = np.stack(ground_truths, axis=0)

    histogram = []
    for channel in range(3):
        hist, bin_edges = np.histogram(ground_truths[:, :, :, channel].ravel(), 255, (0, 255))
        histogram.append(hist)

    histogram = np.asarray(histogram)

    print(len(ground_truths[:, :, :, :].ravel()))
    print(sum(sum(histogram)))

    return histogram


def main():
    # Parse the command line arguments
    args = get_args()

    # input_images = read_images(args.input_images_dir)
    ground_truths = read_images(args.ground_truths_dir)

    histogram = calculate_histogram(ground_truths)

    # print(histogram)

    # cv2.waitKey(0)
    

if __name__ == "__main__":
    main()