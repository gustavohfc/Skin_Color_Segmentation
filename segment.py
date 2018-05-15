from matplotlib import pyplot as plt
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

    return input_images


def calculate_histogram(ground_truths):
    # Resize all images to 512x768
    for i in range(len(ground_truths)):
        ground_truths[i] = np.resize(ground_truths[i], (768, 512, 3))

    # Convert the list to a numpy array of shape (N_IMAGES, 768, 512, 3)
    ground_truths = np.stack(ground_truths, axis=0)

    histograms = []
    for channel in range(2):
        hist, bin_edges = np.histogram(ground_truths[:, :, :, channel].ravel(), 255, (0, 255))
        histograms.append(hist)

    # Remove outliers from the black background
    histograms[0][0] = 0
    histograms[1][0] = 0
    histograms[1][254] = 0

    # Nomarlize
    histograms[0] = histograms[0] / np.linalg.norm(histograms[0])
    histograms[1] = histograms[1] / np.linalg.norm(histograms[1])

    histograms = np.asarray(histograms)

    return histograms

def show_histograms(histograms):
    plt.subplot(2, 1, 1)
    plt.bar(np.arange(255), histograms[0])
    plt.title('Hue')

    plt.subplot(2, 1, 2)
    plot = plt.bar(np.arange(255), histograms[1])
    plt.title('Saturation')

    plt.show()

def segment_images(input_images, histograms):
    for image in input_images:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                hue = image[i, j, 0]
                saturation = image[i, j, 1]

                if histograms[0, hue-1] < 0.05 or histograms[1, saturation-1] < 0.02:
                    image[i, j, :] = 0

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        cv2.imshow("Teste", image)
        cv2.waitKey(0)

def main():
    # Parse the command line arguments
    args = get_args()

    input_images = read_images(args.input_images_dir)
    ground_truths = read_images(args.ground_truths_dir)

    histograms = calculate_histogram(ground_truths)

    show_histograms(histograms)

    segment_images(input_images, histograms)

    # print(histogram)

    # cv2.waitKey(0)


if __name__ == "__main__":
    main()