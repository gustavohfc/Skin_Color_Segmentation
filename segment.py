from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
import cv2

hue_threshold = 0.17
saturation_threshold = 0.01

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_ground_truths_dir", required=True)
    parser.add_argument("--input_images_dir", required=True)
    parser.add_argument("--expected_ground_truths_dir", required=True)

    return parser.parse_args()


def read_images(dir):
    input_images = []

    for file in sorted(os.listdir(dir)):
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
        hist, _ = np.histogram(ground_truths[:, :, :, channel].ravel(), 255, (0, 255))
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
    plot = plt.bar(np.arange(255), histograms[0])
    plt.title('Hue')
    for i, bar in enumerate(plot):
        bar.set_facecolor('r' if histograms[0, i] < hue_threshold else 'g')

    plt.subplot(2, 1, 2)
    plot = plt.bar(np.arange(255), histograms[1])
    plt.title('Saturation')
    for i, bar in enumerate(plot):
        bar.set_facecolor('r' if histograms[1, i] < saturation_threshold else 'g')

    plt.show()


def segment_images(input_images, histograms):
    segmented_images = []

    for image in input_images:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                hue = image[i, j, 0]
                s = image[i, j, 1]

                if histograms[0, hue-1] < hue_threshold or histograms[1, s-1] < saturation_threshold:
                    image[i, j, :] = 0
                else:
                    image[i, j, 0] = 0
                    image[i, j, 1] = 0
                    image[i, j, 2] = 255


        segmented_images.append(image)


    return segmented_images


def evaluate_results(segmented_images, expected_ground_truths):
    acuracia = []
    jaccard_index = []

    for segmented, expected in zip(segmented_images, expected_ground_truths):
        segmented_binarized = segmented[:, :, 2] > 127
        expected_binarized = expected[:, :, 2] > 127

        acuracia.append(sum(np.equal(segmented_binarized, expected_binarized).ravel()) / np.size(segmented_binarized))

        intersection = sum(np.logical_and(segmented_binarized, expected_binarized).ravel())
        union = sum(np.logical_or(segmented_binarized, expected_binarized).ravel())
        jaccard_index.append(intersection / union)

    print("Acurácia (media): {}".format(np.asanyarray(acuracia).mean()))
    print("Acurácia (desvio): {}".format(np.asanyarray(acuracia).std()))

    print("Jaccard index (meida): {}".format(np.asanyarray(jaccard_index).mean()))
    print("Jaccard index (desvio): {}".format(np.asanyarray(jaccard_index).std()))


def main():
    # Parse the command line arguments
    args = get_args()

    input_images = read_images(args.input_images_dir)
    expected_ground_truths = read_images(args.expected_ground_truths_dir)
    train_ground_truths = read_images(args.train_ground_truths_dir)

    if len(input_images) != len(expected_ground_truths):
        print("O número de imagens em 'input_images_dir' e 'expected_ground_truths_dir' deve ser igual.")
        exit(1)

    histograms = calculate_histogram(train_ground_truths)

    show_histograms(histograms)

    segmented_images = segment_images(input_images, histograms)

    evaluate_results(segmented_images, expected_ground_truths)


if __name__ == "__main__":
    main()