import argparse
import tensorflow as tf
import numpy as np
import time
import cv2

import train

hidden_units=train.hidden_units
n_classes=train.n_classes


def get_args():
    parser = argparse.ArgumentParser(description='.')

    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input_image_path", required=True)
    parser.add_argument("--window_size", required=True, type=int, choices=[x for x in range(1, 36) if x % 2 != 0])

    return parser.parse_args()


def load_classifier(model_dir, window_size):
    feature_columns = [tf.feature_column.numeric_column("Images", shape=[window_size, window_size, 3], dtype=tf.int32)]

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        n_classes=n_classes,
        model_dir=model_dir
    )

    return classifier


def segment(classifier, input_image, window_size):
    images = []

    # Extract the windows from the image
    for y in range(0, input_image.shape[0], window_size):
        for x in range(0, input_image.shape[1], window_size):
            window = input_image[y : y+window_size, x : x+window_size]

            if window.shape != (window_size, window_size, 3):
                window = np.zeros((window_size, window_size, 3), dtype=np.int32)

            images.append(window)

    images = np.asarray(images, np.int32)
    image_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"Images": images},
        shuffle=False,
    )

    predictions = list(classifier.predict(image_input_fn))

    segmented_image = np.zeros(input_image.shape)
    for i, y in enumerate(range(0, input_image.shape[0], window_size)):
        for j, x in enumerate(range(0, input_image.shape[1], window_size)):
            if predictions[i + j]["probabilities"][0] > predictions[i + j]["probabilities"][1]:
                segmented_image[y : y + window_size, x : x + window_size] = 0
            else:
                segmented_image[y : y + window_size, x : x + window_size] = 1


    return segmented_image




def main():
    start = time.time()

    args = get_args()


    # classifier = load_classifier(args.model_dir, args.window_size)

    images, labels = train.get_input_data("Images/Input_Originals/", "Images/Input_Ground_Truth/", 11)

    print("Dados de entrada: {} imagens de {}x{} pixels".format(len(images), 11, 11))

    classifier = train.train_new_model(images, labels, 11, "req2_model")



    input_image = cv2.imread(args.input_image_path)
    if input_image is None:
        raise FileNotFoundError("Não foi possivel abrir a imagem " + args.input_image_path)
    segmented_image = segment(classifier, input_image, args.window_size)


    end = time.time()

    print("Time: {}".format(end - start))


    print("sum: {}".format(sum(sum(segmented_image))))


    cv2.imshow("Original", input_image)
    cv2.imshow("Segmented", segmented_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()