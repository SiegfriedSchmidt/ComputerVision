import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from typing import List, Tuple


def show_images(axes, images_with_titles: List[Tuple[np.ndarray, str]]):
    for i, (img, title) in enumerate(images_with_titles):
        plt.subplot(*axes, i + 1)
        # plt.imshow(img, cmap, vmin=0, vmax=255)
        plt.imshow(img, 'gray' if len(img.shape) == 2 else None)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    plt.subplots_adjust(wspace=0, hspace=0.1)


def otsu_threshold(image_path: str):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    threshold_value, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    plt.suptitle("Otsu threshold", fontsize=25)
    show_images((1, 3), [(img, 'Original'), (thresh, 'Threshold')])

    plt.subplot(1, 3, 3)
    plt.hist(img.ravel(), 255)
    plt.axvline(threshold_value, color='k', linestyle='dashed', linewidth=3)
    plt.title('Histogram')
    plt.show()


def to_rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def k_means(image_path: str):
    img = cv.imread(image_path)
    z = np.float32(img.reshape((-1, 3)))  # image size mx3, m - number of pixels

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    images_with_titles = [(img, "Original")]
    for K in (2, 4, 8):
        ret, label, center = cv.kmeans(z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        result = np.uint8(center)[label.flatten()].reshape(img.shape)
        images_with_titles.append((result, f"K={K}"))

    for image, title in images_with_titles:
        image[:] = to_rgb(image)

    plt.suptitle("K means", fontsize=25)
    show_images((2, 2), images_with_titles)
    plt.show()


def numpy_conv(array, filt, padding=(1, 1)):
    array = np.pad(array, [
        (padding[0], padding[0]),
        (padding[1], padding[1]),
    ])

    output_array = np.zeros(((array.shape[0] - filt.shape[0] + 2 * padding[0]) + 1,
                             (array.shape[1] - filt.shape[1] + 2 * padding[1]) + 1))

    for x in range(array.shape[0] - filt.shape[0] + 1):
        for y in range(array.shape[1] - filt.shape[1] + 1):
            window = array[x:x + filt.shape[0], y:y + filt.shape[1]]
            output_values = np.sum(filt * window, axis=(0, 1))
            output_array[x, y] = output_values

    return output_array


def numpy_sobel(img):
    Gy = numpy_conv(img, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
    Gx = numpy_conv(img, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    result = np.sqrt(Gy * Gy + Gx * Gx)
    # result[result > 255] = 255
    result = result / np.max(result) * 255
    return np.uint8(result)


def sobel(image_path: str):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    edges2 = ski.filters.sobel(img)
    edges = numpy_sobel(img)

    images_with_titles = [
        (img, 'original'),
        (edges, 'numpy sobel'),
        (edges2, 'skimage sobel')
    ]

    plt.suptitle("Sobel", fontsize=25)
    show_images((1, 3), images_with_titles)
    plt.show()


def canny_operator(image_path: str):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    edges = cv.Canny(img, 100, 200)

    images_with_titles = [
        (img, 'original'),
        (edges, 'canny'),
    ]

    plt.suptitle("Canny", fontsize=25)
    show_images((1, 2), images_with_titles)
    plt.show()


def find_contours(image_path):
    img = cv.imread(image_path)
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ret, prepared = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # ret, prepared = cv.threshold(imgray, 100, 150, cv.THRESH_BINARY)
    # prepared = cv.Canny(imgray, 100, 200)
    # prepared = numpy_sobel(imgray)
    # prepared = np.array(ski.filters.sobel(imgray))
    # prepared = np.uint8(prepared / np.max(prepared) * 255)
    ret, prepared = cv.threshold(imgray, 229, 255, 0)
    prepared = 255 - prepared

    contours, hierarchy = cv.findContours(prepared, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    img_with_contours = img.copy()

    cv.drawContours(img_with_contours, contours, 0, (255, 0, 0), 6)
    img_with_contours = cv.putText(img_with_contours, '1', contours[0][-200][0], cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 6, cv.LINE_AA)
    cv.drawContours(img_with_contours, contours, 1, (0, 255, 0), 6)
    img_with_contours = cv.putText(img_with_contours, '2', contours[1][-200][0], cv.FONT_HERSHEY_SIMPLEX, 3,
                                   (0, 255, 0), 6, cv.LINE_AA)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_with_contours = cv.cvtColor(img_with_contours, cv.COLOR_BGR2RGB)

    images_with_titles = [
        (img, 'original'),
        (prepared, 'prepared'),
        (img_with_contours, 'with contours'),
    ]

    plt.suptitle("Contours", fontsize=25)
    show_images((1, 3), images_with_titles)
    plt.show()


def main():
    otsu_threshold('images/img.png')
    k_means('images/img.png')
    sobel('images/img.png')
    canny_operator('images/img.png')
    find_contours('images/img_4.png')


if __name__ == '__main__':
    main()
