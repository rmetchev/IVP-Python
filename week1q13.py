import cv2
import numpy as np


def reduce_levels(img, n):
    # check for valid n
    if n < 2 or n > 256 or n != 2**int(np.log2(int(n))):
        raise ValueError("n must be power of 2 integer between 2 and 256", n)
    m = 256//n
    return m*(img//m)


def spatial_average(img, n):
    kernel = np.ones((n, n))/(n * n)
    return cv2.filter2D(img, -1, kernel)


def rotate_image(img, angle):
    height, width = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    return cv2.warpAffine(img, rotation_matrix, (width, height))


def spatial_resolution_reduction(img, n):
    # create new image with shape of img & all pixels having value of 0
    img_reduced = np.zeros(img.shape, dtype=np.uint8)
    # Populate every n-th pixel in both directions
    img_reduced[::n, ::n] = img[::n, ::n]
    # Create filter whose impulse response is n x n unit step function
    kernel = np.ones((n, n))
    # Run filter - this will interpolate all 8 neighbor pixels with the value of the center pixel
    return cv2.filter2D(img_reduced, -1, kernel)


def display(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # read image
    img_file = 'motorhead.jpg'
    img = cv2.imread(img_file)
    display(img)

    # reduce levels
    img2 = reduce_levels(img, 8)
    display(img2)

    # spatial average
    img2 = spatial_average(img, 32)
    display(img2)

    # rotate
    img2 = rotate_image(img, 45)
    display(img2)

    # spatial resolution reduction
    img2 = spatial_resolution_reduction(img, 8)
    display(img2)

if __name__ == '__main__':
    main()

