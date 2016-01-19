import cv2
import numpy as np
import matplotlib.pyplot as plt



def display(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # read image
    img_file = 'images/ace_of_spades.jpg'
    # img_file = 'images/4.2.04.tiff'
    # img_file = 'images/4.2.03.tiff'
    # img_file = 'images/elaine.512.tiff'
    img = cv2.imread(img_file)
    display(img)

    #imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    #imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

if __name__ == '__main__':
    main()

