import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

eps = np.finfo(float).eps

# Compute the histogram of a given image and of its prediction errors. If the pixel being processed is at
# coordinate (0,0), consider
# -	predicting based on just the pixel at (-1,0);
# -	predicting based on just the pixel at (0,1);
# -	predicting based on the average of the pixels at (-1,0), (-1,1), and (0,1).
# Compute the entropy for each one of the predictors in the previous exercise. Which predictor will compress better?

def main():
    # Read image
    # img_file = 'images/ace_of_spades.jpg'       # Motorhead
    # img_file = 'images/4.2.03.tiff'             # Baboon
    # img_file = 'images/elaine.512.tiff'         # B&W
    img_file = 'images/4.2.04.tiff'             # Lena
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

    # Prepare plot
    f, plt_arr = plt.subplots(2, 4)
    f.canvas.set_window_title('Spatial Prediction')

    # Statistics on original image
    hist, bins = np.histogram(img, bins=256, range=(0, 256), density=True)
    mean = np.round(np.mean(img), 2)
    std_dev = np.round(np.std(img), 2)
    entropy = np.round(-np.sum(hist[np.where(hist > 0)] * np.log2(hist[np.where(hist > 0)])), 2)

    # Display image
    plt_arr[0, 0].set_title('Original image')
    plt_arr[0, 0].imshow(img, cmap=cm.Greys_r, vmin=0, vmax=255, interpolation="nearest")
    title = 'm=' + str(mean) + u', \u03C3=' + str(std_dev) + ', H=' + str(entropy)
    plt_arr[1, 0].set_title(title)
    plt_arr[1, 0].plot(bins[:-1], hist)
    plt_arr[1, 0].set_xlim([0, 255])
    print 'm=', mean
    print u'\u03C3=', std_dev
    print 'h=', entropy

    for prediction, i in zip(('left', 'top', 'top-left'), range(1,4)):
        # Calculate predictor and residuals
        # Prefill predictor with center value
        predictor = np.empty_like(img, dtype=np.float)
        predictor.fill(128)
        if prediction == 'left':
            predictor[:, 1:] = img[:, 0:-1]
        elif prediction == 'top':
            predictor[1:, :] = img[0:-1, :]
        elif prediction == 'top-left':
            predictor[:, 1:] = img[:, 0:-1]
            predictor[1:, :] += img[0:-1, :]
            predictor[1:, 1:] += img[0:-1, 0:-1]
            predictor /= 3
        error = img - predictor

        # Statistics on residuals image
        hist, bins = np.histogram(error, bins=512, range=(-256, 256), density=True)
        mean = np.round(np.mean(error), 2)
        mse = np.round(np.sqrt(np.mean(np.abs(error)**2)), 2)
        std_dev = np.round(np.std(error), 2)
        entropy = np.round(-np.sum(hist[np.where(hist > 0)] * np.log2(hist[np.where(hist > 0)])), 2)


        if prediction == 'left':
            title = 'Left prediction residual'
        elif prediction == 'top':
            title = 'Up prediction residual'
        elif prediction == 'top-left':
            title = 'Left, up, up-left prediction residual'

        error_img = ((error + 255)/2).astype(np.uint8)
        plt_arr[0, i].set_title(title)
        plt_arr[0, i].imshow(error_img, cmap=cm.Greys_r, vmin=0, vmax=255, interpolation="nearest")
        title = 'm=' + str(mean) + ', mse=' + str(mse) + u', \u03C3=' + str(std_dev) + ', H=' + str(entropy)
        plt_arr[1, i].set_title(title)
        plt_arr[1, i].plot(bins[:-1], hist)
        plt_arr[1, i].set_xlim([-255, 255])

        print 'm=', mean
        print 'mse=', mse
        print u'\u03C3=', std_dev
        print 'H=', entropy

    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.show()


if __name__ == '__main__':
    main()
