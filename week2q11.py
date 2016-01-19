import cv2
import numpy as np
from scipy.fftpack import dct
from scipy.fftpack import idct
from numpy.fft import fft2
from numpy.fft import ifft2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Calculate PSNR for image quality comparison
def psnr(i0, i1):
    mse = np.mean(np.square(i0 - i1))
    if mse == 0:  # prevent errors with log(0)
        mse = np.finfo(float).eps
    return np.round(20 * np.log10(255) - 10*np.log10(mse), 2)


# Calculate Entropy for compression efficiency comparison
def entropy(i):
    i = i.astype(np.int)
    hist = np.histogram(i, bins=range(np.min(i), np.max(i) + 2), density=True)[0]
    hist = hist[np.where(hist > 0)]
    return np.round(-np.sum(hist * np.log2(hist)), 2)


# Luminance quantization table
def jpeg_quantization_table_luma():
    return np.array([
        [16,  11,  10,  16,  24,  40,  51,  61],
        [12,  12,  14,  19,  26,  58,  60,  55],
        [14,  13,  16,  24,  40,  57,  69,  56],
        [14,  17,  22,  29,  51,  87,  80,  62],
        [18,  22,  37,  56,  68, 109, 103,  77],
        [24,  36,  55,  64,  81, 104, 113,  92],
        [49,  64,  78,  87, 103, 121, 120, 101],
        [72,  92,  95,  98, 112, 100, 103,  99]
    ]).T


# Chroma quantization table
def jpeg_quantization_table_chroma():
    return np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ]).T


# quantize/de-quantize
def quantize(i, q, inverse=False):
    out = np.empty_like(i)
    for x in range(out.shape[1]):
        for y in range(out.shape[0]):
            out[y, x] = q * i[y, x] if inverse else np.trunc(i[y, x] / q)
    return out


# Divide the image into non-overlapping nxn blocks
def gen_nxn(i, n=8):
    block_shape = list(i.shape)
    block_shape[0] = block_shape[1] = n
    out_shape = np.concatenate((np.ceil(np.array(i.shape[0:2]).astype(np.float) / n).astype(np.int), block_shape))
    out = np.zeros(out_shape)
    i_x_split = np.array(np.array_split(i, range(0, n*out_shape[1], n), axis=1)[1:])  # split along x-axis
    for y in range(out_shape[0]):
        out[:, y] = \
            np.array(np.array_split(i_x_split[y], range(0, n*out_shape[0], n), axis=0)[1:])  # split along y-axis
    return out


# Combine the image from non-overlapping nxn blocks
def combine_nxn(i):
    n = i.shape[2]
    out = np.empty(n * np.array(i.shape[0:2]))
    for x in range(i.shape[1]):
        for y in range(i.shape[0]):
            out[y*n:(y + 1)*n, x*n:(x + 1)*n] = i[y, x, :, :]
    return out


# 2-D DCT on nxn blocks
def dct2_nxn(i):
    out = np.empty_like(i)
    for x in range(out.shape[1]):
        for y in range(out.shape[0]):
            # out[y, x] = dct(dct(i[y, x], axis=1, norm='ortho'), axis=0, norm='ortho')
            out[y, x] = cv2.dct(i[y, x])
    return out


# 2-D IDCT on nxn blocks
def idct2_nxn(i):
    out = np.empty_like(i)
    for x in range(out.shape[1]):
        for y in range(out.shape[0]):
            # out[y, x] = idct(idct(i[y, x], axis=1, norm='ortho'), axis=0, norm='ortho')
            out[y, x] = cv2.idct(i[y, x])
    return out


# 2-D FFT on nxn blocks
def fft2_nxn(i):
    y_max , x_max = i.shape[0], i.shape[1]
    n = i.shape[2]
    out = np.empty(i.shape, dtype=np.complex)
    for x in range(x_max):
        for y in range(y_max):
            out[y, x] = fft2(i[y, x])
    return out


# 2-D IFFT on nxn blocks
def ifft2_nxn(i):
    y_max , x_max = i.shape[0], i.shape[1]
    n = i.shape[2]
    out = np.empty(i.shape, dtype=np.complex)
    for x in range(x_max):
        for y in range(y_max):
            out[y, x] = ifft2(i[y, x])
    return out

# display function
def display(img):
    cv2.imshow('image', img.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# main
def main():
    # img_file = 'images/ace_of_spades.jpg'       # Motorhead
    # img_file = 'images/4.2.03.tiff'             # Baboon
    # img_file = 'images/elaine.512.tiff'         # B&W
    img_file = 'images/4.2.04.tiff'             # Lena

    # Do a basic implementation of JPEG
    # read image
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    # display(img)

    for transform in 'dct', 'fft', 'none':
        # Divide the image into non-overlapping 8x8 blocks.
        # Compute the DCT (discrete cosine transform) of each block.
        title = 'Monochrome, transform type = ' + transform
        print '*', title
        f, plt_arr = plt.subplots(2, 3)
        f.canvas.set_window_title(title)

        title = 'Original, H=' + str(entropy(img))
        u = 0
        plt_arr[u//3, u%3].set_title(title)
        plt_arr[u//3, u%3].imshow(img[200:328, 200:328], cmap=cm.Greys_r, vmin=0, vmax=255, interpolation="nearest")

        if transform == 'dct':
            img_dct_8x8 = dct2_nxn(gen_nxn(img, n=8))
        elif transform == 'fft':
            img_dct_8x8 = fft2_nxn(gen_nxn(img, n=8) / (1+0j))
        elif transform == 'none':
            img_dct_8x8 = gen_nxn(img, n=8)

        # Quantize each block. You can do this using the tables in the video or simply divide each coefficient by N,
        # round
        # the result to the nearest integer, and multiply back by N. Try for different values of N. You can also try
        # preserving the 8 largest coefficients (out of the total of 8x8=64), and simply rounding them to the closest
        # integer.
        # Visualize the results after you invert the quantization and the DCT

        # Repeat the above but instead of using the DCT, use the FFT (Fast Fourier Transform).
        # Repeat the above JPEG-type compression but don't use any transform, simply perform quantization on the
        # original image

        # 1. Luma quantization table
        q_table = jpeg_quantization_table_luma()

        if transform == 'dct':
            img_dct_8x8_quantized = quantize(img_dct_8x8, q_table)
            img_decoded = combine_nxn(idct2_nxn(quantize(img_dct_8x8_quantized, q_table, inverse=True)))
        elif transform == 'fft':
            img_dct_8x8_quantized = quantize(img_dct_8x8.real, q_table) + 1j*quantize(img_dct_8x8.imag, q_table)
            # IFFT should be real
            img_decoded = combine_nxn(np.abs(ifft2_nxn(quantize(img_dct_8x8_quantized.real, q_table, inverse=True) +
                                                       1j*quantize(img_dct_8x8_quantized.imag, q_table, inverse=True))))
        elif transform == 'none':
            img_dct_8x8_quantized = quantize(img_dct_8x8, q_table)
            img_decoded = combine_nxn(quantize(img_dct_8x8_quantized, q_table, inverse=True))

        p = psnr(img, img_decoded)
        if transform == 'fft':
            h = entropy(img_dct_8x8_quantized.real + img_dct_8x8_quantized.imag)
        else:
            h = entropy(img_dct_8x8_quantized)

        title = 'Luma q-table, PSNR=' + str(p) + ', H=' + str(h)
        print '**', title
        u = 1
        plt_arr[u//3, u%3].set_title(title)
        plt_arr[u//3, u%3].imshow(img_decoded[200:328, 200:328], cmap=cm.Greys_r, vmin=0, vmax=255, interpolation="nearest")
        # display(img_decoded)

        # 2. Divide by N
        u = 2
        for i in range(2, 7, 2):
            n = 2**i
            if transform == 'dct':
                img_dct_8x8_quantized = np.trunc(img_dct_8x8 / n)
                img_decoded = combine_nxn(idct2_nxn(n*img_dct_8x8_quantized))
            elif transform == 'fft':
                img_dct_8x8_quantized = np.trunc(img_dct_8x8.real / n) + 1j*np.trunc(img_dct_8x8.imag / n)
                img_decoded = combine_nxn(np.abs(ifft2_nxn(n*img_dct_8x8_quantized)))
            elif transform == 'none':
                img_dct_8x8_quantized = np.trunc(img_dct_8x8 / n)
                img_decoded = combine_nxn(n*img_dct_8x8_quantized)

            p = psnr(img, img_decoded)
            if transform == 'fft':
                h = entropy(img_dct_8x8_quantized.real + img_dct_8x8_quantized.imag)
            else:
                h = entropy(img_dct_8x8_quantized)

            title = 'Coeff divide by N=' +str(n) + ', PSNR=' + str(p) + ', H=' + str(h)
            print '**', title
            plt_arr[u//3, u%3].set_title(title)
            plt_arr[u//3, u%3].imshow(img_decoded[200:328, 200:328], cmap=cm.Greys_r, vmin=0, vmax=255, interpolation="nearest")
            u += 1
            # display(img_decoded)

        # 3. preserving the 8 largest coefficients & rounding
        n = 8  # n-largest
        if transform == 'fft':
            img_dct_8x8_quantized = np.zeros_like(img_dct_8x8, dtype=np.complex)
        else:
            img_dct_8x8_quantized = np.zeros_like(img_dct_8x8)

        for x in range(img_dct_8x8.shape[1]):
            for y in range(img_dct_8x8.shape[0]):
                # get array of flat indices of the 8 largest
                indices = np.abs(img_dct_8x8[y, x]).flatten().argsort()[-n:]
                # set up zero-array, flatten, then  populate with 8 largest, and reshape to nxn
                tmp = np.zeros_like(img_dct_8x8[0, 0]).flatten()
                tmp[indices] = np.round(img_dct_8x8[y, x].real.flatten()[indices])
                img_dct_8x8_quantized[y, x] = tmp.reshape(img_dct_8x8[0, 0].shape)

        if transform == 'fft':
            img_decoded = combine_nxn(np.abs(ifft2_nxn(img_dct_8x8_quantized)))
        else:
            img_decoded = combine_nxn(idct2_nxn(img_dct_8x8_quantized))

        p = psnr(img, img_decoded)
        if transform == 'fft':
            h = entropy(np.append(img_dct_8x8_quantized.real, img_dct_8x8_quantized.imag))
        else:
            h = entropy(img_dct_8x8_quantized)

        title = 'Using ' +str(n) + ' largest, PSNR=' + str(p) + ', H=' + str(h)
        print '**', title
        plt_arr[u//3, u%3].set_title(title)
        plt_arr[u//3, u%3].imshow(img_decoded[200:328,200:328], cmap=cm.Greys_r, vmin=0, vmax=255, interpolation="nearest")

        print
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        plt.show()


    # Do JPEG now for color images.
    # In Matlab, use the rgb2ycbcr command to convert the Red-Green-Blue image to a Lumina and Chroma one;
    # then perform the JPEG-style compression on each one of the three channels independently.
    # After inverting the compression, invert the color transform and visualize the result.
    # While keeping the compression ratio constant for the Y channel, increase the compression of the two chrominance
    # channels and observe the results.
    img = cv2.imread(img_file)
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    img_y, img_cr, img_cb = cv2.split(img_ycrcb)

    title = 'Color image'
    print '*', title
    f, plt_arr = plt.subplots(2, 3)
    f.canvas.set_window_title(title)

    title = 'Original, H=' + str(entropy(img_ycrcb))
    u = 0
    # for plotting
    b, g, r = cv2.split(img)
    img_rgb = cv2.merge((r, g, b))
    plt_arr[u//3, u%3].set_title(title)
    plt_arr[u//3, u%3].imshow(img_rgb[200:328, 200:328], vmin=0, vmax=255, interpolation="nearest")
    u += 1

    # DCT
    img_y_dct_8x8 = dct2_nxn(gen_nxn(img_y, n=8))
    img_cr_dct_8x8 = dct2_nxn(gen_nxn(img_cr, n=8))
    img_cb_dct_8x8 = dct2_nxn(gen_nxn(img_cb, n=8))


    # Quantize
    for i in range(0, 10, 2):
        n = 2**i

        # Quantize Y
        #q_table = n * jpeg_quantization_table_luma()
        q_table = jpeg_quantization_table_luma()
        img_y_dct_8x8_quantized = quantize(img_y_dct_8x8, q_table)
        img_y_decoded = combine_nxn(idct2_nxn(quantize(img_y_dct_8x8_quantized, q_table, inverse=True)))

        # Quantize U,V
        q_table = n * jpeg_quantization_table_chroma()
        img_cb_dct_8x8_quantized = quantize(img_cb_dct_8x8, q_table)
        img_cb_decoded = combine_nxn(idct2_nxn(quantize(img_cb_dct_8x8_quantized, q_table, inverse=True)))
        img_cr_dct_8x8_quantized = quantize(img_cr_dct_8x8, q_table)
        img_cr_decoded = combine_nxn(idct2_nxn(quantize(img_cr_dct_8x8_quantized, q_table, inverse=True)))

        img_ycrcb_decoded = cv2.merge((img_y_decoded, img_cr_decoded, img_cb_decoded)).astype(np.uint8)
        img_decoded = cv2.cvtColor(img_ycrcb_decoded, cv2.COLOR_YCR_CB2BGR)

        h = entropy(np.append(np.append(img_y_dct_8x8_quantized, img_cb_dct_8x8_quantized), img_cr_dct_8x8_quantized))
        p = psnr(img, img_decoded)
        title = 'Chroma q-table divided by ' + str(n) + ', PSNR=' + str(p) + ' H=' + str(h)
        print '**', title

        # for plotting
        b, g, r = cv2.split(img_decoded)
        img_rgb = cv2.merge((r, g, b))
        plt_arr[u//3, u%3].set_title(title)
        plt_arr[u//3, u%3].imshow(img_rgb[200:328, 200:328], vmin=0, vmax=255, interpolation="nearest")
        u += 1

    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.show()

if __name__ == '__main__':
    main()

