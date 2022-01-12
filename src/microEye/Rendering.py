from os import name
import re

import cv2
import numba
import numpy as np
import pandas as pd
from pandas.core.window.rolling import Window
import pyqtgraph as pg
from numpy.lib.type_check import imag
from PyQt5.QtCore import *
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift
import tifffile


def explicit2dGauss(mu, sigma, corr, flux, offset, shape):
    '''Generates a 2D Multivariate Gaussian

    Params
    -------
    mu (tuple | list)
        vector holding [yc, xc] the distribution mean
    sigma (tuple | list)
        the standard deviation [y_sigma, x_sigma]
    corr (float)
        the correlation between X and Y
    flux (float)
        the flux
    offset (float)
        the distribution offset
    shape (tuple | list)
        the shape of 2D produced gaussian array

    Returns
    -------
    Z (np.ndarray)
        a 2D Multivariate Gaussian array
    '''
    diag2nd = corr * sigma[0] * sigma[1]
    covar = np.array([[sigma[0], diag2nd], [diag2nd, sigma[1]]])

    cov_inv = np.linalg.inv(covar)  # inverse of covariance matrix
    cov_det = np.linalg.det(covar)  # determinant of covariance matrix
    # Plotting
    y_len = np.arange(0, shape[0])
    x_len = np.arange(0, shape[1])
    X, Y = np.meshgrid(x_len, y_len)
    coe = flux / ((2 * np.pi)**2 * cov_det)**0.5
    return (coe * np.exp(-0.5 * (cov_inv[0, 0]*(X-mu[0])**2 + (cov_inv[0, 1]
            + cov_inv[1, 0])*(X-mu[0])*(Y-mu[1]) + cov_inv[1, 1]*(Y-mu[1])**2))
            ) + offset


def radial_cordinate(shape):
    '''Generates a 2D array with radial cordinates
    with according to the first two axis of the
    supplied shape tuple

    Returns
    -------
    R, Rsq
        Radius 2d matrix (R) and radius squared matrix (Rsq)
    '''
    y_len = np.arange(-shape[0]//2, shape[0]//2)
    x_len = np.arange(-shape[1]//2, shape[1]//2)

    X, Y = np.meshgrid(x_len, y_len)

    Rsq = (X**2 + Y**2)

    return np.sqrt(Rsq), Rsq


class gauss_hist_render:

    def __init__(self, pixelSize=10):
        self._pixel_size = pixelSize
        self._std = 10  # nm
        self._gauss_std = self._std / self._pixel_size
        self._gauss_len = 1 + np.ceil(self._gauss_std * 6)
        if self._gauss_len % 2 == 0:
            self._gauss_len += 1
        self._gauss_shape = [int(self._gauss_len)] * 2
        self._gauss_2d = explicit2dGauss(
            mu=[(self._gauss_len - 1) / 2] * 2,
            sigma=[self._gauss_std] * 2,
            corr=0,
            flux=1,
            offset=0,
            shape=self._gauss_shape)
        self._image = None

    def render(self, X_loc, Y_loc, Intensity, shape=None):
        '''Renders as super resolution image from
        single molecule localizations.

        Params
        -------
        X_loc (np.ndarray)
            Sub-pixel localized points X coordinates
        Y_loc (np.ndarray)
            Sub-pixel localized points Y coordinates
        Intensity (np.ndarray)
            Sub-pixel localized points intensity estimate
        shape tuple(int, int), optional
            Super-res image (height, width), by default None

        Returns
        -------
        Image (np.ndarray)
            the rendered 2D super-res image array
        '''
        if not len(X_loc) == len(Y_loc) == len(Intensity):
            raise Exception('The supplied arguments are of different lengths.')

        if shape is None:
            x_max = int((np.max(X_loc) / self._pixel_size) +
                        4 * self._gauss_len)
            y_max = int((np.max(Y_loc) / self._pixel_size) +
                        4 * self._gauss_len)
        else:
            x_max = shape[1]
            y_max = shape[0]
        n_max = max(x_max, y_max)

        step = int((self._gauss_len - 1) // 2)

        self._image = np.zeros([n_max, n_max])

        for idx in range(len(X_loc)):
            x = round(X_loc[idx] / self._pixel_size) + 4 * step
            y = round(Y_loc[idx] / self._pixel_size) + 4 * step

            self._image[y - step:y + step + 1, x - step:x + step + 1] += \
                Intensity[idx] * self._gauss_2d

        return self._image

    def fromArray(self, data: np.ndarray, shape=None):
        '''Renders as super resolution image from
        single molecule localizations.

        Params
        -------
        data (np.ndarray)
            Array with sub-pixel localization data columns (X, Y, Intensity)
        shape tuple(int, int), optional
            Super-res image (height, width), by default None

        Returns
        -------
        Image (np.ndarray)
            the rendered 2D super-res image array
        '''
        return self.render(data[:, 0], data[:, 1], data[:, 2], shape)


def FRC_resolution_binomial(data: np.ndarray, pixelSize=10):
    '''Fourier Ring Correlation based on
    https://www.frontiersin.org/articles/10.3389/fbinf.2021.817254/

    Parameters
    ----------
    data : np.ndarray
        data to be resolution estimated, columns (X, Y, intensity)
    pixelSize : int, optional
        super resolution image pixel size in nanometers, by default 10
    '''
    n_points = data.shape[0]

    coin = np.random.binomial(1, 0.5, (n_points, 1))
    data = np.hstack((data, coin))

    # Two separate datsets based on the value of the last column are generated.
    data1, data2 = data[data[:, -1] == 0], data[data[:, -1] == 1]

    gaussHist = gauss_hist_render(pixelSize)

    image_1 = gaussHist.fromArray(data1)
    image_2 = gaussHist.fromArray(data2)

    image_1, image_2 = match_shape(image_1, image_2)

    image_1 /= np.sum(image_1)
    image_2 /= np.sum(image_2)

    window = hamming_2Dwindow(image_1.shape[0])

    image_1 *= window
    image_2 *= window

    fft_1 = np.fft.fft2(image_1)
    fft_2 = np.fft.fft2(image_2)
    fft_12 = np.fft.fftshift(
        np.real(fft_1 * np.conj(fft_2)))

    fft_11 = np.fft.fftshift(np.abs(fft_1)**2)
    fft_22 = np.fft.fftshift(np.abs(fft_2)**2)

    R, _ = radial_cordinate(image_1.shape)
    R = np.round(R)

    # Get the Nyquist frequency
    # freq_nyq = int(np.floor(R.shape[0] / 2.0))
    # R_max = int(np.max(R))

    frequencies = np.fft.rfftfreq(R.shape[0], d=pixelSize)
    freq_nyq = frequencies.max()
    R_max = frequencies.shape[0]

    FRC_res = np.zeros((R_max, 5))

    for r in range(1, R_max + 1):
        ring = (R == r)
        FRC_res[r-1, 0] = np.sum(fft_12[ring])
        FRC_res[r-1, 1] = np.sum(fft_11[ring])
        FRC_res[r-1, 2] = np.sum(fft_22[ring])
        FRC_res[r-1, 3] = (FRC_res[r-1, 0] / np.sqrt(
            FRC_res[r-1, 1] * FRC_res[r-1, 2]))
        # FRC_res[r-1, 4] =

    # frequencies = np.linspace(0, 1/(2*pixelSize), R_max)

    FRC = pd.DataFrame(FRC_res[:, 3])
    FRC.interpolate(
        method='linear', inplace=True
    )

    idx = np.where(FRC <= (1/7))[0].min()
    FRC_res = 1 / frequencies[idx]

    # plot results
    plt = pg.plot()

    plt.showGrid(x=True, y=True)
    plt.addLegend()

    # set properties of the label for y axis
    plt.setLabel('left', 'FRC', units='')

    # set properties of the label for x axis
    plt.setLabel('bottom', 'Spatial Frequency [1/nm]', units='')

    plt.setWindowTitle(
        "FRC resolution: {0} nm".format(
            np.round(FRC_res, 1))
        )

    # setting horizontal range
    plt.setXRange(0, frequencies[-1])

    # setting vertical range
    plt.setYRange(0, 1)

    line1 = plt.plot(
        frequencies, FRC.to_numpy()[:, 0],
        pen='r', symbol='x', symbolPen='r',
        symbolBrush=0.2, name='FRC')
    line2 = plt.plotItem.addLine(y=1/7, pen='y')


def FRC_resolution_check_pattern(image, pixelSize=10):
    '''Fourier Ring Correlation based on
    https://doi.org/10.1038/s41467-019-11024-z


    Parameters
    ----------
    image : np.ndarray
        image to be resolution estimated
    pixelSize : int, optional
        super resolution image pixel size in nanometers, by default 10
    '''
    odd, even, oddeven, evenodd = checker_pairs(image)

    window = hamming_2Dwindow(odd.shape[0])

    odd *= window
    even *= window
    oddeven *= window
    evenodd *= window

    odd /= np.sum(odd)
    even /= np.sum(even)
    oddeven /= np.sum(oddeven)
    evenodd /= np.sum(evenodd)

    odd_fft, even_fft, oddeven_fft, evenodd_fft = \
        np.fft.fft2(odd), np.fft.fft2(even), \
        np.fft.fft2(oddeven), np.fft.fft2(evenodd)

    odd_even = np.fft.fftshift(
        np.real(odd_fft * np.conj(even_fft)))
    odd_even2_odd = np.fft.fftshift(
        np.real(oddeven_fft * np.conj(evenodd_fft)))

    odd_sq, even_sq, oddeven_sq, evenodd_sq = (
        np.fft.fftshift(np.abs(odd_fft)**2),
        np.fft.fftshift(np.abs(even_fft)**2),
        np.fft.fftshift(np.abs(oddeven_fft)**2),
        np.fft.fftshift(np.abs(evenodd_fft)**2))

    R, _ = radial_cordinate(odd.shape)

    R = np.round(R)

    # Get the Nyquist frequency
    # freq_nyq = int(np.floor(R.shape[0] / 2.0))
    # R_max = int(np.max(R))

    frequencies = np.fft.rfftfreq(R.shape[0], d=pixelSize)
    freq_nyq = frequencies.max()
    R_max = frequencies.shape[0]

    FRC_res_1 = np.zeros((R_max, 4))
    FRC_res_2 = np.zeros((R_max, 4))

    for r in range(1, R_max + 1):
        ring = (R == r)
        FRC_res_1[r-1, 0] = np.sum(odd_even[ring])
        FRC_res_1[r-1, 1] = np.sum(odd_sq[ring])
        FRC_res_1[r-1, 2] = np.sum(even_sq[ring])
        FRC_res_1[r-1, 3] = (FRC_res_1[r-1, 0] / np.sqrt(
            FRC_res_1[r-1, 1]*FRC_res_1[r-1, 2]))

        FRC_res_2[r-1, 0] = np.sum(odd_even2_odd[ring])
        FRC_res_2[r-1, 1] = np.sum(oddeven_sq[ring])
        FRC_res_2[r-1, 2] = np.sum(evenodd_sq[ring])
        FRC_res_2[r-1, 3] = (FRC_res_2[r-1, 0] / np.sqrt(
            FRC_res_2[r-1, 1]*FRC_res_2[r-1, 2]))

    # frequencies = np.linspace(0, 1, freq_nyq) / (2 * pixelSize)

    FRC_1 = pd.DataFrame(FRC_res_1[:, 3])
    FRC_1.interpolate(
        method='linear', inplace=True
    )

    FRC_2 = pd.DataFrame(FRC_res_2[:, 3])
    FRC_2.interpolate(
        method='linear', inplace=True
    )

    FRC_avg = pd.DataFrame(0.5*(FRC_res_1[:, 3] + FRC_res_2[:, 3]))
    FRC_avg.interpolate(
        method='linear', inplace=True
    )

    idxmax_1 = np.where(FRC_1 <= (1/7))[0].min()
    idxmax_2 = np.where(FRC_2 <= (1/7))[0].min()
    idxmax_avg = np.where(FRC_avg <= (1/7))[0].min()
    FRC_freq = np.array([
        frequencies[idxmax_1],
        frequencies[idxmax_2],
        frequencies[idxmax_avg]])
    FRC_res = 1 / FRC_freq

    def func(x, a, b, c, d):
        return a * np.exp(c * (x - b)) + d

    params = [0.95988146, 0.97979108, 13.90441896, 0.55146136]

    cut_off_corrections = func(FRC_freq, *params)

    FRC_res /= cut_off_corrections

    # plot results
    plt = pg.plot()

    plt.showGrid(x=True, y=True)
    plt.addLegend()

    # set properties of the label for y axis
    plt.setLabel('left', 'FRC', units='')

    # set properties of the label for x axis
    plt.setLabel('bottom', 'Spatial Frequency [1/nm]', units='')

    plt.setWindowTitle(
        "FRC resolution (1st, 2nd, avg): {0} | {1} | {2} nm".format(
            *np.round(FRC_res, 1))
        )

    # setting horizontal range
    plt.setXRange(0, frequencies[-1])

    # setting vertical range
    plt.setYRange(0, 1)

    line1 = plt.plot(
        frequencies * cut_off_corrections[0], FRC_1.to_numpy()[:, 0],
        pen='g', symbol='x', symbolPen='g',
        symbolBrush=0.2, name='1st Set')
    line2 = plt.plot(
        frequencies * cut_off_corrections[1], FRC_2.to_numpy()[:, 0],
        pen='b', symbol='o', symbolPen='b',
        symbolBrush=0.2, name='2nd Set')
    line3 = plt.plot(
        frequencies * cut_off_corrections[2], FRC_avg.to_numpy()[:, 0],
        pen='r', symbol='+', symbolPen='r',
        symbolBrush=0.2, name='Average')
    line4 = plt.plotItem.addLine(y=1/7, pen='y')


def checker_pairs(image: np.ndarray):

    shape = image.shape

    odd_index = [np.arange(1, shape[i], 2) for i in range(len(shape))]
    even_index = [np.arange(0, shape[i], 2) for i in range(len(shape))]

    # first set
    odd = image[odd_index[0], :][:, odd_index[1]]
    even = image[even_index[0], :][:, even_index[1]]

    odd, even = match_shape(odd, even)

    # reverse set
    oddeven = image[odd_index[0], :][:, even_index[1]]
    evenodd = image[even_index[0], :][:, odd_index[1]]

    oddeven, evenodd = match_shape(oddeven, evenodd)

    return odd, even, oddeven, evenodd


def expand_image(image: np.ndarray, shape):
    if image.shape == shape:
        return image
    else:
        ret = np.zeros(shape)
        ret[:image.shape[0], :image.shape[1]] = image
        return ret


def match_shape(image1: np.ndarray, image2: np.ndarray):
    shape = tuple(max(x, y) for x, y in zip(image1.shape, image2.shape))

    if any(map(lambda x, y: x != y, image1.shape, shape)):
        image1 = expand_image(image1, shape)
    if any(map(lambda x, y: x != y, image2.shape, shape)):
        image2 = expand_image(image2, shape)

    return image1, image2


def hamming_2Dwindow(size: int):
    window1d = np.hamming(size)
    return np.sqrt(np.outer(window1d, window1d))

# g = gauss_hist_render(10)
# img = cv2.normalize(g._gauss_2d, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
# img = cv2.resize(
#     img, (0, 0), fx=100, fy=100, interpolation=cv2.INTER_NEAREST)
# cv2.imshow("k", img)
# cv2.waitKey(0)

# H, _, _ = np.histogram2d(
#     np.array(self.locX).copy() * 11.5,
#     np.array(self.locY).copy() * 11.5,
#     np.array(shape).copy() * 12)
# cv2.imshow('localization', H)
# cv2.waitKey(1)
