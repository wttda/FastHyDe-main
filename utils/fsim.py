import numpy as np
from scipy import ndimage
from scipy import signal
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import cv2


def FeatureSIM(imageRef, imageDis):
    # Convert images to double
    imageRef = np.double(imageRef)
    imageDis = np.double(imageDis)

    # Check if images are in [0, 1] range and convert to [0, 255]
    if imageRef.max() <= 1.0 and imageRef.min() >= 0.0:
        imageRef = imageRef * 255.0
        imageDis = imageDis * 255.0

    # Convert to grayscale if color images
    if len(imageRef.shape) == 3:
        Y1 = 0.299 * imageRef[:, :, 0] + 0.587 * imageRef[:, :, 1] + 0.114 * imageRef[:, :, 2]
        Y2 = 0.299 * imageDis[:, :, 0] + 0.587 * imageDis[:, :, 1] + 0.114 * imageDis[:, :, 2]
    else:
        Y1 = imageRef
        Y2 = imageDis

    # Downsample the image
    minDimension = min(Y1.shape)
    F = max(1, round(minDimension / 256))
    aveKernel = np.ones((F, F)) / (F * F)

    Y1 = signal.convolve2d(Y1, aveKernel, mode='same')
    Y2 = signal.convolve2d(Y2, aveKernel, mode='same')
    Y1 = Y1[::F, ::F]
    Y2 = Y2[::F, ::F]

    # Calculate the phase congruency maps
    PC1 = phasecong2(Y1)
    PC2 = phasecong2(Y2)

    # Calculate the gradient map
    dx = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]]) / 16
    dy = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]]) / 16

    IxY1 = signal.convolve2d(Y1, dx, mode='same')
    IyY1 = signal.convolve2d(Y1, dy, mode='same')
    gradientMap1 = np.sqrt(IxY1**2 + IyY1**2)

    IxY2 = signal.convolve2d(Y2, dx, mode='same')
    IyY2 = signal.convolve2d(Y2, dy, mode='same')
    gradientMap2 = np.sqrt(IxY2**2 + IyY2**2)

    # Calculate the FSIM
    T1 = 0.85
    T2 = 160
    PCSimMatrix = (2 * PC1 * PC2 + T1) / (PC1**2 + PC2**2 + T1)
    gradientSimMatrix = (2 * gradientMap1 * gradientMap2 + T2) / (gradientMap1**2 + gradientMap2**2 + T2)
    PCm = np.maximum(PC1, PC2)
    SimMatrix = gradientSimMatrix * PCSimMatrix * PCm
    FSIM = np.sum(SimMatrix) / np.sum(PCm)

    return FSIM

def phasecong2(im):
    nscale = 4  # Number of wavelet scales.
    norient = 4  # Number of filter orientations.
    minWaveLength = 6  # Wavelength of smallest scale filter.
    mult = 2  # Scaling factor between successive filters.
    sigmaOnf = 0.55  # Ratio of the standard deviation of the Gaussian describing the log Gabor filter's transfer function in the frequency domain to the filter center frequency.
    dThetaOnSigma = 1.2  # Ratio of angular interval between filter orientations and the standard deviation of the angular Gaussian function used to construct filters in the freq. plane.
    k = 2.0  # No of standard deviations of the noise energy beyond the mean at which we set the noise threshold point.
    epsilon = 0.0001  # Used to prevent division by zero.

    thetaSigma = np.pi / norient / dThetaOnSigma  # Calculate the standard deviation of the angular Gaussian function used to construct filters in the freq. plane.

    rows, cols = im.shape
    imagefft = fft2(im)  # Fourier transform of image

    zero = np.zeros((rows, cols))
    EO = np.zeros((nscale, norient), dtype=object)  # Array of convolution results.

    estMeanE2n = []
    ifftFilterArray = np.zeros((1, nscale), dtype=object)  # Array of inverse FFTs of filters

    # Pre-compute some stuff to speed up filter construction

    # Set up X and Y matrices with ranges normalised to +/- 0.5
    # The following code adjusts things appropriately for odd and even values of rows and columns.
    if cols % 2:
        xrange = np.arange(-(cols-1)/2, (cols-1)/2+1) / (cols-1)
    else:
        xrange = np.arange(-cols/2, cols/2) / cols

    if rows % 2:
        yrange = np.arange(-(rows-1)/2, (rows-1)/2+1) / (rows-1)
    else:
        yrange = np.arange(-rows/2, rows/2) / rows

    x, y = np.meshgrid(xrange, yrange)

    radius = np.sqrt(x**2 + y**2)  # Matrix values contain *normalised* radius from centre.
    theta = np.arctan2(-y, x)  # Matrix values contain polar angle.
    # (note -ve y is used to give +ve anti-clockwise angles)

    radius = ifftshift(radius)  # Quadrant shift radius and theta so that filters
    theta = ifftshift(theta)  # are constructed with 0 frequency at the corners.
    radius[0, 0] = 1  # Get rid of the 0 radius value at the 0 frequency point (now at top-left corner)
    # so that taking the log of the radius will not cause trouble.

    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    # clear x; clear y; clear theta;    # save a little memory

    # Filters are constructed in terms of two components.
    # 1) The radial component, which controls the frequency band that the filter
    #    responds to
    # 2) The angular component, which controls the orientation that the filter
    #    responds to.
    # The two components are multiplied together to construct the overall filter.

    # Construct the radial filter components...

    # First construct a low-pass filter that is as large as possible, yet falls
    # away to zero at the boundaries.  All log Gabor filters are multiplied by
    # this to ensure no extra frequencies at the 'corners' of the FFT are
    # incorporated as this seems to upset the normalisation process when
    # calculating phase congrunecy.
    lp = lowpassfilter((rows, cols), 0.45, 15)  # Radius .45, 'sharpness' 15

    logGabor = np.zeros((1, nscale), dtype=object)

    for s in range(nscale):
        wavelength = minWaveLength * (mult ** s)
        fo = 1.0 / wavelength  # Centre frequency of filter.
        logGabor[0, s] = np.exp((-(np.log(radius / fo))**2) / (2 * (np.log(sigmaOnf))**2))
        logGabor[0, s] = logGabor[0, s] * lp  # Apply low-pass filter
        logGabor[0, s][0, 0] = 0  # Set the value at the 0 frequency point of the filter
        # back to zero (undo the radius fudge).

    # Then construct the angular filter components...

    spread = np.zeros((1, norient), dtype=object)

    for o in range(norient):
        angl = (o) * np.pi / norient  # Filter angle.

        # For each point in the filter matrix calculate the angular distance from
        # the specified filter orientation.  To overcome the angular wrap-around
        # problem sine difference and cosine difference values are first computed
        # and then the atan2 function is used to determine angular distance.

        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)  # Difference in sine.
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)  # Difference in cosine.
        dtheta = np.abs(np.arctan2(ds, dc))  # Absolute angular distance.
        spread[0, o] = np.exp(-(dtheta**2) / (2 * thetaSigma**2))  # Calculate the angular filter component.

    # The main loop...
    EnergyAll = np.zeros((rows, cols))
    AnAll = np.zeros((rows, cols))

    for o in range(norient):  # For each orientation.
        sumE_ThisOrient = zero  # Initialize accumulator matrices.
        sumO_ThisOrient = zero
        sumAn_ThisOrient = zero
        Energy = zero

        for s in range(nscale):  # For each scale.
            filter = logGabor[0, s] * spread[0, o]  # Multiply radial and angular components to get the filter.
            ifftFilt = np.real(ifft2(filter)) * np.sqrt(rows * cols)  # Note rescaling to match power
            ifftFilterArray[0, s] = ifftFilt  # record ifft2 of filter
            # Convolve image with even and odd filters returning the result in EO
            EO[s, o] = ifft2(imagefft * filter)

            An = np.abs(EO[s, o])  # Amplitude of even & odd filter response.
            sumAn_ThisOrient = sumAn_ThisOrient + An  # Sum of amplitude responses.
            sumE_ThisOrient = sumE_ThisOrient + np.real(EO[s, o])  # Sum of even filter convolution results.
            sumO_ThisOrient = sumO_ThisOrient + np.imag(EO[s, o])  # Sum of odd filter convolution results.
            if s == 0:  # Record mean squared filter value at smallest scale. This is used for noise estimation.
                EM_n = np.sum(filter**2)
                maxAn = An  # Record the maximum An over all scales.
            else:
                maxAn = np.maximum(maxAn, An)

        # Get weighted mean filter response vector, this gives the weighted mean phase angle.
        XEnergy = np.sqrt(sumE_ThisOrient**2 + sumO_ThisOrient**2) + epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        # Now calculate An(cos(phase_deviation) - | sin(phase_deviation)) | by using dot and cross products
        # between the weighted mean filter response vector and the individual filter response vectors at each scale.
        # This quantity is phase congruency multiplied by An, which we call energy.
        for s in range(nscale):
            E = np.real(EO[s, o])
            O = np.imag(EO[s, o])  # Extract even and odd convolution results.
            Energy = Energy + E * MeanE + O * MeanO - np.abs(E * MeanO - O * MeanE)

        # Compensate for noise
        # We estimate the noise power from the energy squared response at the smallest scale.  If the noise is Gaussian
        # the energy squared will have a Chi-squared 2DOF pdf.  We calculate the median energy squared response
        # as this is a robust statistic.  From this we estimate the mean.
        # The estimate of noise power is obtained by dividing the mean squared energy value by the mean squared filter value

        medianE2n = np.median(np.abs(EO[0, o])**2)
        meanE2n = -medianE2n / np.log(0.5)
        estMeanE2n.append(meanE2n)

        noisePower = meanE2n / EM_n  # Estimate of noise power.

        # Now estimate the total energy^2 due to noise
        # Estimate for sum(An^2) + sum(Ai.*Aj.*(cphi.*cphj + sphi.*sphj))

        EstSumAn2 = zero
        for s in range(nscale):
            EstSumAn2 = EstSumAn2 + ifftFilterArray[0, s]**2

        EstSumAiAj = zero
        for si in range(nscale - 1):
            for sj in range(si + 1, nscale):
                EstSumAiAj = EstSumAiAj + ifftFilterArray[0, si] * ifftFilterArray[0, sj]

        sumEstSumAn2 = np.sum(EstSumAn2)
        sumEstSumAiAj = np.sum(EstSumAiAj)

        EstNoiseEnergy2 = 2 * noisePower * sumEstSumAn2 + 4 * noisePower * sumEstSumAiAj

        tau = np.sqrt(EstNoiseEnergy2 / 2)  # Rayleigh parameter
        EstNoiseEnergy = tau * np.sqrt(np.pi / 2)  # Expected value of noise energy
        EstNoiseEnergySigma = np.sqrt((2 - np.pi / 2) * tau**2)

        T = EstNoiseEnergy + k * EstNoiseEnergySigma  # Noise threshold

        # The estimated noise effect calculated above is only valid for the PC_1 measure.
        # The PC_2 measure does not lend itself readily to the same analysis.  However
        # empirically it seems that the noise effect is overestimated roughly by a factor
        # of 1.7 for the filter parameters used here.

        T = T / 1.7  # Empirical rescaling of the estimated noise effect to suit the PC_2 phase congruency measure
        Energy = np.maximum(Energy - T, zero)  # Apply noise threshold

        EnergyAll = EnergyAll + Energy
        AnAll = AnAll + sumAn_ThisOrient

    ResultPC = EnergyAll / AnAll
    return ResultPC

def lowpassfilter(sze, cutoff, n):
    if cutoff < 0 or cutoff > 0.5:
        raise ValueError('cutoff frequency must be between 0 and 0.5')
    if n % 1 != 0 or n < 1:
        raise ValueError('n must be an integer >= 1')

    rows, cols = sze
    if cols % 2:
        xrange = np.arange(-(cols-1)/2, (cols-1)/2+1) / (cols-1)
    else:
        xrange = np.arange(-cols/2, cols/2) / cols

    if rows % 2:
        yrange = np.arange(-(rows-1)/2, (rows-1)/2+1) / (rows-1)
    else:
        yrange = np.arange(-rows/2, rows/2) / rows

    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x**2 + y**2)
    f = 1 / (1 + (radius / cutoff)**(2 * n))
    return np.fft.ifftshift(f)

# Example usage
if __name__ == "__main__":
    # Load images (replace with your image paths)
    imageRef = cv2.imread('Noisy.jpg', cv2.IMREAD_GRAYSCALE)
    imageDis = cv2.imread('TNN.jpg', cv2.IMREAD_GRAYSCALE)

    # Ensure images are of the same size
    if imageRef.shape != imageDis.shape:
        imageDis = cv2.resize(imageDis, (imageRef.shape[1], imageRef.shape[0]))

    # Normalize images to [0, 1]
    imageRef = imageRef / 255.0
    imageDis = imageDis / 255.0

    # Compute FSIM
    fsim = FeatureSIM(imageRef, imageDis)
    print(f"FSIM: {fsim}")