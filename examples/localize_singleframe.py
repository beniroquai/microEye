
import NanoImagingPack as nip
import numpy as np
from microEye.Filters import BandpassFilter
from microEye.fitting.fit import CV_BlobDetector
from microEye.fitting.results import FittingMethod
from microEye.fitting.fit import localize_frame

# tune parameters
image = np.random.rand(100, 100)
index = 1
filtered = nip.gaussf(image, 1.5)
varim = None
detector = CV_BlobDetector()
filter = BandpassFilter()

# localize  frame 
# params = > x,y,background, max(0, intensity), magnitudeX / magnitudeY
frames, params, crlbs, loglike = localize_frame(
            index,
            image,
            filtered,
            varim,
            filter,
            detector,
            rel_threshold=0.4,
            PSFparam=np.array([1.5]),
            roiSize=13,
            method=FittingMethod._2D_Phasor_CPU)

