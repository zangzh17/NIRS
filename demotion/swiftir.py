#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''SWiFT-IR - Image registration tools using partially whitened spectra

This is a reimplementation by Daniel Wagenaar <daw@caltech.edu> of the
original SWiFT-IR programs (swim, mir, iscale, iavg, remod) written
by Art Wetzel <awetzel@psc.edu>.

The goal of this implementation is to make use of modern libraries like
OpenCV and numpy for efficiency and to provide a pythonic interface to
the algorithms in the hopes of facilitating integration with other tools.

This implementation does not quite (yet) implement all the functionality
of Art's programs.

Documentation is presently only in the form of docstrings for each function.
The main functions to look at are: LOADIMAGE, SWIM, MIRAFFINE, AVERAGEIMAGE,
AVERAGEIMAGENOBLACK, REMOD, SAVEIMAGE.

Example scripts that implement the most common use cases of the original
programs are planned.'''

import numpy as np
import cv2

apo = []
apo0 = []


def si_unpackSize(siz):
    '''SI_UNPACKSIZE - Interprets size arguments
    Sizes are NOT ANYMORE rounded up to next even number. 
    Size may be given as one or two numbers. Unpacks to tuple.'''
    if type(siz)==np.ndarray or type(siz)==list or type(siz)==tuple:
        w = siz[0]
        h = siz[-1]
    else:
        w = h = siz
    return (w, h)


def reptoshape(mat, pattern):
    '''REPTOSHAPE - Repeat a matrix to match shape of other matrix
    REPTOSHAPE(mat, pattern) returns a copy of the matrix MAT replicated
    to match the shape of PATTERN. For instance, if MAT is an N-vector
    or an Nx1 matrix, and PATTERN is NxK, the output will be an NxK matrix
    of which each the columns is filled with the contents of MAT.
    Higher dimensional cases are handled as well, but non-singleton dimensions
    of MAT must always match the corresponding dimension of PATTERN.'''
    ps = [x for x in pattern.shape]
    ms = [x for x in mat.shape]
    while len(ms)<len(ps):
        ms.append(1)
    mat = np.reshape(mat, ms)
    for d in range(len(ps)):
        if ms[d]==1:
            mat = np.repeat(mat, ps[d], d)
        elif ms[d] != ps[d]:
            raise ValueError('Cannot broadcast'  + str(mat.shape) + ' to '
                             + str(pattern.shape))
    return mat


def shiftAffine(afm, dx):
    '''SHIFTAFFINE - Add a translation to amn affine transform
    SHIFTAFFINE(afm, dx) composes the transformation x -> x + DX after
    the affine transformation AFM.'''
    return afm + np.array([[0,0,dx[0]],[0,0,dx[1]]])
            

def invertAffine(afm):
    '''INVERTAFFINE - Invert affine transform
    INVERTAFFINE(afm), where AFM is a 2x3 affine transformation matrix,
    returns the inverse transform.'''
    afm = np.vstack((afm, [0,0,1]))
    ifm = np.linalg.inv(afm)
    return ifm[:2,:]


def invertLinear(tfm):
    '''INVERTLINEAR - Invert linear transform
    INVERTLINEAR(tfm), where TFM is a 2x2 linear transformation matrix,
    returns the inverse transform.'''
    return np.linalg.inv(tfm)


def composeAffine(afm, bfm):
    '''COMPOSEAFFINE - Compose two affine transforms
    COMPOSEAFFINE(afm1, afm2) returns the affine transform AFM1 ∘ AFM2
    that applies AFM1 after AFM2.
    Affine matrices must be 2x3 numpy arrays.'''
    afm = np.vstack((afm, [0,0,1]))
    bfm = np.vstack((bfm, [0,0,1]))
    fm = np.matmul(afm, bfm)
    return fm[0:2,:]


def applyAffine(afm, xy):
    '''APPLYAFFINE - Apply affine transform to a point
    xy_ = APPLYAFFINE(afm, xy) applies the affine matrix AFM to the point XY
    Affine matrix must be a 2x3 numpy array. XY may be a list, tuple, or
    2-vector, or a 2xN array.
    The output is always a vector or array.
    (AFM may also be a 3x3 matrix, but the bottom row is simply ignored.)))'''
    if not type(xy)==np.ndarray:
        xy = np.array([xy[0], xy[1]])
    return np.matmul(afm[:2,:2], xy) + reptoshape(afm[:2,2], xy)


def identityAffine():
    '''IDENTITYAFFINE - Return an idempotent affine transform
    afm = IDENTITYAFFINE() returns an affine transform that is
    an identity transform.'''
    return np.array([[1., 0., 0.],
                     [0., 1., 0.]])


def mirAffine(pa, pb):
    '''MIRAFFINE - Calculate affine transform for matching points
    afm, rms, iworst = MIRAFFINE(pa, pb) calculates the affine transform
    AFM that maps the points PA (2xN matrix) to the points PB (2xN matrix).
    If there are three or fewer points, the resulting AFM is a transform
    such that AFM PA = PB. In particular, if there is only one pair of
    points, the result is a simple translation; if there are two, a
    translation combined with an isotropic scaling and rotation;
    if there are three, a perfectly fitting affine transform.
    If there are more than three pairs, the result is the least-
    squares solution: rms = sqrt(sum(||AFM PA - PB||²)) is minimized.
    The index of the worst matching point is returned as well.'''

    N = pa.shape[1]

    if N==0:
        return (np.array([[1., 0, 0], [0, 1, 0]]), 0, None)    
    elif N==1:
        return (np.array([[1., 0, (pb[0]-pa[0])[0]],
                          [0, 1, (pb[1]-pa[1])[0]]]), 0, 0)
    elif N==2:
        # Add a fictive point, by rotating the second point 90 degrees
        # around the first point.
        # This trick is due to Art Wetzel.
        pa = np.hstack((pa, [[pa[0,0] - (pa[1,0]-pa[1,1])],
                             [pa[1,0] + (pa[0,0]-pa[0,1])]]))
        pb = np.hstack((pb, [[pb[0,0] - (pb[1,0]-pb[1,1])],
                             [pb[1,0] + (pb[0,0]-pb[0,1])]]))
        N = 3
    if N==3:
        afm = cv2.getAffineTransform(pa.astype(np.float32).transpose(),
                                     pb.astype(np.float32).transpose())
        return (afm, 0, 0)
    
    # Add row of ones so we can treat the affine as a 3x3 matrix
    # multiplying against [x, y, 1]'.
    pa = np.vstack((pa, np.ones((1,N))))
    pb = np.vstack((pb, np.ones((1,N))))

    # Python solves a M = b rather than M a = b, so transpose everything
    afm = np.linalg.lstsq(pa.transpose(), pb.transpose(), rcond=-1)[0].T
    afm[2,:] = [0, 0, 1] # This should already be more or less true

    # lstsq doesn't return individual residuals, so must recalculate    
    resi = np.matmul(afm, pa) - pb
    err = np.sum(resi*resi, 0)
    rms = np.sqrt(np.mean(err))
    iworst = np.argmax(err)
    
    afm = afm[:2,:] # Drop the trivial bottom row
    return (afm, rms, iworst)


def mirRotate(pa, pb):
    '''MIRROTATE - Return optimal pure rotation
    afm = MIRROTATE(pa, pb), where PA and PB both contain exactly 2 points,
    is like MIRAFFINE, except that the result is a pure rotation plus 
    translation.'''
    ca = np.mean(pa, 1)
    cb = np.mean(pb, 1)
    da = pa[:,1] - pa[:,0]
    db = pb[:,1] - pb[:,0]
    phia = np.arctan2(da[1], da[0])
    phib = np.arctan2(db[1], db[0])
    dphi = phib - phia
    dx = cb - ca
    afm0 = np.array([[np.cos(dphi), -np.sin(dphi), dx[0]],
                     [np.sin(dphi), np.cos(dphi), dx[1]]])
    dp = cb - applyAffine(afm0, ca)
    return shiftAffine(afm0, dp)


def mirIterate(pa, pb, ethresh=3.0, leastpts=4):
    '''MIRITERATE - Iteratively calls MIRAFFINE until error is reasonable
    (afm, rms, npts) = MIRITERATE(pa, pb) iteratively calls MIRAFFINE.
    MIRITERATE returns if the RMS error is below a threshold. Otherwise,
    it drops the worst point pair and iterates.
    Optional third argument ETHRESH specifies the threshold. The default
    value is 3.0.
    Optional fourth argument LEASTPTS specifies that we won't drop points
    if that would cause the number of point pairs to fall below LEASTPTS.
    The default value is 4.
    In addition to the resulting transform, MIRITERATE returns the final
    RMS error and the number of points retained'''

    while True:
        (afm, rms, iworst) = mirAffine(pa, pb)
        if rms>ethresh and pa.shape[1]>leastpts:
            pa = np.delete(pa, iworst, 1)
            pb = np.delete(pb, iworst, 1)
        else:
            break
    return (afm, rms, pa.shape[1])


def movingToStationary(afm, pmov):
    '''MOVINGTOSTATIONARY - Calculate positions in stationary image
    psta = MOVINGTOSTATIONARY(afm, pmov) calculates positions in
    stationary image given positions in moving image and an affine
    transformation as returned by MIRAFFINE(psta, pmov).'''
    psta = applyAffine(invertAffine(afm), pmov)
    return psta


def stationaryToMoving(afm, psta):
    '''STATIONARYTOMOVING - Calculate positions in moving image
    pmov = STATIONARYTOMOVING(afm, psta) calculates positions in
    moving image given positions in stationary image and an affine
    transformation returned by MIRAFFINE(psta, pmov).'''
    pmov = applyAffine(afm, psta)
    return pmov


def toLocal(afm, topleft):
    '''TOLOCAL - Modify affine transform for use with local coordinates
    lafm = TOLOCAL(afm, topleft) takes an affine that transforms global
    model coordinates to coordinates in a moving image, and returns an
    affine for use on a stationary image located at the given position. 
    That is, if AFM acts by 
      p_moving = AFM p_stationary, 
    then LAFM acts by
      p_moving = LAFM p_stationary = AFM (p_stationary + topleft).'''
    dp = applyAffine(afm, topleft) - applyAffine(afm, [0,0])
    return shiftAffine(afm, dp)


def toGlobal(afm, topleft):
    '''TOGLOBAL - Modify affine transform for use with global coordinates
    afm = TOGLOBAL(lafm, topleft) takes an affine that transforms 
    coordinates in a stationary image to coordinates in a moving image 
    and returns an affine that for use on global model coordinates, given
    that the stationary image is located at the provided TOPLEFT position.
    That is, if LAFM acts by 
      p_moving = LAFM p_stationary, 
    then AFM acts by
      p_moving = AFM p_model = LAFM (p_model - topleft).'''
    dp = applyAffine(afm, topleft) - applyAffine(afm, [0,0])
    return shiftAffine(afm, -dp)


def loadImage(ifn, stretch=False):
    '''LOADIMAGE - Load an image for alignment work
    img = LOADIMAGE(ifn) loads the named image, which can then serve as
    either the “stationary” or the “moving” image.
    Images are always converted to 8 bits. Optional second argument STRETCH 
    enables contrast stretching. STRETCH may be given as a percentage,
    or simply as True, which implies 0.1%.
    The current implementation can only read from the local file system.
    Backends for http, DVID, etc., would be a useful extension.'''
    if type(stretch)==bool and stretch:
        stretch = 0.1
    img = cv2.imread(ifn, cv2.IMREAD_ANYDEPTH + cv2.IMREAD_GRAYSCALE)
    if stretch:
        N = img.size
        ilo = int(.01*stretch*N)
        ihi = int((1-.01*stretch)*N)
        vlo = np.partition(img.reshape(N,), ilo)[ilo]
        vhi = np.partition(img.reshape(N,), ihi)[ihi]
        nrm = np.array([255.999/(vhi-vlo)], dtype=np.float32)
        img[img<vlo] = vlo
        img = ((img-vlo) * nrm).astype(np.uint8)
    else:
        if img.dtype==np.uint16:
            img = (img/256).astype(np.uint8)
        elif img.dtype==np.uint32:
            img = (img/(256*256*256)).astype(np.uint8)
        elif img.dtype==np.float or img.dtype==np.float32:
            img = (255.999*img).astype(np.uint8)
    return img


def saveImage(img, ofn, qual=None):
    '''SAVEIMAGE - Save an image
    SAVEIMAGE(img, ofn) saves the image IMG to the file named OFN.
    Optional third argument specifies jpeg quality as a number between
    0 and 100, and must only be given if OFN ends in ".jpg".'''
    if qual is None:
        cv2.imwrite(ofn, img)
    else:
        cv2.imwrite(ofn, img, (cv2.IMWRITE_JPEG_QUALITY, qual))


def extractStraightWindow(img, xy=None, siz=512, border=None):
    '''EXTRACTSTRAIGHTWINDOW - Extract a window from an image
    win = EXTRACTSTRAIGHTWINDOW(img, xy, siz) extracts a window of
    size SIZ, centered on XY, from the given image. XY do not have to
    be integer-valued.
    SIZ may be a scalar or a pair of numbers (width, height).
    SIZ defaults to 512.
    If XY is not given, the window is taken from the center of the image.
    It is OK if the window does not fit fully inside of the image. In
    that case, undefined pixels are given the value of the optional
    BORDER argument, or of the average of the image if not specified.'''    
    siz = si_unpackSize(siz)
    Y,X = img.shape
    if xy is None:
        xy = [X/2, Y/2]
    x0 = xy[0] - siz[0]/2
    y0 = xy[1] - siz[1]/2
    x1 = xy[0] + siz[0]/2
    y1 = xy[1] + siz[1]/2
    if x0 > 0 and x1 < X and y0 > 0 and y1 < Y:
        # Fully internal
        return cv2.getRectSubPix(img, siz, (xy[0]-.5, xy[1]-.5))
    else:
        # Overhanging some edge
        x0 = int(x0)
        if x0<0:
            x0 = 0
        x1 = int(x1)
        if x1 > X:
            x1 = X
        y0 = int(y0)
        if y0<0:
            y0 = 0
        y1 = int(y1)
        if y1 > Y:
            y1 = Y
        if border is None:
            border = int(np.mean(img[y0:y1,x0:x1]))
        dx = xy[0] - siz[0]/2
        dy = xy[1] - siz[1]/2
        afm = np.array([[1,0,dx],[0,1,dy]])
        return cv2.warpAffine(img, afm, siz,
                              flags=cv2.INTER_LINEAR
                              + cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=border)


def extractROI(img, rect):
    '''EXTRACTROI - Extract rectangular window from an image
    win = EXTRACTROI(img, (x0,y0,w,h)) extracts a rectangular window
    from an image.
    X0, Y0, W, and H must be integers. ROIs must fit inside the image.
    See also EXTRACTSTRAIGHTWINDOW.'''
    x0 = rect[0]
    y0 = rect[1]
    x1 = x0 + rect[2]
    y1 = y0 + rect[3]
    return img[y0:y1, x0:x1]


def extractTransformedWindow(img, xy=None, tfm=None, siz=512):
    '''EXTRACTTRANSFORMEDWINDOW - Extract a window with transform
    win = EXTRACTTRANSFORMEDWINDOW(img, xy, tfm, siz) extracts a window of
    size SIZ, centered on XY, from the given image.
    TFM must be a 2x2 transformation matrix. It is internally modified with
    a translation T such that the center point XY is not moved by the 
    combination of translation and transformation.
    SIZ may be a scalar or a pair of numbers (width, height). 
    SIZ defaults to 512.
    If TFM is not given, an identity matrix is used.
    If XY is not given, the window is taken from the center of the image.
    To transform an entire image, AFFINEIMAGE may be easier to use.'''
    siz = si_unpackSize(siz)
    if xy is None:
        xy = [img.shape[1]//2, img.shape[0]//2]
    if tfm is None:
        tfm = np.array([[1., 0],
                        [0, 1.]])
    elif type(tfm)==tuple or type(tfm)==list:
        tfm = np.array([[tfm[0], tfm[1]],
                        [tfm[2], tfm[3]]])
    xy = np.array([xy[0], xy[1]])
    tsiz = np.matmul(tfm, np.array([siz[0], siz[1]])/2)
    dxy = xy - tsiz
    afm = np.hstack((tfm, np.reshape(dxy, (2,1))))
    return cv2.warpAffine(img, afm, siz,
                          flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR)


def affineImage(afm, img, rect=None):
    '''AFFINEIMAGE - Apply an affine transformation to an image
    res = AFFINEIMAGE(afm, img) returns an image the same size of IMG
    looking up pixels in the original using affine transformation.
    res = AFFINEIMAGE(afm, img, rect), where rect is an (x0,y0,w,h)-tuple
    as from MODELBOUNDS, returns the given rectangle of model space.
    Example: if AFM is the result of MIRAFFINE(pstat, pmov), where PSTAT
      and PMOV are corresponding points on a stationary image ISTAT and a
      moving image IMOV, then AFFINEIMAGE(afm, imov) overlays the moving
      image on top of the stationary image.'''
    if rect is None:
        return cv2.warpAffine(img,
                              afm,
                              (img.shape[1],img.shape[0]),
                              flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR)
    else:
        p1 = applyAffine(afm, (0,0))
        p2 = applyAffine(afm, (rect[0], rect[1]))
        return cv2.warpAffine(img,
                              shiftAffine(afm, p2-p1),
                              (rect[2], rect[3]),
                              flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR)
    

def modelBounds(afm, img):
    '''MODELBOUNDS - Returns image bounding rectangle in model space
    (x0, y0, w, h) = MODELBOUNDS(afm, img) returns the bounding
    rectangle of the moving image IMG in model space if pixel lookup
    is through affine transform AFM.'''
    inv = invertAffine(afm)
    p0 = np.floor(applyAffine(inv, [0,0])).astype('int32')
    p1 = np.ceil(applyAffine(inv, img.shape)).astype('int32')
    return (p0[0], p0[1], p1[0], p1[1])


def apodize(img, gray=None):
    '''APODIZE - Multiplies a windowing function into an image
    APODIZE(img) multiplies the outer 1/4 of the image with a cosine 
    fadeout to gray (defined as the mean of the image, unless specifically
    given as optional argument). 
    Alignment improves drastically if at least one of the images is
    apodized. Often it is not necessary to apodize both.
    This function is not threadsafe, as it stores the most recent
    apodization kernel for reuse.'''
    if gray is None:
        gray = np.mean(img)
    if type(apodize.apo)==np.ndarray and np.all(img.shape==apodize.apo.shape):
        pass        # Use old apodization window
    else:
        nx = img.shape[1]
        ny = img.shape[0]
        xx = np.linspace(-1, 1, nx)
        vv = np.ones((nx,), dtype=np.float32)
        idx = np.abs(xx)>.5
        vv[idx] = .5 - .5*np.cos(2*np.pi*xx[idx])
        if ny==nx:
            ww = vv
        else:
            yy = np.linspace(-1, 1, ny)
            ww = np.ones((ny,), dtype=np.float32)
            idx = np.abs(yy)>.5
            ww[idx] = .5 - .5*np.cos(2*np.pi*yy[idx])
        apodize.apo = ww.reshape(ny, 1) * vv.reshape(1, nx)
        apodize.apo0 = 1 - apodize.apo
    return apodize.apo * img + apodize.apo0 * gray
apodize.apo = None
apodize.apo0 = None


def fft(img):
    '''FFT - Discrete Fourier Transform
    FFT(img) returns the two-dimensional DFT of a real YxX image as an YxXx2
    tensor where real and imaginary parts follow each other.'''
    return cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)


def ifft(img):
    '''IFFT - Discrete Fourier Transform
    IFFT(spec) returns the two-dimensional inverse DFT of a (complex) YxXx2
    spectrum as a real YxX image. (Any complex part of the result is dropped.)
    The input must be an YxXx2 tensor as produced by FFT.'''
    return cv2.idft(img, flags=cv2.DFT_REAL_OUTPUT)


def _alignmentImage(sta, mov, wht=-.65):
    '''_ALIGNMENTIMAGE - Calculate SWIM correlogram between two image tiles
    img = _ALIGNMENTIMAGE(sta, mov) calculates the SWIM correlogram
    pair of image tiles. Tiles are typically cut from a larger image 
    using EXTRACTSTRAIGHTWINDOW or EXTRACTTRANSFORMEDWINDOW and one of 
    them should be apodized first.
    This function will apply Fourier transforms to both of the images. 
    If you are going to use it on the same image more than once, you can 
    precompute the transform using SWIFTIR's FFT function. This function
    automatically recognizes that the Fourier transform has already 
    been applied.
    Optional third argument WHT can specify whitening exponent. It
    defaults to -0.65.'''
    if len(sta.shape)<3:
        sta = fft(sta - np.mean(sta))
    if len(mov.shape)<3:
        mov = fft(mov - np.mean(mov))
    # Following calculates sta * mov.conj()
    prd = np.stack((sta[:,:,0]*mov[:,:,0]+sta[:,:,1]*mov[:,:,1],
                    sta[:,:,1]*mov[:,:,0]-sta[:,:,0]*mov[:,:,1]), 2)
    # Following calculates ||prd||^2
    pw = prd[:,:,0]*prd[:,:,0] + prd[:,:,1]*prd[:,:,1] + 1e-40
    prd[:,:,0] = prd[:,:,0] * pw**(wht/2)
    prd[:,:,1] = prd[:,:,1] * pw**(wht/2)
    shf = ifft(prd)
    return shf


def _findPeak(img):
    '''_FINDPEAK - Finds the location of a peak in an image
    (x,y) = _FINDPEAK(img) finds the x- and y-coordinates of the
    highest peak in an image. This function folds x coordinates in the
    right half of the image to be negative. Same for y coordinates in
    the bottom half.'''
    idx = np.argmax(img)
    S = img.shape
    xmax = idx % S[1]
    ymax = idx // S[1]
    if xmax >= S[1]/2:
        xmax -= S[1]
    if ymax >= S[0]/2:
        ymax -= S[0]
    return xmax, ymax


def _extractFoldedAroundPoint(img, xy, rad=5):
    '''_EXTRACTFOLDEDAROUNDPOINT - Extracts area around a point
    win = _EXTRACTFOLDEDAROUNDPOINT(img, xy) extracts a small area from 
    image IMG around the point XY. The image is treated as a torus, i.e.,
    for a 100x100 image, position x = -5, 95, and 195 are all the same.
    Optional third argument RAD specifies radius of the extracted area.
    Full size of extracted area is (2 RAD + 1) x (2 RAD + 1).'''
    n = 2*rad + 1
    xx = range(n) + xy[0] - rad
    yy = range(n) + xy[1] - rad
    # Fold to nonnegative:
    S = img.shape
    xx = xx % S[1]
    yy = yy % S[0]
    xx = np.repeat(np.reshape(np.array(xx), (1,n)), n, 0)
    yy = np.repeat(np.reshape(np.array(yy), (n,1)), n, 1)
    return img[yy,xx]


def _centerOfMass(img):
    '''_CENTEROFMASS - Calculates the center of mass in an image
    (x, y, vx, vy) = _CENTEROFMASS(img) calculates the center of mass (x,y)
    in the image IMG. The very center of the image is (0,0).
    _CENTEROFFMASS also returns the shifted second moments of x and y.'''
    img = img.copy()
    Y,X = img.shape
    xx = np.array(range(X))
    yy = np.array(range(Y))
    xx -= (X-1)//2
    yy -= (Y-1)//2
    xx = np.repeat(np.reshape(xx, (1,X)), Y, 0)
    yy = np.repeat(np.reshape(yy, (Y,1)), X, 1)
    img[img<0] = 0
    sumw = np.sum(img)
    sumx = np.sum(img*xx)
    sumy = np.sum(img*yy)
    x = sumx / (sumw + 1e-40)
    y = sumy / (sumw + 1e-40)
    xx = xx - sumx / (sumw + 1e-40)
    yy = yy - sumy / (sumw + 1e-40)
    sumxx = np.sum(img*xx*xx)
    sumyy = np.sum(img*yy*yy)
    vx = sumxx / (sumw + 1e-40)
    vy = sumyy / (sumw + 1e-40)
    return (x, y, vx, vy)


def swim(sta, mov, wht=-.65, rad=5):
    '''SWIM - Calculate alignment between two image tiles
    (dx, dy, sx, sy, snr) = SWIM(sta, mov) calculates the optimal
    shift (dx,dy) between a pair of image tiles. Tiles are typically
    cut from a larger image using EXTRACTSTRAIGHTWINDOW or
    EXTRACTTRANSFORMEDWINDOW and one of them should be apodized first.
    SWIM will apply Fourier transforms to both of the images. If you
    are going to run SWIM on the same image more than once, you can 
    precompute the transform using SWIFTIR's FFT function. SWIM 
    automatically recognizes that the Fourier transform has already 
    been applied.
    Optional third argument WHT can specify whitening exponent. It
    defaults to -0.65.
    DX is positive if common features in image MOV are found to the
    right of the same features in image STA.
    DY is positive if common features in image MOV are found below
    the same features in image STA.
    SWIM also returns the estimated width of the peak (SX,SY) and the
    overall signal-to-noise ratio SNR in the shift image.'''
    shf = _alignmentImage(sta, mov, wht)
    xy = _findPeak(shf)
    win = _extractFoldedAroundPoint(shf, xy, rad)
    cm = _centerOfMass(win - win.mean())
    dx = xy[0] + cm[0]
    dy = xy[1] + cm[1]
    sx = np.sqrt(cm[2])
    sy = np.sqrt(cm[3])
    snr = (shf[xy[1],xy[0]] - np.mean(shf)) / (np.std(shf) + 1e-40)
    return -dx, -dy, sx, sy, snr



def stationaryPatches(img, pp, siz=512):
    '''STATIONARYPATCHES - Extract several patches from a stationary image
    fff = STATIONARYPATCHES(img, pp, siz) extracts several patches from a
    stationary image IMG. Coordinates PP must be given as a 2xN array and
    specify the centers of the patches.
    Patch size SIZ is given as (W,H) or simply W and defaults to 512.
    The result is a list of arrays that are apodized and Fourier
    transformed so they can be fed directly to MULTISWIM.'''
    n = pp.shape[1]
    res = []
    for k in range(n):
        res.append(fft(apodize(extractStraightWindow(img, pp[:,k], siz))))
    return res


def movingPatches(img, pp, afm, siz=512):
    '''MOVINGPATCHES - Extract several patches from a moving image
    fff = MOVINGPATCHES(img, pp, afm, siz) extracts several patches
    from a moving image IMG. Coordinates PP are specified as a 2xN array
    and specify the centers of the patches in coordinates of the moving 
    image. That is, AFM is _not_ applied to PP.
    Use pp=STATIONARYTOMOVING(pstat) if you prefer to specify world
    coordinates.
    Patch size SIZ is given as (W,H) or simply W and defaults to 512.
    AFM is an affine transform that is applied to the extracted
    image. Note that any translation specified in the AFM is ignored.
    The result is a list of arrays that are Fourier transformed so
    they can be fed directly to MULTISWIM. Note that moving patches
    are _not_ apodized.'''
    n = pp.shape[1]
    res = []
    tfm = afm[:,0:2]
    pp = pp.astype(np.float32)
    for k in range(n):
        res.append(fft(extractTransformedWindow(img, pp[:,k], tfm, siz)))
    return res


def multiSwim(stas, movs, wht=-.65, pp=None, afm=None):
    '''MULTISWIM - Run SWIM on multiple patches within an image
    (dp, ss, snr) = MULTISWIM(stas, movs) performs SWIM on each of
    the patches in STAS and MOVS, which should have been extracted
    from a stationary and a moving image using STATIONARYPATCHES and
    MOVINGPATCHES respectively.
    Optional third argument WHT specifies whitening exponent; default
    is -0.65.
    Result DP is to be subtracted from the points in the stationary image 
    if the points in the moving image are to be kept.
    MULTISWIM also returns SX and SY from SWIM in a 2xN array SS and
    the SNR values from SWIM in a vector.
    If optional arguments PP and AFM are included (as for
    MOVINGPATCHES), MULTISWIM will _instead_ calculalate DP as the
    shift to be _added_ to the points in the moving image if the
    points in the stationary image are to be kept.'''
    
    n = len(stas)
    dp = np.zeros((2,n))
    ss = np.zeros((2,n))
    snr = np.zeros(n)
    for k in range(n):
        (dx, dy, sx, sy, snr1) = swim(stas[k], movs[k], wht)
        dp[0,k] = dx
        dp[1,k] = dy
        ss[0,k] = sx
        ss[1,k] = sy
        snr[k] = snr1

    if afm is None or pp is None:
        return (dp, ss, snr)

    n = pp.shape[1]
    pp = pp.astype(np.float32)
    dpb = 0*pp
    for k in range(n):
        p = applyAffine(afm, pp[:,k])
        q = applyAffine(afm, pp[:,k] - dp[:,k])
        dpb[:,k] = p - q
    return (dpb, ss, snr)


def scaleImage(img, fac=2):
    '''SCALEIMAGE - Rescale an image
    im1 = SCALEIMAGE(img, fac) returns a version of the image IMG scaled
    down by the given factor in both dimensions. FAC is optional and 
    defaults to 2. FAC must be integer.
    Before scaling, the image is trimmed on the right and bottom so
    that the width and height are a multiple of FAC.
    Pixels in the output image are the average of all underlying pixels
    in the input image.'''
    S = img.shape
    K = (S[0]//fac, S[1]//fac)
    if fac*K[0] < S[0] or fac*K[1] < S[1]:
        img = img[0:fac*K[0],0:fac*K[1]]
    return cv2.resize(img, None, None, 1.0/fac, 1.0/fac, cv2.INTER_AREA)
    

def meanImage(ifns, stretch=False):
    '''MEANIMAGE - Calculate average image from multiple files
    img = MEANIMAGE(ifns) loads, one by one, all the images from the files 
    in IFNS, and returns an averaged image (as float32).
    Optional second argument STRETCH enables contrast stretching for
    the individual images; see LOADIMAGE.'''
    if len(ifns)==0:
        return np.array([], dtype=np.float32)
    sumimg = loadImage(ifns[0], stretch).asType(np.float32)
    N = len(ifns)
    for k in range(1, N):
        sumimg += loadImage(ifns[k], stretch)
    return sumimg / N


def meanImageNoBlack(ifns, stretch=False, minvalid=1):
    '''MEANIMAGENOBLACK - Average without black pixels
    img = MEANIMAGENOBLACK(ifns) is like MEANIMAGE, except that it does
    not include black pixels (value = 0) in the mean. Optional argument
    MINVALID specifies that a black result is to be return in any pixel
    that doesn't have at least that many valid (nonblack) inputs.
    MEANIMAGENOBLACK also accepts the STRETCH argument as in MEANIMAGE.'''
    if len(ifns)==0:
        return np.array([], dtype=np.float32)
    sumimg = loadImage(ifns[0], stretch).asType(np.float32)
    nimg = (sumimg>0).asType('int16')
    N = len(ifns)
    for k in range(1, N):
        img = loadImage(ifns[k], stretch)
        nimg += nimg>0
        sumimg += img
    if minvalid>1:
        sumimg[nimg<minvalid] = 0
    return sumimg / nimg


def buildout(ifns, saver, nbase=21, loader=None, aux=None, update=False):
    '''BUILDOUT - Create reference images from a stack from the center out
    BUILDOUT(ifns, saver) loads the images in IFNS starting from the 
    center of the pile, then calls SAVER to align the next image to 
    the current running average. The SAVER function is called like this:

      saver(ifn, localimg, runningavg, aux)

    and is expected to align the LOCALIMG to the RUNNINGAVG image and either
    save the resulting image or store the transformation parameters in a 
    database. The final argument AUX comes straight from the like-named 
    optional argument to BUILDOUT. 
    As a special case, SAVER is called with None as RUNNINGAVG for
    the central image of the stack.
    By default, BUILDOUT uses 21 images for its running average. This can
    be overridden using the NBASE parameter. NBASE is rounded up to the
    next odd number.
    By default, BUILDOUT uses LOADIMAGE to load the files. This can be
    overridden using the LOADER parameter which must be a function that takes
    an element of IFNS and produces the corresponding image. In this scenario,
    IFNS does not necessarily have to contain filenames: Anything else that
    your LOADER function understands is fine too.
    Optional argument UPDATE specifies that the aligned images rather than
    the original images should be incorporated into the running average. In
    that case (only) your SAVER function must return that aligned image.
    '''

    S = len(ifns)
    if loader is None:
        loader = loadImage
    s0 = S//2
    halfn = (nbase+1)//2
    nbase = 2*halfn + 1
    img = loader(ifns[s0]).astype(np.float32)
    x = saver(ifns[s0], img, None, aux)
    if update:
        img = x

    # First, build up the central stack
    #print('buildup central')
    basestack = [img]
    basesum = img.copy()
    for ds in range(1,halfn+1):
        # One image before s0:
        sa = s0 - ds
        if sa>=0:
            imga = loader(ifns[sa]).astype(np.float32)
            x = saver(ifns[sa], imga, basesum / len(basestack), aux)
            if update:
                img = x
        # One image beyond s0:
        sb = s0 + ds
        if sb<S:
            imgb = loader(ifns[sb]).astype(np.float32)
            x = saver(ifns[sb], imgb, basesum / len(basestack), aux)
            if update:
                img = x
        # Add them to the stack on either end
        if sa>=0:
            basestack.insert(0, imga)
            basesum += imga
        if sb<S:
            basestack.append(imgb)
            basesum += imgb
    # Now our base stack contains the central NBASE images, if there are
    # that many, and we have processed those images.

    # Let's work our way back to zero
    #print('to start')
    stack = basestack.copy() # This is a shallow copy!
    ssum = basesum.copy()
    for s in range(s0-halfn-1, -1, -1):
        if len(stack)>nbase:
            ssum -= stack.pop()
        img = loader(ifns[s]).astype(np.float32)
        x = saver(ifns[s], img, ssum / len(stack), aux)
        if update:
            img = x
        ssum += img
        stack.insert(0, img)
    
    # Let's work our way out to the end
    #print('to end')
    stack = basestack
    ssum = basesum
    for s in range(s0+halfn+1, S):
        if len(stack)>nbase:
            ssum -= stack.pop(0)
        img = loader(ifns[s]).astype(np.float32)
        x = saver(ifns[s], img, ssum / len(stack), aux)
        if update:
            img = x
        ssum += img
        stack.append(img)


def remod(ifns, ofnbase=None, halfwidth=10, halfexclwidth=0, topbot=False,
          loader=None, saver=None):
    '''REMOD - Create reference images from a stack
    REMOD(ifns, ofnbase) loads all the images in IFNS (not all at once)
    and produces output images that are a boxcar average of nearby images
    with the central image excluded.
    Optional third argument HALFWIDTH specifies the width of the box car
    to be 2*HALFWIDTH+1. HALFWIDTH defaults to 10.
    Optional fourth argument HALFEXCLWIDTH specifies the number of
    excluded images as 2*HALFEXCLWIDTH+1. HALFEXCLWIDTH defaults to zero,
    so only one central image is excluded.
    Optional fifth argument TOPBOT specifies that images near the top and
    bottom should be calculated, even though they do not have enough
    neighbors to calculate the average over a full box car.
    OFNBASE may be a string with a single '%i' or equivalent in it to
    receive the image number or it may be a function that takes an integer
    and returns a string. Numbering of images is along IFNS: with
    HALFWIDTH=10 and TOPBOT=False, the first saved image has number 10.
    When used in this default way, images are loaded with LOADIMAGE and
    saved with SAVEDIMAGE.
    There is a completely different way to use this function as well:
    You can specify an alternative LOADER function. In that case,
    REMOD calls your LOADER function with the n-th element of the IFNS
    list and expect to get an image back. You can put anything in that
    IFNS list: filenames, or other identifying information that your
    LOADER can interpret.
    You can also specify an alternative SAVER function. In that case,
    your saver function is called also with (1) the produced image; (2)
    the n-th element of the IFNS list; (3) the central image of the stack.
    OFNBASE is not used in this form. 
    Note that unless TOPBOT is True, the first and the last HALFWIDTH
    images do not get produced.'''
    stack = []
    Z = 2*halfwidth
    N = len(ifns)
    if Z>N:
        Z = N
    abovesum = None
    belowsum = None
    nabove = 0
    nbelow = 0
    keepin = halfwidth - halfexclwidth

    def idx(k):
        return k % Z

    if loader is None:
        loader = loadImage

    def writeImage(k):
        img = belowsum.copy()
        if not abovesum is None:
            img += abovesum
        if nabove+nbelow:
            img /= nabove+nbelow
        if saver is None:
            if type(ofnbase)==str:
                ofn = ofnbase % k
            else:
                ofn = ofnbase(k)
            saveImage(img.astype(np.uint8), ofn)
        else:
            #print('saving', N, len(stack), k, halfwidth, idx(k))
            saver(img.astype(np.uint8), ifns[k], stack[idx(k)])

    # Perhaps confusingly, "below" means greater k; "above" means lesser k.
    for k in range(N):
        if nbelow >= keepin:
            belowsum -= stack[idx(k - keepin)]
            nbelow -= 1

        nwimg = loader(ifns[k]).astype(np.float32)

        if belowsum is None:
            belowsum = nwimg.copy()
        else:
            belowsum += nwimg
        nbelow += 1

        if k>=halfwidth:
            if k>=Z or topbot:
                writeImage(k-halfwidth)

        if nabove >= keepin:
            abovesum -= stack[idx(k-Z)]
            nabove -= 1
        if k>=halfwidth+halfexclwidth:
            if abovesum is None:
                abovesum = stack[idx(k - halfwidth - halfexclwidth)].copy()
            else:
                abovesum += stack[idx(k - halfwidth - halfexclwidth)]
            nabove += 1
   
        if k<Z:
            stack.append(nwimg)
        else:
            stack[idx(k)] = nwimg
    #print('1', abovesum, nabove)
    if topbot:
        for k in range(N, N+halfwidth):
            if nbelow>0 and k>=keepin:
                belowsum -= stack[idx(k - keepin)]
                nbelow -= 1
            if k>=halfwidth:
                #print(f'{k}, {k-halfwidth} below={belowsum}/{nbelow} above={abovesum}/{nabove}')
                writeImage(k-halfwidth)
            if nabove >= keepin:
                abovesum -= stack[idx(k - Z)]
                nabove -= 1
            #print('grab', k, halfwidth, halfexclwidth, len(stack))
            if k>=halfwidth+halfexclwidth:
                if abovesum is None:
                    abovesum = stack[idx(k - halfwidth + halfexclwidth)].copy()
                else:
                    abovesum += stack[idx(k - halfwidth + halfexclwidth)]
                nabove += 1


def copyTriangle(dst, src, tria, afm):
    '''COPYTRIANGLE - Copy a triangular area with affine transform
    dst = COPYTRIANGLE(dst, src, tria, afm) fills the area defined by
    TRIA in the DST image with pixels from the SRC image. Pixel lookup
    in the SRC image is through affine transform AFM: p_src = AFM(p_dst).
    TRIA must be a 2x3 array of integer pixel coordinates in the DST
    image. AFM must be a 2x3 array defining an affine transform.'''
    
    p_topleft = np.min(tria, 1)
    p_bottomright = np.max(tria, 1) + 1
    box_size = (p_bottomright - p_topleft)

    clip = cv2.warpAffine(floa, shiftAffine(afm, -p_topleft),
                          (box_size[0],box_size[1]))
    mask = np.zeros((box_size[1], box_size[0]), dtype=np.uint8)
    shiftria = tria - np.repeat(np.reshape(p_topleft,(2,1)),3,1)
    dst = base[p_topleft[1]:p_bottomright[1],p_topleft[0]:p_bottomright[0]]
    cv2.bitwise_and(dst, 255-mask, dst)
    cv2.bitwise_and(clip, mask, clip)
    cv2.bitwise_or(dst, clip, dst)
    return dst
    

######################################################################

if __name__=='__main__':

    import time
    import matplotlib.pyplot as plt
    
    def simpleTest(img, im2):
        cut1 = extractStraightWindow(img)
        cut2 = extractStraightWindow(im2)
        apo1 = apodize(cut1)
        ff1 = fft(apo1)
        plt.figure(1)
        plt.imshow(apo1)
        plt.figure(2)
        plt.imshow(cut2)
        plt.figure(3)
        plt.imshow(alignmentImage(apo1, cut2))
        dxy = swim(ff1, cut2)
        print('Point in center:', dxy)
        
    def multiTest(img, im2):
        pa = np.array([[3000, 7000, 3000, 7000],
                       [3000, 3000, 7000, 7000]])
        stas = stationaryPatches(img, pa)
        pb = pa
        movs = movingPatches(img, pb, identityAffine())
        dp, ss, snr = multiSwim(stas, movs)
    
        print('original points', pb)
        pb = pb + dp
        print('after first swim', pb)
        print('peak width', ss)
        print('snr', snr)
    
        afm, rms, iworst = mirAffine(pa, pb)
        print('affine matrix from mir', afm, rms, iworst)

        pb = stationaryToMoving(afm, pa)
        movs = movingPatches(img, pa, afm)
        print('affine places points at', pb)
        dp, ss, snr = multiSwim(stas, movs)
        pb = pb + dp
        print('after second swim', pb)
        print('peak width', ss)
        print('snr', snr)
        
        afm, rms, iworst = mirAffine(pa, pb)
        print('affine matrix from mir', afm, rms, iworst)
        
        pb = stationaryToMoving(afm, pa)
        movs = movingPatches(img, pa, afm)
        print('affine places points at', pb)
        dp, ss, snr = multiSwim(stas, movs)
        pb = pb + dp
        print('after third swim', pb)
        print('peak width', ss)
        print('snr', snr)

        afm, rms, iworst = mirAffine(pa, pb)
        print('affine matrix from mir', afm, rms, iworst)

    def testRemod():
        ifns = np.array(range(30))
        ofnbase = 'out-%i-'
        remod(ifns, ofnbase, 3, 0, True, True)

        
    t = time.time()
    print('Loading image 1')
    img = loadImage('eric-sbem-test-190519/3-day_Ganglion2/Image1.tif',.1)
    print('dt = ', time.time() - t)
          
    t = time.time()
    print('Loading image 2')
    im2 = loadImage('eric-sbem-test-190519/3-day_Ganglion2/Image2.tif',.1)
    print('dt = ', time.time() - t)

    t = time.time()
    print('Running simple test in center')
    simpleTest(img, im2)
    print('dt = ', time.time() - t)

    t = time.time()
    print('Running iterative test with 4 patches')
    multiTest(img, im2)
    print('dt = ', time.time() - t)

    
