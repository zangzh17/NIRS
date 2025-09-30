#%%

import swiftir
import cv2
import numpy as np
import matplotlib.pyplot as plt
# plt.ion()

srcroot='E:/250716_mouse/m1'
ifn = 'record_16072025_154554_inj3bk_100hz_n2n_ss_3.tif'
bn = 'record_16072025_154554_inj3bk_100hz_n2n_ss'
ok, tif = cv2.imreadmulti(f"{srcroot}/{ifn}", flags=cv2.IMREAD_UNCHANGED)
tif = np.stack(tif, 0)
T, Y, X = tif.shape
print(T,Y,X)

plt.figure(1)
plt.clf()
plt.imshow(tif[50])

#%%
x0 = 108 # approx. location of our cell
y0 = 128
W0 = 150

tt0 = range(T//2-50, T//2+50)
  
img0 = tif[tt0].mean(0).astype(np.float32) # prelim. target image

# Get ROI from average image
roi0 = swiftir.extractStraightWindow(img0, [x0, y0], W0)

plt.clf()
plt.imshow(roi0)

roi0apo = swiftir.apodize(roi0)
plt.clf()
plt.imshow(roi0apo)
# Align all of those ROIs
rois = []
#ddx = []
#ddx1 = []
for t in tt0:
    img1 = tif[t].astype(np.float32)
    img1roi = swiftir.extractStraightWindow(img1, [x0, y0], W0)
    (dx, dy, sx, sy, snr) = swiftir.swim(roi0apo, img1roi)
    #ddx.append(dx)
    img1mv = swiftir.extractStraightWindow(img1, [x0+dx, y0+dy], W0)
    #(dx1, dy1, sx, sy, snr) = swiftir.swim(roi0apo, img1mv)
    #ddx1.append(dx1)
    # -- uncomment to verify that second extraction is indeed better
    rois.append(img1mv)

# Get average after prelim. alignment
roi1 = np.mean(rois, 0)
roi1apo = swiftir.apodize(roi1)

plt.clf()
plt.imshow(roi1apo)

#%%
W = X - 5
H = Y - 5

# Now align all the images
imgs = []
for s in range(6):
    ok, stack3d = cv2.imreadmulti(f"{srcroot}/{bn}_{s+1}.tif", flags=cv2.IMREAD_UNCHANGED)
    stack3d = np.stack(stack3d, 0)
    stack3dmv = []
    for t in range(T):
        img1 = tif[t].astype(np.float32)
        img1roi = swiftir.extractStraightWindow(img1, [x0, y0], W0)
        (dx, dy, sx, sy, snr) = swiftir.swim(roi1apo, img1roi)
        stack3dmv.append(swiftir.extractStraightWindow(stack3d[t].astype(np.float32), [X/2+dx, Y/2+dy], [W,H]))
    stack3dmv = np.stack(stack3dmv, 0)
    out = [img.astype(np.uint16) for img in stack3dmv]
    cv2.imwritemulti(f"{srcroot}/{bn}_demotion_{s+1}.tif", out)


# %%
print(f"{srcroot}/{bn}_{s+1}.tif")
# %%
