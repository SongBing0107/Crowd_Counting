import numpy as np
import math

def Get_Density_Map_Gaussian(Image, points):
    # a zero numpy array shape like original image
    Image_density = np.zeros((Image.shape), dtype=int)
    H, W, C = Image_density.shape
    
    points_x, points_y = points.shape

    if points_x == 1:
        x1 = max(1, min(W, round(points[0, 0])))
        y1 = max(1, min(H, round(points[0, 1])))
        Image_density[y1, x1] = 255
        return Image_density

    for i in range(1, len(points) + 1):
        fsz = 15
        sigma = 4.0
        H = matlab_style_gauss2D((fsz, fsz), sigma=sigma)

        print('H.shape = {}'.format(H.shape))
        print('H = {}'.format(H))   

        x = min(W, max(1, abs(int(math.floor(points[i, 0])))))
        y = min(H, max(1, abs(int(math.floor(points[i, 1])))))
        
        if x > W or y > H:
            continue

        x1 = x - int(math.floor(fsz / 2))
        y1 = y - int(math.floor(fsz / 2))
        x2 = x + int(math.floor(fsz / 2))
        y2 = y + int(math.floor(fsz / 2))
        
        dfx1 = 0
        dfx2 = 0
        dfy1 = 0
        dfy2 = 0
        change_flag = False
        if x1 < 1:
            dfx1 = abs(x1) + 1
            x1 = 1
            change_flag = True
        if y1 < 1:
            dfy1 = abs(y1) + 1
            y1 = 1
            change_flag = True
        if x2 > W:
            dfx2 = x2 - W
            x2 = W
            change_flag = True
        if y2 > H:
            dfy2 = y2 - H
            y2 = H
            change_flag = True
        
        x1h = dfx1 + 1
        y1h = dfy1 + 1
        x2h = fsz - dfx2
        y2h = fsz - dfy2

        if change_flag is True:
            H = matlab_style_gauss2D((float(y2h - y1h + 1),
                                      float(x2h - x1h + 1)),
                                      sigma=sigma)

        Image_density[y1:y2, x1:x2] = Image_density[y1:y2, x1:x2] + H


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [float((ss - 1) / 2) for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x * x + y * y) / (float(2 * sigma * sigma)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h

