import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib import animation

# Zwraca koordynaty pixeli znajdującej się na odcinku od (x0,y0) do (x1,y1)
def bresenham(x0, y0, x1, y1,img_size):
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        if(x0 + x*xx + y*yx <= img_size and x0 + x*xx + y*yx >= 0 and y0 + x*xy + y*yy <= img_size and y0 + x*xy + y*yy >= 0):
            yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

# Znajduje koordynaty końca i początku promienia (alfa - kąt nastawienia promienia do poziomu, ro - odległość od środka obrazka)
def find_edges(ro,r,alfa,img_size):
    if(ro == 0):
        centerX = img_size/2
        centerY = img_size/2
    else:
        if(alfa % 360 <= 90):
            offsetX = -math.sin(math.pi * (alfa)/180)*ro
            offsetY = math.cos(math.pi * (alfa)/180)*ro
        elif(alfa % 360 <= 180):
            offsetX = math.sin(math.pi * (180 + alfa) / 180) * ro
            offsetY = -math.cos(math.pi * (180 + alfa) / 180) * ro
        centerX = img_size/2 + offsetX
        centerY = img_size/2 + offsetY
    x0 = centerX + r * math.cos(math.pi * alfa/180)
    y0 = centerY + r * math.sin(math.pi * alfa/180)
    x1 = centerX + r * math.cos(math.pi * (alfa+180)/ 180)
    y1 = centerY + r * math.sin(math.pi * (alfa+180)/ 180)
    return x0,y0,x1,y1

# Zwraca pixele przez które przechodzi promień
def get_ray_pixels(img_size,alfa,ro):
    pixel_array = []
    x0, y0, x1, y1 = find_edges(ro, math.sqrt(2) * img_size / 2, alfa, img_size)
    for i in bresenham(int(x0), int(y0), int(x1), int(y1), img_size):
        pixel_array.append(i)

    return pixel_array

# Zwraca średnią wartość (0-255) pixeli w promieniu
def get_ray_value(ro, alfa, img_shape):
    img_size = img.shape[0]
    ray_pixels = get_ray_pixels(img_size-1,alfa,ro)

    sum = 0
    for i in ray_pixels:
        sum += img[i[0]][i[1]]
    if (len(ray_pixels) == 0):
        avg = 0
    else:
        avg = int(sum / len(ray_pixels))
    return avg

# Wykonuje transformatę Radona dla obrazka, n - liczba emiterów, l - rozpiętość układu emiterów/detektorów (w pixelach), d_alfa - krok układu emiter/detektor
def radon_transform(img_src,n,l, d_alfa):
    img = cv2.imread(img_src,cv2.IMREAD_GRAYSCALE)

    img_size = img.shape[0]

    alfa = np.arange(0, 180.0, d_alfa)
    ro = np.arange(-l/2, l/2, l/n)

    vals = []
    for i in alfa:
        for j in ro:
            vals.append(get_ray_value(j,i,img_size))
    val = list(map(int, vals))

    radon = np.array(val).reshape(int(180/d_alfa), n)

    # Sinogram
    plt.imshow( radon , cmap = 'gray')
    plt.show()

    # Zwraca
    return radon

if __name__ == '__main__':

    img = cv2.imread('tomograf-zdjecia/Kwadraty2.jpg',cv2.IMREAD_GRAYSCALE)
    radon = radon_transform('tomograf-zdjecia/Kwadraty2.jpg',200,200,1)


