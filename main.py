import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib import animation
from numpy.fft import fft, ifft
from PIL import Image
from scipy import signal
import pydicom
import datetime
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset


# Zwraca koordynaty pixeli znajdujących się na odcinku od (x0,y0) do (x1,y1)
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
def get_ray_value(ro, alfa, img):
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
    print(img.shape)
    alfa = np.arange(0, 180.0, d_alfa)
    ro = np.arange(-l/2, l/2, l/n)

    vals = []
    for i in alfa:
        for j in ro:
            vals.append(get_ray_value(j,i,img))
    val = list(map(int, vals))

    radon = np.array(val).reshape(int(180/d_alfa), n)

    # Sinogram
    plt.imshow( radon , cmap = 'gray')
    plt.show()

    return radon

def generate_filter(count):
    filt = [0] * count
    start = int(count / 2)
    for i in range(0, count - start): # positive indices
        num = None
        if i == 0:
            num = 1
        elif i % 2 == 0:
            num = 0
        else:
            num = (-4 / (math.pi ** 2)) / (i ** 2)
        filt[i + start] = num

    for i in range(0, start): # negative indices
        num = None
        substitute_i = start - i
        if substitute_i % 2 == 1:
            num = (-4 / (math.pi ** 2)) / (substitute_i ** 2)
        else:
            num = 0
        filt[i] = num

    return filt

    # return [1 if i == 0 else 0 if i % 2 == 0 else (-4 / (math.pi ** 2)) / (i ** 2) for i in f]

def generate_filter2(count):
    f = range(0, count)
    return [1 if i == 0 else 0 if i % 2 == 0 else (-4 / (math.pi ** 2)) / (i ** 2) for i in f]

def inv_radon(img, l, n):
    img2 = np.full((l, l, 2), 0, dtype=np.uint64)
    alfa = np.arange(0, 180, 1)
    ro = np.arange(int(-l / 2), int(l / 2), int(l / n))
    mat = np.zeros((n, n))

    for i in alfa:
        for j in ro:
            for k in get_ray_pixels(img2.shape[0], i, j):
                mat[k[0] - 1][k[1] - 1] += img[i - 1][int(l / 2) + j - 1]
    mat = mat / 180
    return mat

def filter_row(row, f): #  doesn't work if len(f) is 1
    l = len(f)
    offset_end = int((l - 1) / 2)
    offset_start = offset_end + (1 if l % 2 == 0 else 0)
    return signal.convolve(row, f)[offset_start:-offset_end]

def filter_img(img, size = 95):
    height = img.shape[0]
    width = img.shape[1]

    filtered_matrix = []
    f = generate_filter(size)

    for i in range(0, height):
        filtered_matrix.append(filter_row(img[i, :], f))

    # filtered_img = np.uint8(np.array(filtered_matrix))
    filtered_img = np.array(filtered_matrix)
    return filtered_img

def get_patient_id(name):
    return abs(hash(name)) % (10 ** 8)

def save_dicom(filename, name, image, study_description, study_date = ""):
    filename_little_endian = filename + ".dcm"

    file_meta = FileMetaDataset()
    # file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # FIXME https://stackoverflow.com/questions/14350675/create-pydicom-file-from-numpy-array

    ds = FileDataset(filename_little_endian, {},
                     file_meta=file_meta, preamble=b"\0" * 128)

    ds.Modality = 'WSD'
    ds.ContentDate = str(datetime.date.today()).replace('-', '')
    ds.ContentTime = str(time.time())  # milliseconds since the epoch
    ds.StudyInstanceUID = '1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093'
    ds.SeriesInstanceUID = '1.3.6.1.4.1.9590.100.1.1.369231118011061003403421859172643143649'
    ds.SOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    ds.SOPClassUID = 'Secondary Capture Image Storage'

    ds.PatientName = name
    ds.PatientID = str(get_patient_id(name))
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.StudyDescription = study_description
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.PixelSpacing = r"1\1"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.HighBit = 7
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"
    ds.Rows = image.shape[0]
    ds.Columns = image.shape[1]

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    ds.PixelData = image.tobytes()
    ds.SmallestImagePixelValue = str.encode("\x00\x00")
    ds.LargestImagePixelValue = str.encode("\xff\xff")

    dt = datetime.datetime.now()
    ds.StudyDate = dt.strftime('%Y%m%d')

    ds.save_as(filename_little_endian)

if __name__ == '__main__':

    n = 200
    l = 200

    img = cv2.imread("tomograf-zdjecia/Kropka.jpg", cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap="gray")
    plt.show()
    radon = radon_transform("tomograf-zdjecia/Kropka.jpg",l,n,1)

    filtered_radon = filter_img(radon, 95)
    plt.imshow(filtered_radon, cmap='gray')
    plt.show()

    mat = inv_radon(filtered_radon, l, n)
    plt.imshow(mat, cmap='gray')
    plt.show()
    save_dicom("jorji", "Jorji Costava", mat, "ligma")
    exit()
