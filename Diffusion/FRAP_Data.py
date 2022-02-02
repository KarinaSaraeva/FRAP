import os
import numpy as np
from skimage import io
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import gaussian as ski_gaussian
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image

from fit import fit_gaussian, gaussian
from cymm_fit import cymm_fit_gaussian, cymm_gaussian
from scipy import interpolate


class FRAP_Data():
    def __init__(self, name, dir_name):  # initialization
        self.name = name
        self.input_dir = dir_name
        list = os.listdir(self.input_dir)  # dir is your directory path
        N = len(list)
        self.N = N

    def cut(self, a, b, c, d, save_dir): #save_dir is where to save
        list = os.listdir(self.input_dir)  # dir is your directory path
        N = len(list)
        self.N = N
        for i in range(N):
            im = Image.open(os.path.join(self.input_dir, os.listdir(self.input_dir)[i]))
            im_crop = im.crop((a, b, c, d))
            im_crop.save("".join((save_dir, f'/{i}.png')))



    def find_ROI(self, num): #num sets the gauss width

        image1 = rgb2gray(io.imread(os.path.join(self.input_dir, os.listdir(self.input_dir)[1])))
        image0 = rgb2gray(io.imread(os.path.join(self.input_dir, os.listdir(self.input_dir)[0])))

        # plt.matshow(image1, cmap=plt.cm.gray)
        # plt.show()

        # image1[320:350, 550:590] = interpolation(image1)[320:350, 550:590]
        # image0[320:350, 550:590] = interpolation(image0)[320:350, 550:590]

        image1 = ski_gaussian(image1, sigma=5, mode='constant', cval=0.0)
        image0 = ski_gaussian(image0, sigma=5, mode='constant', cval=0.0)

        plt.matshow(image1, cmap=plt.cm.gray)
        plt.show()

        grayscale_im = (image1)/np.mean(image1[:100, :100])
        grayscale_im0 = (image0)/np.mean(image0[:100, :100])

        spot_im = grayscale_im - grayscale_im0
        # spot_im = grayscale_im - np.mean(grayscale_im[:100, :100])

        spot_im = (-1) * (spot_im - np.mean(spot_im[:100, :100]))

        plt.matshow(spot_im, cmap=plt.cm.gray)
        plt.show()

        params, cov, pcov = fit_gaussian(spot_im)
        (height, x, y, width_x, width_y) = params
        gau = gaussian(*params)(*np.indices(spot_im.shape))
        width = max(width_x, width_y)

        a1 = int(x) - int(num*width)
        a2 = int(x) + int(num*width)
        b1 = int(y) - int(num*width)
        b2 = int(y) + int(num*width)
        self.x1 = a1
        self.x2 = a2
        self.y1 = b1
        self.y2 = b2
        self.center = [x, y]
        self.width = width
        spot_im = spot_im[self.x1:self.x2, self.y1:self.y2]
        self.spot_im = spot_im

        self.gau = gau[self.x1:self.x2, self.y1:self.y2]


    def drawROI(self):
        plt.matshow(self.spot_im, cmap=plt.cm.gray)  # optional
        plt.title("ROI")
        plt.show()


    def flow(self):
        u = np.zeros((len(self.t_grid), len(self.x_grid), len(self.y_grid)))
        image0 = rgb2gray(io.imread(f'D:/04 ИАЭ/FRAP/processed/imagesDPPC/DPPC-Rhod(T42)/{0}.png'))
        # image0[320:350, 550:590] = interpolation(image0)[320:350, 550:590]  # !!! useless for good camera
        image0 = ski_gaussian(image0, sigma=5, mode='constant', cval=0.0)
        grayscale_im0 = image0/np.mean(image0[:100, :100])


        input_dir = os.path.dirname(self.input_dir)
        for i in range(self.N_tgrid):
            image1 = rgb2gray(
                io.imread(f'D:/04 ИАЭ/FRAP/processed/imagesDPPC/DPPC-Rhod(T42)/{(self.N // self.N_tgrid)*i + 1}.png'))

            # image[320:350, 550:590] = interpolation(image)[320:350, 550:590]  # !!! useless for good camera

            image1 = ski_gaussian(image1, sigma=5, mode='constant', cval=0.0)

            grayscale_im = image1 / np.mean(image0[:100, :100])

            spot_im = grayscale_im - grayscale_im0
            # spot_im = grayscale_im

            spot_im = (-1) * (spot_im - np.mean(spot_im[:100, :100]))

            # plt.matshow(image, cmap=plt.cm.gray)  # optional
            # plt.title("image")
            # plt.show()

            params, cov, pcov = fit_gaussian(spot_im)
            (height, x, y, width_x, width_y) = params

            a1 = int(x - abs(self.x2 - self.x1)/2)
            a2 = a1 + abs(self.x2 - self.x1)
            b1 = int(y - abs(self.x2 - self.x1)/2)
            b2 = b1 + abs(self.x2 - self.x1)

            # plt.matshow(image, cmap=plt.cm.gray)
            # plt.show()

            u[i][:, :] = spot_im[a1:a2, b1:b2]
            # plt.matshow(image, cmap=plt.cm.gray)  # optional
            # plt.title("image")
            # plt.show()

        self.arr = u



    def save_corrected(self):

        for i in range(len(self.t_grid)):
            filename = f"D:/04 ИАЭ/FRAP/processed/7.10/images_correctedf'/{i}.csv"
            pd.DataFrame(self.arr[i, :, :]).to_csv(filename)


    def read_corrected(self):

        u = np.zeros((len(self.t_grid), len(self.x_grid), len(self.y_grid)))
        for i in range(len(self.t_grid)):
            filename = f"D:/04 ИАЭ/FRAP/processed/7.10/images_corrected/{i}.csv"
            df = pd.read_csv(filename, index_col=0)
            u[i, :, :] = df.values
        self.arr = u

    def find_grid_corrected(self, N_tgrid):
        filename = f"D:/04 ИАЭ/FRAP/processed/7.10/images_corrected/{0}.csv"
        df = pd.read_csv(filename)
        image = df.values

        self.x1 = 0
        self.x2 = image.shape[0]

        self.y1 = 0
        self.y2 = image.shape[0]

        self.N = N_tgrid
        self.N_tgrid = N_tgrid
        self.t_grid = np.linspace(0, self.duration, N_tgrid)
        self.x_grid = np.arange(self.x1, self.x2, 1)
        self.y_grid = np.arange(self.y1, self.y2, 1)

    def find_grid(self, N_tgrid, duration):

        self.duration = duration  # seconds
        self.N_tgrid = N_tgrid
        print('the first is', 1, '.png')
        print('the last picture is:', (self.N // self.N_tgrid) * self.N_tgrid + 1, '.png')
        self.t_grid = np.linspace(0, self.duration, N_tgrid)
        self.x_grid = np.arange(self.x1, self.x2, 1)
        self.y_grid = np.arange(self.y1, self.y2, 1)



    def save_to_arr(self):
        u = np.zeros((len(self.t_grid), len(self.x_grid), len(self.y_grid)))
        input_dir = os.path.dirname(self.input_dir)
        i = 0
        for f in os.listdir(input_dir):
            if f.lower().endswith('.png') is True:
                image = io.imread(os.path.join(input_dir, f))
                grayscale_im = rgb2gray(image)
                background_im = grayscale_im[1:10, 1:10]
                spot_im = grayscale_im[self.x1:self.x2, self.y1:self.y2]
                spot_im = 1 - spot_im / np.mean(background_im)  # optional: relevant intensity
                u[i][:, :] = spot_im
                i = i + 1
        self.arr = u

    def correction(self):
        u = np.zeros((len(self.t_grid), len(self.x_grid), len(self.y_grid)))
        image0 = rgb2gray(io.imread(os.path.join(self.input_dir, os.listdir(self.input_dir)[0])))
        background_im = image0[1:100, 1:100]
        image0 = 1 - image0 / np.mean(background_im)

        input_dir = os.path.dirname(self.input_dir)
        for i in range(self.N_tgrid):
                image = rgb2gray(io.imread(os.path.join(self.input_dir, os.listdir(self.input_dir)[(self.N//self.N_tgrid)*i+1])))
                background_im = image[1:100, 1:100]
                image = 1 - image / np.mean(background_im)
                image = image - image0
                image = ski_gaussian(image, sigma=3, mode='constant', cval=0.0)
                spot_im = image[self.x1:self.x2, self.y1:self.y2]
                u[i][:, :] = spot_im
        self.arr = u

    def multi_spot_gauss(self, path): #path: where to save info
        self.peak_arr = np.zeros(self.N_tgrid)
        self.r_squared_arr = np.zeros(self.N_tgrid)
        self.x_arr = np.zeros(self.N_tgrid)
        self.y_arr = np.zeros(self.N_tgrid)
        self.width_x_arr = np.zeros(self.N_tgrid)
        self.width_y_arr = np.zeros(self.N_tgrid)
        input_dir = os.path.dirname(self.input_dir)
        self.fit_arr = np.zeros((len(self.t_grid), len(self.x_grid), len(self.y_grid)))
        ii = 0
        filenames = []

        for i in range(self.N_tgrid):
            spot_im = self.arr[i]
            params, cov, pcov = cymm_fit_gaussian(spot_im)
            fit = cymm_gaussian(*params)
            self.fit_arr[i, :, :] = fit
            # plt.contour(fit(*np.indices(spot_im.shape)), cmap=plt.cm.gray) # optional
            residuals = fit(*np.indices(spot_im.shape)) - spot_im
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((spot_im - np.mean(spot_im)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            # print(r_squared)
            (height, x, y, width) = params
            width_x = width
            width_y = width
            self.peak_arr[i] = (1 - height)
            self.r_squared_arr[i] = (r_squared)
            self.x_arr[i] = x
            self.y_arr[i] = y
            self.width_x_arr[i] = width_x
            self.width_y_arr[i] = width_y

        data = np.r_['1,2,0', self.t_grid, self.peak_arr, self.x_arr, self.y_arr, self.width_x_arr, self.width_y_arr]
        frame = pd.DataFrame(data, columns=['time', 'peak', 'x', 'y', 'width_x', 'width_y'])
        frame.to_csv(path)

    def open_data(self): #reading from directory
        pass

    def save_gif(self):
        frames = []
        for f in os.listdir(self.input_dir):
            if f.lower().endswith('.png') is True:
                new_frame = Image.open(os.path.join(self.input_dir, f))
                frames.append(new_frame)
        frames[0].save('D:/04 ИАЭ/FRAP/TimeLapseLong01.07.gif', save_all=True, append_images=frames[1:], duration=30,
                       loop=0)





def interpolation(arr):
    len_x = int((arr.shape[0]) / 50)
    len_y = int((arr.shape[1]) / 50)
    x_grid = np.linspace(0, arr.shape[0] - 1, arr.shape[0], dtype=int)
    y_grid = np.linspace(0, arr.shape[1] - 1, arr.shape[1], dtype=int)
    x_grid_new = np.linspace(0, arr.shape[0] - 1, len_x, dtype=int)
    y_grid_new = np.linspace(0, arr.shape[1] - 1, len_y, dtype=int)
    arr_new = np.zeros((len_x, len_y))

    for i in range(len_x):
        for j in range(len_y):
            arr_new[i][j] = arr[x_grid_new[i]][y_grid_new[j]]

    f = interpolate.interp2d(y_grid_new, x_grid_new, arr_new, kind='cubic')

    return f(y_grid, x_grid)
