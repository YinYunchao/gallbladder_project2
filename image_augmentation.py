import numpy as np
import random
import matplotlib.pyplot as plt
from skimage import exposure, img_as_float
from skimage.filters import gaussian
# from skimage.segmentation import find_boundaries
from scipy.ndimage import rotate, gaussian_filter, map_coordinates

class image_augmentation:
    '''
    The class is used to augment images randomly before model training.

    param:
    augmentation_list: the list including the augmentations going to be used, choices including
                        ['rotate','contrast','gaussianNoise','GaussianBlur']
    img_arr: a 3d image array
    label_arr: the mask of 3d scan, if morphological changes, the label will be changed together with img_arr
    **each augmentation has a parameter execute_prob, indicating the probability of the aug being executed,
      can be changed in each aug func

    '''
    def __init__(self, img_arr, label_arr, augmentation_list):
        self.img_arr = img_arr
        self.label_arr = label_arr
        self.augmentation_list = augmentation_list

    def RandomRotate(self, angle_spectrum = 30, mode = 'constant', order = 0, execute_prob = 0.5):
        '''
        this function rotate the image around its center to a random angle within range of +-angle_spectrum
        the padding mode can be [edge, linear_ramp, maximum, mean, median, minimum, reflect, symmetric] etc, based on np.pad
        the order is the order of interpolation, range form 0-5, based on skimage.transform.wrap
        the returned image will be same size as input because of reshape = False, didn't provide as param in func
        '''
        if random.randint(0,100)<execute_prob*100:
            rotate_angle = random.randint(0,angle_spectrum*2)-angle_spectrum
            #print(rotate_angle)
            self.img_arr = rotate(self.img_arr, rotate_angle, mode = mode,
                                  order=order, reshape=False)
            self.label_arr = rotate(self.label_arr, rotate_angle, mode = mode,
                                  order=order, reshape=False)

    def ElasticDeformation(self, spline_order=3, alpha=2000, sigma=50, apply_3d=True, 
                            execute_prob = 1.0):
        '''
        Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order (or CZYX if the data is 4D).
        Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62

        param:
        spline_order: the order of spline interpolation (use 0 for labeled images)
        alpha: scaling factor for deformations
        sigma: smoothing factor for Gaussian filter
        apply_3d: if True apply deformations in each axis
        '''
        if random.randint(0,100)<execute_prob*100:
            volume_shape = self.img_arr.shape
            if apply_3d:
                dz = gaussian_filter(np.random.randn(volume_shape),sigma,mode="reflect")*alpha
            else:
                dz = np.zeros_like(self.img_arr)
            dy,dx = [gaussian_filter(np.random.randn(volume_shape),sigma,mode='reflect')*alpha for _ in range(2)]
            z_dim,y_dim,x_dim = volume_shape
            z,y,x = np.meshgrid(np.arange(z_dim),np.arange(y_dim),np.arange(x_dim),indexing='ij')
            indices = z+dz, y+dy, x+dx
            self.img_arr = map_coordinates(self.img_arr, indices, order=spline_order, mode='reflect')


    def RandomContrast(self, gain = [0.9,1.1], execute_prob = 0):
        '''
        Gave up, both log and gamma don't work for med img, especially after windowing
        change the contrast of image by gamma transform
        detail: skimage.exposure.adjust_gamma()
        the default random gamma ranges (0.5,1.7), checked visually
        '''
        print('in contrast')
        if random.randint(0,100)<execute_prob*100:
            # gamma = random.uniform(gamma[0],gamma[1])
            gain = 0.75
            self.img_arr = exposure.adjust_log(self.img_arr,gain)


    def EqualHist(self,execute_prob = 0.1):
        if random.randint(0,100)<execute_prob*100:
            self.img_arr = exposure.equalize_hist(self.img_arr)



    def additiveGaussianNoise(self, scale_range = [0.0,0.04], execute_prob = 0.1):
        '''
        additive Gaussian noise to the image

        param: scale for the generated gaussian noise
        '''
        #print('in gaussian noise')
        if random.randint(0,100)<execute_prob*100:
            scale = random.uniform(scale_range[0],scale_range[1])
            #print(scale)
            shape = self.img_arr.shape
            gaussian_noise = np.random.normal(0.0,scale=scale,size=shape)
            self.img_arr = self.img_arr+gaussian_noise

    def GaussianBlur(self,sigma_range = [0.0,0.25],mode = 'nearest',execute_prob = 0.5):
        '''
        Gaussian blur to the image

        param: 
        the sigma of gaussian filter, detail on skimage.filter.gaussian, maxi can be 0.25
        mode: [reflect, constant, nearest, mirror, wrap]
        '''
        if random.randint(0,100)<execute_prob*100:
            sigma = random.uniform(sigma_range[0],sigma_range[1])
            self.img_arr = gaussian(self.img_arr,sigma,mode = mode)

        
    def default(self):
        print( 'the augmentation is not included in this class!(also check the spell)')
        
    #################################################################################################
    def Execute(self):
        '''
        call the augmentations included in the augmentation_list, aka execute
        return the augmented image
        '''
        for augmentation in self.augmentation_list:
            if augmentation=='rotate':
                self.RandomRotate()
            elif augmentation=='deformation':
                self.ElasticDeformation()
            elif augmentation=='contrast':
                self.RandomContrast()
            elif augmentation == 'gaussianNoise':
                self.additiveGaussianNoise()
            elif augmentation == 'GaussianBlur':
                self.GaussianBlur()
            elif augmentation == 'EqualHist':
                self.EqualHist()
            else:
                self.default()
        return self.img_arr, self.label_arr

    def display_img_hist(self, image, axes, bin = 256):
        '''
        display the histogram of a 3d scan, the image displayed is the middle slice of the scan

        '''
        image = img_as_float(image)
        ax_img = axes[0]
        ax_hist = axes[1]
        # Display image
        slice_ind = int(image.shape[2]/2)
        ax_img.imshow(image[:,:,slice_ind], cmap=plt.cm.gray)
        ax_img.set_axis_off()
        # Display histogram, greylevel 0 and 256 are removed,the pattern because neearest neighbour interpolation
        values, bin_edge = np.histogram(image.ravel(),bins=256)
        ax_hist.bar(x= range(1,bin-1), height = values[1:-1])
        ax_hist.set_xlabel('Pixel intensity')
        return ax_img, ax_hist
