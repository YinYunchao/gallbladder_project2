import os
import random 
import SimpleITK as sitk
import torch
import numpy as np
import matplotlib.pyplot as plt
from image_augmentation import image_augmentation

class ImageLoadPipe(object):
    """
    This is a standard pytorch image loader pipeline, check documentation on pytorch website
    three parameters could decide format of returned images:
    if_transform: True, add augmentation to the img; False, no augmentation
    augmentation_list: valid when if_transform=True, give the required augmentations

    scan_list: consist of img_id and label [['GBXXX',0/1],....]. need to return tensor of img and label for classification tast
                if segmentation task make scan_list [['GBXXX'],['GBXXX']......]
    if_pad: True, pad the output images to the same size; Flase, return the original unpadded img
    pad_size: valid when if_pad=True, gives the required output img size

    if_part_slice: True, return only part of the slices, it's used when the 3D img is too large; False, return the full 3D image
    slice_num: valid when if_part_slice=True, gives how many slices does the output img need to be

    if_return_mask: True, return label arr; False, crop the labeled regrion on img arr and return img arr 
    """
    def __init__(self, scan_path, scan_list, mask_path, if_pad=False, pad_size = [500,500,82],
                if_transform=None, augmentation_list = None, if_part_slices=False, slice_num=None,
                if_return_mask = False):
        self.scan_path = scan_path
        self.mask_path = mask_path
        self.scan_list = scan_list

        self.if_transform = if_transform
        self.transform = image_augmentation
        self.augmentation_list = augmentation_list

        self.if_pad = if_pad
        self.pad_size = pad_size

        self.if_part_slices = if_part_slices
        self.slice_num = slice_num

        self.if_return_mask = if_return_mask


    def __len__(self):
        return len(self.scan_list)

    def __getitem__(self,idx):
        # print(self.scan_list[idx,0])
        img_arr, mask_arr = self.read_nii(os.path.join(self.scan_path, str(self.scan_list[idx,0]+'_CT.nii')),
                                          os.path.join(self.mask_path, str(self.scan_list[idx,0]+'_CT.nii')))
        # print(img_arr.shape)
        # mask_arr = self.read_nii(os.path.join(self.mask_path, self.scan_list[idx]),if_mask=True)
        if self.if_transform:
            obj = self.transform(img_arr, mask_arr, self.augmentation_list)
            img_arr, mask_arr = obj.Execute()
            
        if self.if_part_slices:
            slice_ind = random.randint(0,img_arr.shape[2]-self.slice_num)
            img_arr = img_arr[:,:,slice_ind:slice_ind+self.slice_num]
            mask_arr = mask_arr[:,:,slice_ind:slice_ind+self.slice_num]

        if self.if_pad:
            img_arr,mask_arr = self.padding(img_arr,mask_arr)
        if self.if_return_mask:
            img_arr = np.expand_dims(img_arr, axis = 0)
            mask_arr = np.expand_dims(mask_arr, axis = 0)
            img_tensor = torch.from_numpy(img_arr)
            mask_tensor = torch.from_numpy(mask_arr)
            return img_tensor, mask_tensor
        else:
            img_arr = img_arr * mask_arr
            img_arr = np.expand_dims(img_arr, axis = 0)
            img_tensor = torch.from_numpy(img_arr)
            
            # label = np.eye(2)[self.scan_list[idx,1]]
            # label_tensor = torch.from_numpy(np.expand_dims(label,axis=0))
            label_tensor = torch.from_numpy(np.array(self.scan_list[idx,1]))
            label_tensor = label_tensor.type(torch.LongTensor)
            return img_tensor, label_tensor


    def read_nii(self, img_path, mask_path):
        """
        This function read image from nifti file and return it as torch tensor
        """
        img_itk = sitk.ReadImage(img_path)
        img_arr = sitk.GetArrayFromImage(img_itk).transpose([1,2,0])
        img_arr = img_arr.astype(np.float32)

        mask_itk = sitk.ReadImage(mask_path)
        mask_arr = sitk.GetArrayFromImage(mask_itk).transpose([1,2,0])
        mask_arr[mask_arr[:, :, :] > 0] = 1.0
        mask_arr = mask_arr.astype(np.float32)
        return img_arr,mask_arr

    def resample(self, img_itk, new_spacing, interpolator=sitk.sitkNearestNeighbor):
        '''
        a func to resample the image and change the resolution
        '''
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetInterpolator(interpolator)
        resample_filter.SetOutputDirection(img_itk.GetDirection())
        resample_filter.SetOutputOrigin(img_itk.GetOrigin())
        resample_filter.SetOutputSpacing(new_spacing)
        new_size = np.ceil(np.array(img_itk.GetSize()) * np.array(img_itk.GetSpacing()) / new_spacing)
        resample_filter.SetSize([int(new_size[0]), int(new_size[1]), int(new_size[2])])
        resampled_img = resample_filter.Execute(img_itk)
        return resampled_img

    def padding(self,img_arr,label_arr):
        shape = np.array(img_arr.shape)
        pad_num = (self.pad_size-shape)/2
        pad_L = np.floor(pad_num).astype(int)
        pad_R = np.ceil(pad_num).astype(int)
        padded_img = np.pad(img_arr,((pad_L[0],pad_R[0]),
                                     (pad_L[1],pad_R[1]),
                                     (pad_L[2],pad_R[2])))
        padded_label = np.pad(label_arr,((pad_L[0],pad_R[0]),
                                     (pad_L[1],pad_R[1]),
                                     (pad_L[2],pad_R[2])))
        return padded_img,padded_label


    def img_display(self, img, mask):
        '''
        this func display the img and mask, the middle slice as well
        '''
        figure = plt.figure(figsize = (1,2))
        img_sliceid = int(img.shape[-1]/2)
        figure.add_subplot(1,2,1)
        plt.title('scan')
        plt.axis('off')
        plt.imshow(img[0, 0, :, :, img_sliceid], cmap = 'gray')
        figure.add_subplot(1, 2, 2)
        plt.title('mask')
        plt.axis('off')
        plt.imshow(mask[0, 0, :, :, img_sliceid], cmap='gray')
        plt.show()
    





