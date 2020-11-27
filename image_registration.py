import SimpleITK as sitk
import os
import numpy as np
import nibabel as nib

from skimage import io

import utils.general as gen
import matplotlib.pyplot as plt

# data_dir = gen.create_and_verify(
#     "data", "Maria Rodriguez",
#     "images", list_=True)
#
# adc_dir = data_dir[0]
# dwi_dir = data_dir[1]
# t2_dir = data_dir[2]
#
# adc_img = nib.load(adc_dir).get_fdata()
# dwi_img = nib.load(dwi_dir).get_fdata()
# t2_img = nib.load(t2_dir).get_fdata()
#
# img0 = dwi_img[..., 24]
# img1 = t2_img[..., 19]

def resampler_method(fixed, moving, transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(transform)

    out = resampler.Execute(moving)
    pixelType = sitk.sitkUInt8

    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), pixelType)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), pixelType)

    cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)

    cimg_array = sitk.GetArrayFromImage(cimg)
    return cimg_array

def registration_1(img0, img1):
    pixelType = sitk.sitkFloat32
    fixed = sitk.Cast(sitk.GetImageFromArray(img0), pixelType)
    moving = sitk.Cast(sitk.GetImageFromArray(img1), pixelType)

    # We Create the Registration Object
    R = sitk.ImageRegistrationMethod()

    # We Create an Initial Transform
    # Then, We use a Similarity Transform
    # It uses rotation in radians, isotropic translation
    # with fixed center and translation
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Similarity2DTransform()
    )
    R.SetInitialTransform(initial_transform)

    # We set the similarity metric
    # Uses the mutual information between two images
    # using the method of Mattes et al.
    R.SetMetricAsMattesMutualInformation()

    # We set the optimizer
    R.SetOptimizerAsGradientDescent(
        learningRate=0.5, 
        numberOfIterations=1000
    )

    # We set the interpolator
    R.SetInterpolator(sitk.sitkLinear)

    # We Excute the method
    final_transform = R.Execute(fixed, moving)

    cimg_array = resampler_method(fixed, moving, final_transform)

    return cimg_array

# cimg_array = registration_1(img0, img1)
#
# gen.visualize_subplot(
#     [img0, img1, cimg_array], ["DWI", "T2", "Registered"],
#     (1, 3), (18, 6)
# )
#
# plt.show()

# img0 = dwi_img[..., 10]
# img1 = t2_img[..., 5]

# gen.visualize_subplot(
#     [img0, img1], ["DWI", "T2"],
#     (1, 2), (12, 6)
# )

# plt.show()