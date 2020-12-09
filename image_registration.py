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

def command_iteration(method: sitk.ImageRegistrationMethod):
    print(f"{method.GetOptimizerIteration()} = \
    {method.GetMetricValue()} : \
    {method.GetOptimizerPosition()}")

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
    R.SetInitialTransform(initial_transform, inPlace=False)

    # We set the similarity metric
    # Uses the mutual information between two images
    # using the method of Mattes et al.
    R.SetMetricAsMattesMutualInformation()

    # We set the optimizer
    R.SetOptimizerAsGradientDescent(
        learningRate=0.5, 
        numberOfIterations=1000,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    # We set the interpolator
    R.SetInterpolator(sitk.sitkLinear)

    # We Excute the method
    final_transform = R.Execute(fixed, moving)

    cimg_array = resampler_method(fixed, moving, final_transform)

    return cimg_array

def registration_2(img0, img1):
    pixelType = sitk.sitkFloat32
    fixed = sitk.Cast(sitk.GetImageFromArray(img0), pixelType)
    moving = sitk.Cast(sitk.GetImageFromArray(img1), pixelType)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(
        minStep=1e-4, numberOfIterations=1000,
        relaxationFactor=0.5, learningRate=2.0,
        gradientMagnitudeTolerance=1e-8
    )

    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    final_transform = R.Execute(fixed, moving)

    cimg_array = resampler_method(fixed, moving, final_transform)

    return cimg_array

def registration_3(img0, img1):
    pixelType = sitk.sitkFloat32
    fixed = sitk.Cast(sitk.GetImageFromArray(img0), pixelType)
    moving = sitk.Cast(sitk.GetImageFromArray(img1), pixelType)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsJointHistogramMutualInformation()

    R.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0,
        numberOfIterations=1000,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=5
    )

    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    final_transform = R.Execute(fixed, moving)

    cimg_array = resampler_method(fixed, moving, final_transform)

    return cimg_array

def registration_4(img0, img1):
    pixelType = sitk.sitkFloat32
    fixed = sitk.Cast(sitk.GetImageFromArray(img0), pixelType)
    moving = sitk.Cast(sitk.GetImageFromArray(img1), pixelType)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(
        minStep=1e-4, numberOfIterations=500,
        relaxationFactor=0.5, learningRate=2.0,
        gradientMagnitudeTolerance=1e-8
    )
    R.SetOptimizerScalesFromIndexShift()

    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Similarity2DTransform()
    )
    R.SetInitialTransform(initial_transform)
    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    final_transform = R.Execute(fixed, moving)

    cimg_array = resampler_method(fixed, moving, final_transform)

    return cimg_array

def registration_5(img0, img1):
    pixelType = sitk.sitkFloat32
    fixed = sitk.Cast(sitk.GetImageFromArray(img0), pixelType)
    moving = sitk.Cast(sitk.GetImageFromArray(img1), pixelType)

    numberOfBins = 24
    samplingPercentage = 0.1

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfBins)
    R.SetMetricSamplingPercentage(samplingPercentage, sitk.sitkWallClock)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetOptimizerAsRegularStepGradientDescent(
        minStep=1e-4, numberOfIterations=500,
        relaxationFactor=0.5, learningRate=2.0,
        gradientMagnitudeTolerance=1e-8
    )

    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    final_transform = R.Execute(fixed, moving)

    cimg_array = resampler_method(fixed, moving, final_transform)

    return cimg_array

def registration_6(img0, img1):
    pixelType = sitk.sitkFloat32
    fixed = sitk.Cast(sitk.GetImageFromArray(img0), pixelType)
    moving = sitk.Cast(sitk.GetImageFromArray(img1), pixelType)

    transform_domain_mesh_size = [8] * moving.GetDimension()
    initial_transform = sitk.BSplineTransformInitializer(fixed,
                                                         transform_domain_mesh_size)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()

    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                           numberOfIterations=500,
                           maximumNumberOfCorrections=5,
                           maximumNumberOfFunctionEvaluations=1000,
                           costFunctionConvergenceFactor=1e7)

    R.SetInitialTransform(initial_transform, True)
    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

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