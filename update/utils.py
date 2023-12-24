import os
import shutil
import torch.nn as nn
import torch
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import math
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as compare_ssim

cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9


def make_dir(path, overwrite=False):
    path = os.path.normpath(path)
    if not os.path.exists(path):
        os.mkdir(path)
    elif os.path.exists(path) and overwrite:
        shutil.rmtree(path)
        os.mkdir(path)


def pad_image(field, target_shape, pytorch=True, stacked_complex=True, padval=0, mode='constant'):
    """Pads a 2D complex field up to target_shape in size

    Padding is done such that when used with crop_image(), odd and even dimensions are
    handled correctly to properly undo the padding.

    field: the field to be padded. May have as many leading dimensions as necessary
        (e.g., batch or channel dimensions)
    target_shape: the 2D target output dimensions. If any dimensions are smaller
        than field, no padding is applied
    pytorch: if True, uses torch functions, if False, uses numpy
    stacked_complex: for pytorch=True, indicates that field has a final dimension
        representing real and imag
    padval: the real number value to pad by
    mode: padding mode for numpy or torch
    """
    if pytorch:
        if stacked_complex:
            size_diff = np.array(target_shape) - np.array(field.shape[-3:-1])
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(target_shape) - np.array(field.shape[-2:])
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(target_shape) - np.array(field.shape[-2:])
        odd_dim = np.array(field.shape[-2:]) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 0 - odd_dim) // 2

        if pytorch:
            pad_axes = [int(p)  # convert from np.int64
                        for tple in zip(pad_front[::-1], pad_end[::-1])
                        for p in tple]
            return nn.functional.pad(field, pad_axes, mode=mode, value=padval)
        else:
            leading_dims = field.ndim - 2  # only pad the last two dims
            if leading_dims > 0:
                pad_front = np.concatenate(([0] * leading_dims, pad_front))
                pad_end = np.concatenate(([0] * leading_dims, pad_end))
            return np.pad(field, tuple(zip(pad_front, pad_end)), mode,
                          constant_values=padval)
    else:
        return field


def crop_image(field, target_shape, pytorch=True, stacked_complex=True):
    """Crops a 2D field, see pad_image() for details

    No cropping is done if target_shape is already smaller than field
    """
    if target_shape is None:
        return field

    if pytorch:
        if stacked_complex:
            size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
        odd_dim = np.array(field.shape[-2:]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 0 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if pytorch and stacked_complex:
            return field[(..., *crop_slices, slice(None))]
        else:
            return field[(..., *crop_slices)]
    else:
        return field


def rect_to_polar(real, imag):
    """Converts the rectangular complex representation to polar"""
    mag = torch.pow(real**2 + imag**2, 0.5)
    ang = torch.atan2(imag, real)
    return mag, ang


def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular"""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag


def ifftshift(tensor):
    """ifftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, -math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, -math.floor(size[3] / 2.0), 3)
    return tensor_shifted


def fftshift(tensor):
    """fftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, math.floor(size[3] / 2.0), 3)
    return tensor_shifted


def roll_torch(tensor, shift, axis):
    """implements numpy roll() or Matlab circshift() functions for tensors"""
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)

def psnr2(target_amp, recon_amp):
    target_amp = target_amp.cpu().detach().numpy()
    recon_amp = recon_amp.cpu().detach().numpy()
    mse = np.mean((target_amp - recon_amp) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def rmse2(target_amp, recon_amp):
    target_amp = target_amp.cpu().detach().numpy()
    recon_amp = recon_amp.cpu().detach().numpy()
    mse = np.mean((target_amp - recon_amp) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def ssim2(target_amp, recon_amp, color = False):
    target_amp = target_amp.cpu().detach().numpy()[0, 0, :, :]
    recon_amp = recon_amp.cpu().detach().numpy()[0, 0, :, :]
    ssim = compare_ssim(target_amp,recon_amp,multichannel = color)
    return ssim


def predict(net, propagator_forward, propagator_backward, image, prop_dist, feature_size, wavelength, precomputed_H, precomputed_H_inverse, downscale_factor):
    
    obj_real, obj_imag = polar_to_rect(image, torch.zeros_like(image))
    object_field = torch.complex(obj_real, obj_imag)
    input_field = propagator_backward(u_in=object_field, z=-prop_dist, feature_size=feature_size, wavelength=wavelength,
                                dtype=torch.float32, precomped_H=precomputed_H_inverse)

    input_amp = input_field.abs()
    input_field = (input_field - torch.min(input_amp))/(torch.max(input_amp) - torch.min(input_amp))

    input_real = input_field.real
    input_img = input_field.imag
    input_cat = torch.cat([input_real, input_img], 1)

    input_unshuffle = F.pixel_unshuffle(input_cat, downscale_factor)

    slm_phase_shuffle = net(input_unshuffle)
    slm_phase = F.pixel_shuffle(slm_phase_shuffle, downscale_factor)
    
    slm_real, slm_imag = polar_to_rect(torch.ones_like(slm_phase), slm_phase)
    slm_field = torch.complex(slm_real, slm_imag)
    recon_field = propagator_forward(u_in=slm_field, z=prop_dist, feature_size=feature_size, wavelength=wavelength,
                                dtype=torch.float32, precomped_H=precomputed_H)

    recon_amp = recon_field.abs()
    recon_phase = recon_field.angle()

    return recon_amp, recon_phase, slm_phase


def to_tensorboard(epoch, recon_amp, recon_phase, target, loss, psnr, ssim, writer, train_valid):
    recon_amp = recon_amp.cpu().detach().numpy()[0,0,...]
    recon_amp = cv2.normalize(recon_amp.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    recon_amp = recon_amp[np.newaxis,...]

    recon_phase = recon_phase.cpu().detach().numpy()[0,0,...]
    recon_phase = (recon_phase + np.pi) % (2 * np.pi) / (2 * np.pi)
    recon_phase = recon_phase[np.newaxis,...]

    target = target.cpu().detach().numpy()[0,0,...]
    target = cv2.normalize(target.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    target = target[np.newaxis,...]

    writer.add_image(train_valid + '/amplitude', recon_amp, epoch)

    writer.add_image(train_valid + '/phase', recon_phase, epoch)

    writer.add_image(train_valid + '/target', target, epoch)


    writer.add_scalar(train_valid + '/loss', loss, epoch)
    writer.add_scalar(train_valid + '/psnr', psnr, epoch)
    writer.add_scalar(train_valid + '/ssim', ssim, epoch)


def save_results(out_amp, slm_phase, recon_phase, epoch, res_dir, psnrValue, ssimValue, zero_shift=False):
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    I_r_np = out_amp
    I_r_np = I_r_np.cpu().detach().numpy()
    I_r_np = cv2.normalize(I_r_np.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    # I_r_np = np.clip(I_r_np, 0.0, 1.0)
    I_r_np = I_r_np[0, 0, :, :]
    I_r_name = "recon_intensity_epoch_{:03d}_psnr_{:3f}_ssim_{:4f}.jpg".format(epoch, psnrValue, ssimValue)
    I_r_name = os.path.join(res_dir, I_r_name)
    cv2.imencode('.jpg', np.uint8(I_r_np * 255))[1].tofile(I_r_name)

    if zero_shift:
        linear_phase = torch.arange(slm_phase.size()[2], device=slm_phase.device) * torch.pi/2
        slm_phase  = slm_phase + linear_phase[None, None, :, None]
    phase_np = slm_phase
    phase_np = phase_np.cpu().detach().numpy()
    phase_np = phase_np[0, 0, :, :]
    phase_np = (phase_np + np.pi) % (2 * np.pi) / (2 * np.pi)
    phase_np = np.flip(phase_np, 0)
    phase_name = "target_phase_epoch_{:03d}_psnr_{:3f}_ssim_{:4f}.bmp".format(epoch, psnrValue, ssimValue)
    phase_name = os.path.join(res_dir, phase_name)
    cv2.imencode('.bmp', np.uint8(phase_np * 255))[1].tofile(phase_name)

    phase_np = recon_phase
    phase_np = phase_np.cpu().detach().numpy()
    phase_np = phase_np[0, 0, :, :]
    phase_np = (phase_np + np.pi) % (2 * np.pi) / (2 * np.pi)
    phase_name = "recon_phase_epoch_{:03d}_psnr_{:3f}_ssim_{:4f}.jpg".format(epoch, psnrValue, ssimValue)
    phase_name = os.path.join(res_dir, phase_name)
    cv2.imencode('.jpg', np.uint8(phase_np * 255))[1].tofile(phase_name)



def generate_calibration_img(slm_res, roi_res, num_circles, circle_radius):
    space_btw_circs = [int(np.ceil(roi / (num_circs - 1))) for roi, num_circs in zip(roi_res, (num_circles[1], num_circles[0]))]
    cali_image = np.zeros(slm_res, dtype=np.float32)
    for i in range(num_circles[0]):
        for j in range(num_circles[1]):
            cv2.circle(cali_image, ((slm_res[1]-roi_res[1])//2+i*space_btw_circs[1], (slm_res[0]-roi_res[0])//2+j*space_btw_circs[0]),circle_radius,(255,255,255),-1)

    # check1 = (num_circles[0]-1)*space_btw_circs[1]
    # check2 = (num_circles[1]-1)*space_btw_circs[0]

    # plt.imshow(cali_image, cmap = 'gray')
    # plt.show()
    return cali_image
