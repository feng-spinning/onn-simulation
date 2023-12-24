import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from utils import *
import torch.fft
import torch.nn.functional as F



def propagation_ASM_encrypted1(u_in, L,M, lmda_rgb, z, linear_conv=True,
                    padtype='zero', return_H=False, precomped_H=None,
                    return_H_exp=False, precomped_H_exp=None,
                    dtype=torch.float32):

    wavelength = np.array([lmda_rgb, lmda_rgb, lmda_rgb])


    if linear_conv:
        # preprocess with padding for linear conv.
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        if padtype == 'zero':
            padval = 0
        elif padtype == 'median':
            padval = torch.median(torch.pow((u_in**2).sum(-1), 0.5))
        u_in = pad_image(u_in, conv_size, padval=padval, stacked_complex=False)

    if precomped_H is None and precomped_H_exp is None:
        # resolution of input field, should be: (num_images, num_channels, height, width, 2)
        field_resolution = u_in.size()

        # number of pixels
        num_y, num_x = field_resolution[2], field_resolution[3]
        wavelength = np.repeat(np.expand_dims(wavelength, axis=1), num_y, axis=1)
        wavelength = np.repeat(np.expand_dims(wavelength, axis=2), num_x, axis=2)

        # sampling inteval size
        dy = L / M
        dx = L / M

        # size of the field
        y, x = (dy * float(num_y), dx * float(num_x))

        # frequency coordinates sampling
        fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
        fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)

        # momentum/reciprocal space
        FX, FY = np.meshgrid(fx, fy)
        FX = np.repeat(np.expand_dims(FX, axis=0), 3, axis=0)
        FY = np.repeat(np.expand_dims(FY, axis=0), 3, axis=0)
        # transfer function in numpy (omit distance)
        HH = 2 * math.pi * np.sqrt(1 / wavelength**2 - (FX**2 + FY**2))

        # create tensor & upload to device (GPU)
        H_exp = torch.tensor(HH, dtype=dtype).to(u_in.device)

        ###
        # here one may iterate over multiple distances, once H_exp is uploaded on GPU

        # reshape tensor and multiply
        H_exp = torch.reshape(H_exp, (1, *H_exp.size()))

    # handle loading the precomputed H_exp value, or saving it for later runs
    elif precomped_H_exp is not None:
        H_exp = precomped_H_exp

    if precomped_H is None:
        # multiply by distance
        H_exp = torch.mul(H_exp, z)

        # band-limited ASM - Matsushima et al. (2009)
        fy_max = 1 / np.sqrt((2 * z * (1 / y))**2 + 1) / wavelength
        fx_max = 1 / np.sqrt((2 * z * (1 / x))**2 + 1) / wavelength
        H_filter1 = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=dtype)

        fy_half = 1 / (4 * dy)
        fx_half = 1 / (4 * dx)
        H_filter2 = torch.tensor(((np.abs(FX) < fx_half) & (np.abs(FY) < fy_half)).astype(np.uint8), dtype=dtype)
        # H_filter2 = torch.tensor((np.sqrt(np.abs(FX)**2 + np.abs(FY)**2) < np.sqrt((np.abs(fy_half)**2 + np.abs(fx_half)**2)/2)).astype(np.uint8), dtype=dtype)

        H_filter = H_filter1 * H_filter2

        # get real/img components
        H_real, H_imag = polar_to_rect(H_filter.to(u_in.device), H_exp)

        H = torch.stack((H_real, H_imag), 4)
        H = ifftshift(H)
        H = torch.view_as_complex(H)
    else:
        H = precomped_H

    # return for use later as precomputed inputs
    if return_H_exp:
        return H_exp
    if return_H:
        return H

    # For who cannot use Pytorch 1.7.0 and its Complex tensors support:
    # # angular spectrum
    # U1 = torch.fft(ifftshift(u_in), 2, True)
    #
    # # convolution of the system
    # U2 = mul_complex(H, U1)
    #
    # # Fourier transform of the convolution to the observation plane
    # u_out = fftshift(torch.ifft(U2, 2, True))

    U1 = torch.fft.fftn(ifftshift(u_in), dim=(-2, -1), norm='ortho')

    U2 = H * U1

    u_out = fftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))

    if linear_conv:
        # return crop_image(u_out, input_resolution) # using stacked version
        return crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False)  # using complex tensor
    else:
        return u_out


def propagation_ASM_encrypted2(u_in, feature_size, wavelength, z, linear_conv=True,
                    padtype='zero', return_H=False, precomped_H=None,
                    return_H_exp=False, precomped_H_exp=None,
                    dtype=torch.float32):


    if linear_conv:
        # preprocess with padding for linear conv.
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        if padtype == 'zero':
            padval = 0
        elif padtype == 'median':
            padval = torch.median(torch.pow((u_in**2).sum(-1), 0.5))
        u_in = pad_image(u_in, conv_size, padval=padval, stacked_complex=False)

    if precomped_H is None and precomped_H_exp is None:
        # resolution of input field, should be: (num_images, num_channels, height, width, 2)
        field_resolution = u_in.size()

        # number of pixels
        num_y, num_x = field_resolution[2], field_resolution[3]

        # sampling inteval size
        dy, dx = feature_size

        # size of the field
        y, x = (dy * float(num_y), dx * float(num_x))

        # frequency coordinates sampling
        fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
        fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)

        # momentum/reciprocal space
        FX, FY = np.meshgrid(fx, fy)

        # transfer function in numpy (omit distance)
        HH = 2 * math.pi * np.sqrt(1 / wavelength**2 - (FX**2 + FY**2))

        # create tensor & upload to device (GPU)
        H_exp = torch.tensor(HH, dtype=dtype).to(u_in.device)

        ###
        # here one may iterate over multiple distances, once H_exp is uploaded on GPU

        # reshape tensor and multiply
        H_exp = torch.reshape(H_exp, (1, 1, *H_exp.size()))

    # handle loading the precomputed H_exp value, or saving it for later runs
    elif precomped_H_exp is not None:
        H_exp = precomped_H_exp

    if precomped_H is None:
        # multiply by distance
        H_exp = torch.mul(H_exp, z)

        # band-limited ASM - Matsushima et al. (2009)
        fy_max = 1 / np.sqrt((2 * z * (1 / y))**2 + 1) / wavelength
        fx_max = 1 / np.sqrt((2 * z * (1 / x))**2 + 1) / wavelength
        H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=dtype)

        # get real/img components
        H_real, H_imag = polar_to_rect(H_filter.to(u_in.device), H_exp)
        H_real = pad_image(H_real, [i * 2 for i in H_real.size()[-2:]], padval=0, stacked_complex=False)
        H_imag = pad_image(H_imag, [i * 2 for i in H_imag.size()[-2:]], padval=0, stacked_complex=False)

        H = torch.stack((H_real, H_imag), 4)
        H = ifftshift(H)
        H = torch.view_as_complex(H)
    else:
        H = precomped_H

    # return for use later as precomputed inputs
    if return_H_exp:
        return H_exp
    if return_H:
        return H


    U1 = torch.fft.fftn(ifftshift(u_in), dim=(-2, -1), norm='ortho')
    padx = (H.size(2) - U1.size(2))//2
    pady = (H.size(3) - U1.size(3))//2
    U1 = ifftshift(F.pad(fftshift(U1),pad = (pady,pady,padx,padx), mode="constant", value=0))

    U2 = H * U1

    u_out = fftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))
    # u_out = u_out.resize(origin_size[0], origin_size[1], origin_size[2], origin_size[3])

    if linear_conv:
        # return crop_image(u_out, input_resolution) # using stacked version
        return crop_image(u_out, [i * 2 for i in input_resolution], pytorch=True, stacked_complex=False)  # using complex tensor
    else:
        return u_out




def propagation_ASM(u_in, feature_size, wavelength, z, linear_conv=True,
                              padtype='zero', return_H=False, precomped_H=None,
                              return_H_exp=False, precomped_H_exp=None,
                              dtype=torch.float32):
    if linear_conv:
        # preprocess with padding for linear conv.
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        if padtype == 'zero':
            padval = 0
        elif padtype == 'median':
            padval = torch.median(torch.pow((u_in ** 2).sum(-1), 0.5))
        u_in = pad_image(u_in, conv_size, padval=padval, stacked_complex=False)

    if precomped_H is None and precomped_H_exp is None:
        # resolution of input field, should be: (num_images, num_channels, height, width, 2)
        field_resolution = u_in.size()

        # number of pixels
        num_y, num_x = field_resolution[2], field_resolution[3]

        # sampling inteval size
        dy, dx = feature_size

        # size of the field
        y, x = (dy * float(num_y), dx * float(num_x))

        # frequency coordinates sampling
        fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
        fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)

        # momentum/reciprocal space
        FX, FY = np.meshgrid(fx, fy)

        # transfer function in numpy (omit distance)
        HH = 2 * math.pi * np.sqrt(1 / wavelength ** 2 - (FX ** 2 + FY ** 2))

        # create tensor & upload to device (GPU)
        H_exp = torch.tensor(HH, dtype=dtype).to(u_in.device)

        ###
        # here one may iterate over multiple distances, once H_exp is uploaded on GPU

        # reshape tensor and multiply
        H_exp = torch.reshape(H_exp, (1, 1, *H_exp.size()))

    # handle loading the precomputed H_exp value, or saving it for later runs
    elif precomped_H_exp is not None:
        H_exp = precomped_H_exp

    if precomped_H is None:
        # multiply by distance
        H_exp = torch.mul(H_exp, z)

        # band-limited ASM - Matsushima et al. (2009)
        fy_max = 1 / np.sqrt((2 * z * (1 / y)) ** 2 + 1) / wavelength
        fx_max = 1 / np.sqrt((2 * z * (1 / x)) ** 2 + 1) / wavelength
        H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=dtype)
        # H_filter = torch.ones_like(H_exp)


        # get real/img components
        H_real, H_imag = polar_to_rect(H_filter.to(u_in.device), H_exp)

        H = torch.stack((H_real, H_imag), 4)
        H = ifftshift(H)
        H = torch.view_as_complex(H)
    else:
        H = precomped_H

    # return for use later as precomputed inputs
    if return_H_exp:
        return H_exp
    if return_H:
        return H

    # For who cannot use Pytorch 1.7.0 and its Complex tensors support:
    # # angular spectrum
    # U1 = torch.fft(ifftshift(u_in), 2, True)
    #
    # # convolution of the system
    # U2 = mul_complex(H, U1)
    #
    # # Fourier transform of the convolution to the observation plane
    # u_out = fftshift(torch.ifft(U2, 2, True))

    U1 = torch.fft.fftn(ifftshift(u_in), dim=(-2, -1), norm='ortho')

    U2 = H * U1

    u_out = fftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))

    if linear_conv:
        # return crop_image(u_out, input_resolution) # using stacked version
        return crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False)  # using complex tensor
    else:
        return u_out


def propagation_ASM_mask(u_in, feature_size, wavelength, z, mask, linear_conv=True,
                              padtype='zero', return_H=False, precomped_H=None,
                              return_H_exp=False, precomped_H_exp=None,
                              dtype=torch.float32):
    if linear_conv:
        # preprocess with padding for linear conv.
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        if padtype == 'zero':
            padval = 0
        elif padtype == 'median':
            padval = torch.median(torch.pow((u_in ** 2).sum(-1), 0.5))
        u_in = pad_image(u_in, conv_size, padval=padval, stacked_complex=False)

    if precomped_H is None and precomped_H_exp is None:
        # resolution of input field, should be: (num_images, num_channels, height, width, 2)
        field_resolution = u_in.size()

        # number of pixels
        num_y, num_x = field_resolution[2], field_resolution[3]

        # sampling inteval size
        dy, dx = feature_size

        # size of the field
        y, x = (dy * float(num_y), dx * float(num_x))

        # frequency coordinates sampling
        fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
        fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)

        # momentum/reciprocal space
        FX, FY = np.meshgrid(fx, fy)

        # transfer function in numpy (omit distance)
        HH = 2 * math.pi * np.sqrt(1 / wavelength ** 2 - (FX ** 2 + FY ** 2))

        # create tensor & upload to device (GPU)
        H_exp = torch.tensor(HH, dtype=dtype).to(u_in.device)

        ###
        # here one may iterate over multiple distances, once H_exp is uploaded on GPU

        # reshape tensor and multiply
        H_exp = torch.reshape(H_exp, (1, 1, *H_exp.size()))

    # handle loading the precomputed H_exp value, or saving it for later runs
    elif precomped_H_exp is not None:
        H_exp = precomped_H_exp

    if precomped_H is None:
        # multiply by distance
        H_exp = torch.mul(H_exp, z)

        # band-limited ASM - Matsushima et al. (2009)
        fy_max = 1 / np.sqrt((2 * z * (1 / y)) ** 2 + 1) / wavelength
        fx_max = 1 / np.sqrt((2 * z * (1 / x)) ** 2 + 1) / wavelength
        H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=dtype)
        # H_filter = torch.ones_like(H_exp)


        # get real/img components
        H_real, H_imag = polar_to_rect(H_filter.to(u_in.device), H_exp)

        H = torch.stack((H_real, H_imag), 4)
        H = ifftshift(H)
        H = torch.view_as_complex(H)
    else:
        H = precomped_H

    # return for use later as precomputed inputs
    if return_H_exp:
        return H_exp
    if return_H:
        return H

    mask = ifftshift(mask)

    U1 = torch.fft.fftn(ifftshift(u_in), dim=(-2, -1), norm='ortho')

    U2_mask = mask * U1

    U2 = U2_mask * H

    u_out = fftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))

    target = fftshift(torch.fft.ifftn(U2_mask, dim=(-2, -1), norm='ortho'))

    if linear_conv:
        # return crop_image(u_out, input_resolution) # using stacked version
        return crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False), \
               crop_image(target, input_resolution, pytorch=True, stacked_complex=False)# using complex tensor
    else:
        return u_out, target



def propagation_mask(mask, dtype=torch.float32):

    U1 = torch.ones_like(mask)

    U2 = mask * U1

    u_out = fftshift(torch.fft.ifftn(ifftshift(U2), dim=(-2, -1), norm='ortho'))

    return u_out