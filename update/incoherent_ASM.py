from utils import *
import torch.fft

from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, Pad, InterpolationMode

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def propagation_ASM_incoherent(u_in, L, M, wavelength, z, linear_conv=True,
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
        num_y, num_x = field_resolution[1], field_resolution[2]

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
        H_filter1 = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=dtype)

        fy_half = 1 / (4 * dy)
        fx_half = 1 / (4 * dx)
        H_filter2 = torch.tensor(((np.abs(FX) < fx_half) & (np.abs(FY) < fy_half)).astype(np.uint8), dtype=dtype)
        # H_filter2 = torch.tensor((np.sqrt(np.abs(FX)**2 + np.abs(FY)**2) < np.sqrt((np.abs(fy_half)**2 + np.abs(fx_half)**2)/2)).astype(np.uint8), dtype=dtype)

        H_filter = H_filter1 * H_filter2

        # get real/img components
        H_real, H_imag = polar_to_rect(H_filter.to(u_in.device), H_exp)

        H = torch.stack((H_real, H_imag), 4)
        H = torch.view_as_complex(H)
        H_padder = Pad((H.size()[-1]//2, H.size()[-2]//2), fill=0, padding_mode='constant')
        H_cropper = CenterCrop((H.size()[-2], H.size()[-1]))

        H = H_padder(H)
        
        H_space = fftshift(torch.fft.fftn(ifftshift(H), dim=(-2, -1), norm='ortho'))
        H_power = torch.conj_physical(H_space) * H_space

        H_space_h = torch.arange(H_space.size()[-2]).to(u_in.device) - H_space.size()[-2]//2
        H_space_w = torch.arange(H_space.size()[-1]).to(u_in.device) - H_space.size()[-1]//2
        H_space_H, H_space_W = torch.meshgrid(H_space_h, H_space_w, indexing='ij')
        mask = torch.zeros_like(H_space_H)
        mask[((H_space_H*H_space.size()[-1]/H_space.size()[-2]) ** 2 + H_space_W ** 2)<=(H_space.size()[-1]//2)**2] = 1

        H_power = H_power * mask

        H = fftshift(torch.fft.ifftn(ifftshift(H_power), dim=(-2, -1), norm='ortho'))

        H = H_cropper(H)

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
    U1 = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(u_in), dim=(-2, -1), norm='ortho'))

    U2 = H * U1

    u_out = fftshift(torch.fft.ifftn(ifftshift(U2), dim=(-2, -1), norm='ortho'))

    if linear_conv:
        # return crop_image(u_out, input_resolution) # using stacked version
        return crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False)  # using complex tensor
    else:
        return u_out