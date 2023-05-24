import os
import random
from dataclasses import dataclass
from argparse import ArgumentParser

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.functional import hflip, vflip, resize

parser = ArgumentParser()
parser.add_argument("--in_dir", type=str, help="input dir")
parser.add_argument("--list", type=str, help="path to list file")
parser.add_argument("--out_dir", type=str, help="output dir")


@dataclass
class Sample:
    name: str
    left: Image
    right: Image
    displ: Image
    dispr: Image
    disp0l: Image
    disp0r: Image


def read_sample(dir, sample_name):
    left_name = os.path.join(dir, sample_name, 'img_L.tif')
    right_name = os.path.join(dir, sample_name, 'img_R.tif')
    displ_name = os.path.join(dir, sample_name, 'disp_L_lidar.tif')
    dispr_name = os.path.join(dir, sample_name, 'disp_R_lidar.tif')
    disp0l_name = os.path.join(dir, sample_name, 'disp_L_lidar0.tif')
    disp0r_name = os.path.join(dir, sample_name, 'disp_R_lidar0.tif')
    
    return Sample(
        name=sample_name,
        left=Image.open(left_name),
        right=Image.open(right_name),
        displ=Image.open(displ_name),
        dispr=Image.open(dispr_name),
        disp0l=Image.open(disp0l_name),
        disp0r=Image.open(disp0r_name)
    )


def store_sample(dir, sample):
    sample_name = sample.name + f"_{random.randint(1, 100000)}"
    print(f'Storing sample {sample_name}')
    os.mkdir(os.path.join(dir, sample_name))

    left_name = os.path.join(dir, sample_name, 'img_L.tif')
    right_name = os.path.join(dir, sample_name, 'img_R.tif')
    displ_name = os.path.join(dir, sample_name, 'disp_L_lidar.tif')
    dispr_name = os.path.join(dir, sample_name, 'disp_R_lidar.tif')
    disp0l_name = os.path.join(dir, sample_name, 'disp_L_lidar0.tif')
    disp0r_name = os.path.join(dir, sample_name, 'disp_R_lidar0.tif')

    sample.left.save(left_name)
    sample.right.save(right_name)
    sample.displ.save(displ_name)
    sample.dispr.save(dispr_name)
    sample.disp0l.save(disp0l_name)
    sample.disp0r.save(disp0r_name)

        
def choose(probs):
    prob_sum = 0
    i = 0
    rnd = random.random()
    for p in probs:
        prob_sum += p
        if rnd < prob_sum:
            return i
        i += 1
    return i



def aug(func):
    def wrapper(samples):
        for sample in samples:
            yield func(sample)

    return wrapper

# horizontal flip begin
def hflip_sample(sample):
    return Sample(
            name=sample.name,
            left=hflip(sample.right),
            right=hflip(sample.left),
            displ=hflip(sample.dispr),
            dispr=hflip(sample.displ),
            disp0l=hflip(sample.disp0r),
            disp0r=hflip(sample.disp0l)
        )


def hor_flip_aug(sample, flip_prob):
    c = choose([1 - flip_prob, flip_prob])
    if c == 0:
        return sample
    else:
        print(f'hflip {sample.name}')
        return hflip_sample(sample)
# horizontal flip end

# vetrical flip begin
def vflip_sample(sample):
    return Sample(
            name=sample.name,
            left=vflip(sample.left),
            right=vflip(sample.right),
            displ=vflip(sample.displ),
            dispr=vflip(sample.dispr),
            disp0l=vflip(sample.disp0l),
            disp0r=vflip(sample.disp0r)
        )


def vert_flip_aug(sample, flip_prob):
    c = choose([1 - flip_prob, flip_prob])
    if c == 0:
        return sample
    else:
        print(f'vflip {sample.name}')
        return vflip_sample(sample)
# vertical flip end

# disparity shift begin
def crop_right(img, sz):
    return img.crop((0, 0, img.size[0] - sz, img.size[1]))


def crop_left(img, sz):
    return img.crop((sz, 0, img.size[0], img.size[1]))


def shift_disp(disp, shift):
    return Image.fromarray(np.asarray(disp) + shift)


def shift_sample(sample, shift):
    if shift == 0:
        return sample
    
    elif shift > 0: # increase disparity
        return Sample(
            name=sample.name,
            left=crop_right(sample.left, shift),
            right=crop_left(sample.right, shift),
            displ=shift_disp(crop_right(sample.displ, shift), shift),
            dispr=shift_disp(crop_left(sample.dispr, shift), shift),
            disp0l=shift_disp(crop_right(sample.disp0l, shift), shift),
            disp0r=shift_disp(crop_left(sample.disp0r, shift), shift)
        )
        
    else: # decrease disparity
        return Sample(
            name=sample.name,
            left=crop_left(sample.left, -shift),
            right=crop_right(sample.right, -shift),
            displ=shift_disp(crop_left(sample.displ, -shift), shift),
            dispr=shift_disp(crop_right(sample.dispr, -shift), shift),
            disp0l=shift_disp(crop_left(sample.disp0l, -shift), shift),
            disp0r=shift_disp(crop_right(sample.disp0r, -shift), shift)
        )
    

def shift_aug(sample, prob, max_abs_shift):
    c = choose([1 - prob, prob])
    if c == 0:
        return sample
    else:
        min_disp = np.nanmin(np.asarray(sample.disp0l))
        min_shift = max(-min_disp + 3, -max_abs_shift)
        max_shift = max_abs_shift

        shift = random.randint(min_shift, max_shift)
        print(f'shift {sample.name} by {shift}')

        return shift_sample(sample, shift)
# disparity shift end

# warping begin
def prepare_disp(disp, width):
    disp_zero_np = np.array(disp) 
    disp_zero_np[np.isnan(disp_zero_np)] = 0
    disp_zero_np = disp_zero_np[:, :width]
    return cv2.medianBlur(cv2.medianBlur(disp_zero_np, 5), 5)


def prepare_image(image, width):
    image_np = np.array(image)
    return np.expand_dims(image_np, -1)[:, :width]


def get_occlusion_mask(shifted, process_width):
    mask_up = shifted > 0
    mask_down = shifted > 0

    shifted_up = np.ceil(shifted)
    shifted_down = np.floor(shifted)

    for col in range(process_width - 2):
        loc = shifted[:, col:col + 1]  # keepdims
        loc_up = np.ceil(loc)
        loc_down = np.floor(loc)

        _mask_down = ((shifted_down[:, col + 2:] != loc_down) * (
        (shifted_up[:, col + 2:] != loc_down))).min(-1)
        _mask_up = ((shifted_down[:, col + 2:] != loc_up) * (
        (shifted_up[:, col + 2:] != loc_up))).min(-1)

        mask_up[:, col] = mask_up[:, col] * _mask_up
        mask_down[:, col] = mask_down[:, col] * _mask_down

    mask = mask_up + mask_down
    return mask


def project_image(image, disp_map, background_image, orig_height, process_width, disable_background=True):
    x_grid, y_grid = np.meshgrid(np.arange(process_width), np.arange(orig_height))

    image = np.array(image)
    # max_val = 255.
    max_val = np.max(image)

    background_image = np.array(background_image)

    # set up for projection
    warped_image = np.zeros_like(image).astype(float)
    warped_image = np.stack([warped_image] * 2, 0)
    pix_locations = x_grid - disp_map

    # find where occlusions are, and remove from disparity map
    mask = get_occlusion_mask(pix_locations, process_width)
    masked_pix_locations = pix_locations * mask - process_width * (1 - mask)

    # do projection - linear interpolate up to 1 pixel away
    weights = np.ones((2, orig_height, process_width)) * 10000

    for col in range(process_width - 1, -1, -1):
        loc = masked_pix_locations[:, col]
        loc_up = np.ceil(loc).astype(int)
        loc_down = np.floor(loc).astype(int)
        weight_up = loc_up - loc
        weight_down = 1 - weight_up

        mask = loc_up >= 0
        mask[mask] = \
            weights[0, np.arange(orig_height)[mask], loc_up[mask]] > weight_up[mask]
        weights[0, np.arange(orig_height)[mask], loc_up[mask]] = \
            weight_up[mask]
        warped_image[0, np.arange(orig_height)[mask], loc_up[mask]] = \
            image[:, col][mask] / max_val

        mask = loc_down >= 0
        mask[mask] = \
            weights[1, np.arange(orig_height)[mask], loc_down[mask]] > weight_down[mask]
        weights[1, np.arange(orig_height)[mask], loc_down[mask]] = weight_down[mask]
        warped_image[1, np.arange(orig_height)[mask], loc_down[mask]] = \
            image[:, col][mask] / max_val

    weights /= weights.sum(0, keepdims=True) + 1e-7  # normalise
    weights = np.expand_dims(weights, -1)
    warped_image = warped_image[0] * weights[1] + warped_image[1] * weights[0]
    warped_image *= max_val

    # now fill occluded regions with random background
    if not disable_background:
        warped_image[warped_image.max(-1) == 0] = background_image[warped_image.max(-1) == 0]

    # warped_image = warped_image.astype(np.uint8)

    return warped_image


def warp_right_from_left(sample: Sample, scale: float): # -> sample
    orig_w, orig_h = sample.left.size
    process_width = orig_w
    
    scaled_disp = (scale * prepare_disp(sample.disp0l, process_width)).round()
    
    projected_right = project_image(
        prepare_image(sample.left, process_width), 
        scaled_disp, 
        None, 
        orig_h, 
        process_width)[:,:,0]
        
    scaled_displ = (scale * np.array(sample.displ)).round()    
    scaled_disp0l = (scale * np.array(sample.disp0l)).round()
    
    return Sample(
        name=sample.name,
        left=sample.left,
        right=Image.fromarray(projected_right.round()),
        displ=Image.fromarray(scaled_displ),
        dispr=Image.fromarray(np.full((orig_h, orig_w), np.nan)), # we lose right disparity here
        disp0l=Image.fromarray(scaled_disp0l),
        disp0r=Image.fromarray(np.full((orig_h, orig_w), np.nan)) # we lose right disparity here
    )


def smooth_disp(disp_np):
    disp_np_zeroed = disp_np.copy()
    disp_np_zeroed[np.isnan(disp_np_zeroed)] = 0
    disp_np_zeroed = cv2.medianBlur(cv2.medianBlur(disp_np_zeroed, 5), 3)

    sizes = [5,15,31,51]
    blurred = disp_np_zeroed.copy()
    
    for s in sizes:
        blurred = cv2.blur(blurred, (s,s))
      
    mask = (disp_np_zeroed == 0)
    disp_np_zeroed[mask] = blurred[mask]
    return disp_np_zeroed


def warp_right(right, dispr, scale): # scale > 0 => left; scale < 0 => right
    W = right.size[0]
    H = right.size[1]

    scale_abs = abs(scale)
    
    if scale < 0:
        dispr = hflip(dispr)
        right = hflip(right)

    prep_disp = smooth_disp(np.asarray(dispr))
    prep_disp = (scale_abs * prep_disp).round()

    prep_right = prepare_image(right, W)

    projectedr = project_image(prep_right, prep_disp, None, H, W)
    
    if scale < 0:
        projectedr_result = hflip(Image.fromarray(projectedr[:, :, 0]))
    else:
        projectedr_result = Image.fromarray(projectedr[:, :, 0])
        
    #return projectedr_result

    smooth_result = Image.fromarray(cv2.medianBlur(np.asarray(projectedr_result), 3))
    return smooth_result


def warp_right_from_right(sample: Sample, scale: float):
    orig_w, orig_h = sample.left.size

    projected_right = warp_right(sample.right, sample.disp0r, scale - 1)
        
    scaled_displ = (scale * np.array(sample.displ)).round()    
    scaled_disp0l = (scale * np.array(sample.disp0l)).round()
    
    return Sample(
        name=sample.name,
        left=sample.left,
        right=projected_right,
        displ=Image.fromarray(scaled_displ),
        dispr=Image.fromarray(np.full((orig_h, orig_w), np.nan)), # we lose right disparity here
        disp0l=Image.fromarray(scaled_disp0l),
        disp0r=Image.fromarray(np.full((orig_h, orig_w), np.nan)) # we lose right disparity here
    )
    

def warp_aug(sample, warp_prob, max_scale_diff):
    c = choose([1 - warp_prob, warp_prob])
    if c == 0:
        return sample
    elif c == 1: # warp right from left
        scale = 1 + (2 * random.random() - 1) * max_scale_diff
        print(f'warp {sample.name} with scale {scale}')
        # return warp_right_from_left(sample, scale) 
        return warp_right_from_right(sample, scale)
# warping end

# scale begin
def scale_image(img, scale):
    new_size = (np.array(img.size) * scale).astype(int)
    return img.resize(new_size)


def scale_disp(disp, scale):
    new_size = (np.array(disp.size) * scale).astype(int)
    scaled_disp = disp.resize(new_size, Image.NEAREST)
    scaled_disp = (np.asarray(scaled_disp) * scale).round()
    return Image.fromarray(scaled_disp)


def scale_sample(sample, scale):
    return Sample(
            name=sample.name,
            left=scale_image(sample.left, scale),
            right=scale_image(sample.right, scale),
            displ=scale_disp(sample.displ, scale),
            dispr=scale_disp(sample.dispr, scale),
            disp0l=scale_disp(sample.disp0l, scale),
            disp0r=scale_disp(sample.disp0r, scale)
        )


def scale_aug(sample, prob, max_scale_diff):
    c = choose([1 - prob, prob])
    if c == 0:
        return sample
    elif c == 1:
        scale = 1 + (2 * random.random() - 1) * max_scale_diff
        print(f'scale {sample.name} with scale {scale}')
        return scale_sample(sample, scale)
# scale end


# crop begin
def make_crop_box(src_h, src_w, crop_h, crop_w):
    max_left = src_w - crop_w
    max_top = src_h - crop_h
    left = random.randrange(0, max_left)
    top = random.randrange(0, max_top)
    return (left, top, left + crop_w, top + crop_h)


def random_crop_sample(sample, crop_size):
    box = make_crop_box(sample.left.size[1], sample.left.size[0], crop_size[1], crop_size[0])
    return Sample(
            name=sample.name,
            left=sample.left.crop(box),
            right=sample.right.crop(box),
            displ=sample.displ.crop(box),
            dispr=sample.dispr.crop(box),
            disp0l=sample.disp0l.crop(box),
            disp0r=sample.disp0r.crop(box)
        )


def crop_aug(samples, crop_size, n_crops): # size = (w, h)
    for s in samples:
        for i in range(n_crops):
            yield random_crop_sample(s, crop_size)
# crop end


pipeline = [
    aug(lambda s: hor_flip_aug(s, 0.5)),
    aug(lambda s: warp_aug(s, 0.5, 0.3)),
    aug(lambda s: shift_aug(s, 0.5, 32)),
    aug(lambda s: scale_aug(s, 0.3, 0.2)),
    lambda ss: crop_aug(ss, (700, 450), 5),
    aug(lambda s: vert_flip_aug(s, 0.5)),
]


def run(samples, pipeline):
    for f in pipeline:
        samples = f(samples)

    return samples


def read_samples(dir, sample_names):
    for sample_name in sample_names:
        print(f'Reading {sample_name}')
        yield read_sample(dir, sample_name)


if __name__ == "__main__":
    args = parser.parse_args()
    sample_names = []
    with open(args.list) as list_file:
        for line in list_file:
            sample_names.append(line.rstrip())
    
    result = run(read_samples(args.in_dir, sample_names), pipeline)

    os.mkdir(args.out_dir)
    for sample in result:
        store_sample(args.out_dir, sample)
