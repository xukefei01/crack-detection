import cv2
import numpy as np
import os


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label=None):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class Normalize(object):
    def __init__(self, mean_val, std_val, val_scale=1):
        # set val_scale = 1 if mean and std are in range (0,1)
        # set val_scale to other value, if mean and std are in range (0,255)
        self.mean = np.array(mean_val, dtype=np.float32)
        self.std = np.array(std_val, dtype=np.float32)
        self.val_scale = 1 / 255.0 if val_scale == 1 else 1

    def __call__(self, image, label=None):
        image = image.astype(np.float32)
        image = image * self.val_scale
        image = image - self.mean
        image = image * (1 / self.std)
        return image, label


class ConvertDataType(object):
    def __call__(self, image, label=None):
        if label is not None:
            label = label.astype(np.int64)
        return image.astype(np.float32), label


class Pad(object):
    def __init__(self, size, ignore_label=255, mean_val=0, val_scale=1):
        # set val_scale to 1 if mean_val is in range (0, 1)
        # set val_scale to 255 if mean_val is in range (0, 255) 
        factor = 255 if val_scale == 1 else 1

        self.size = size
        self.ignore_label = ignore_label
        self.mean_val = mean_val
        # from 0-1 to 0-255
        if isinstance(self.mean_val, (tuple, list)):
            self.mean_val = [int(x * factor) for x in self.mean_val]
        else:
            self.mean_val = int(self.mean_val * factor)

    def __call__(self, image, label=None):
        h, w, c = image.shape
        pad_h = max(self.size - h, 0)
        pad_w = max(self.size - w, 0)

        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)

        if pad_h > 0 or pad_w > 0:

            image = cv2.copyMakeBorder(image,
                                       top=pad_h_half,
                                       left=pad_w_half,
                                       bottom=pad_h - pad_h_half,
                                       right=pad_w - pad_w_half,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=self.mean_val)
            if label is not None:
                label = cv2.copyMakeBorder(label,
                                           top=pad_h_half,
                                           left=pad_w_half,
                                           bottom=pad_h - pad_h_half,
                                           right=pad_w - pad_w_half,
                                           borderType=cv2.BORDER_CONSTANT,
                                           value=self.ignore_label)
        return image, label


# TODO
class CenterCrop(object):
    def __init__(self, crop_size):
        self.crop_h = crop_size
        self.crop_w = crop_size

    def __call__(self, image, label):
        h, w, c = image.shape
        h_start = (h - self.crop_h) // 2
        w_start = (w - self.crop_w) // 2
        image = image[h_start: h_start + self.crop_h, w_start: w_start + self.crop_w, :]
        if label is not None:
            label = label[h_start: h_start + self.crop_h, w_start: w_start + self.crop_w, :]

        return image, label


# TODO
class Resize(object):
    def __init__(self, size):
        assert type(size) in [int, tuple], "检查输入size类型"
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, image, label=None):
        image = cv2.resize(image, dsize=self.size, interpolation=cv2.INTER_NEAREST)  # 最邻近差值

        if label is not None:
            label = cv2.resize(label, dsize=self.size, interpolation=cv2.INTER_NEAREST)

        return image, label


# TODO
class RandomFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, label=None):
        prob_of_flip = np.random.rand()
        if prob_of_flip > self.prob:
            image = cv2.flip(image, 1)
            if label is not None:
                label = cv2.flip(label, 1)

        return image, label


# TODO
class RandomCrop(object):
    def __init__(self, size):
        assert type(size) in [int, tuple], "CHECK SIZE TYPE!"
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, image, label):
        h, w = image.shape[:2]
        h_start = np.random.randint(0, h - self.size[0] + 1)
        w_start = np.random.randint(0, w - self.size[1] + 1)
        h_end, w_end = h_start + self.size[0], w_start + self.size[1]

        image = image[h_start:h_end, w_start:w_end, :]
        label = label[h_start:h_end, w_start:w_end]

        return image, label


# TODO
class Scale(object):
    def __init__(self, scale=0.5):
        self.scale = scale

    def __call__(self, image, label=None):
        image = cv2.resize(image, dsize=None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, dsize=None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)

        return image, label


# TODO
class RandomScale(object):
    def __init__(self, scales=None):
        if scales is None:
            scales = [0.5, 1, 2, 3]
        self.scales = scales

    def __call__(self, image, label=None):
        scale = float(np.random.choice(self.scales))
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return image, label


def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    image_path = os.listdir('E:\dataset\CFD\images/')
    label_path = os.listdir('E:\dataset\CFD\masks/')
    image = cv2.imread(image_path, 1)
    label = cv2.imread(label_path, 0)
    # TODO: crop_size
    # TODO: Transform: RandomSacle, RandomFlip, Pad, RandomCrop
    # print(image.shape, label.shape)
    crop_size = 448
    transform = Compose([RandomScale(scales=[0.5, 1, 2]),
                         RandomFlip(prob=0.5),
                         Pad(size=crop_size),
                         RandomCrop(size=crop_size)])
    for i in range(10):
        # TODO: call transform
        image, label = transform(image, label)
        # TODO: save image
        save_path1 = os.path.join(cwd, 'images')
        save_path2 = os.path.join(cwd, 'masks')
        if not os.path.exists(save_path1):
            os.makedirs(save_path1)
        if not os.path.exists(save_path2):
            os.makedirs(save_path2)
        cv2.imwrite(os.path.join(save_path1, 'CFD_{}.jpg'.format(i)), image)
        cv2.imwrite(os.path.join(save_path2, 'CFD_{}.jpg'.format(i)), label)


if __name__ == "__main__":
    main()
