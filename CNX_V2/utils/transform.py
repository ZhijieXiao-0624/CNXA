from PIL import Image
from torchvision.transforms import *
import matplotlib.pyplot as plt


class Undistorted2Resize(object):
    def __init__(self, height, width, resize=True):
        self.resize = resize
        self.height = height
        self.width = width

    def __call__(self, img):
        iw, ih = img.size

        if self.resize:
            scale = min(self.width / iw, self.height / ih)

            nw = int(iw * scale)
            nh = int(ih * scale)

            img = img.resize((nw, nh), Image.BICUBIC)
            new_img = Image.new('RGB', (self.width, self.height), (128, 128, 128))
            new_img.paste(img, ((self.width - nw) // 2, (self.height - nh) // 2))
        else:
            new_img = img.resize((self.width, self.height), Image.BICUBIC)

        return new_img


# if __name__ == '__main__':
#     img = Image.open('E://self_dataset/imagenet100/imagenet100/n02105505/n02105505_48.JPEG')
#
#     transform = Undistorted2Resize(300, 300)
#     img_r = transform(img)
#
#     plt.figure(12)
#     plt.subplot(121)
#     plt.imshow(img)
#     plt.subplot(122)
#     plt.imshow(img_r)
#     plt.show()