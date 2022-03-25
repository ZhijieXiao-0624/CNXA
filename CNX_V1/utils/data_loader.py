# - 继承torch中Dataset，自动吞吐数据量

import os
from PIL import Image

from torch.utils.data import Dataset


def read_images(img_path):
    got_img = False

    if not os.path.exists(img_path):
        raise IOError("'{}' does not exits".format(img_path))

    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            raise IOError("'{}' does not read".format(img_path))

    return img


class ImageNet100Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        super(ImageNet100Dataset, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, label = self.dataset[index]

        img = read_images(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

"""
if __name__ == '__main__':
    dataset = ImageNet100(root='E://self_dataset/imagenet100/imagenet100/')

    train_load = Image100Dataset(dataset.train)

    for batch_id, (img, label) in enumerate(train_load):
        print("{}, {}, {}".format(batch_id, img, label))
"""