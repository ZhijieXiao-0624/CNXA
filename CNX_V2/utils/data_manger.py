# - 处理数据集
import os
import random


class ImageNet100():
    def __init__(self, root='data', **kwargs):
        self.root = root  # 数据集的绝对路径

        self._check_before_run()

        train, val, total, num_classes = self._process_dir()

        train_labels = self._set_label(train)
        val_labels = self._set_label(val)

        self._logs(len(train), len(val), train_labels, val_labels, total, num_classes)
        random.shuffle(train)

        self.train = train
        self.val = val
        self.total = total
        self.num_classes = num_classes

    def _check_before_run(self):
        if not os.path.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))

    def _process_dir(self):
        class_dict = {}  # 将100个种类对应到0~99的字典
        train_data = list()  # 存放训练数据信息 “绝对路径，类别”
        val_data = list()  # 存放验证数据信息 “绝对路径，类别”

        for idex, class_name in enumerate(os.listdir(self.root)):
            class_dict[class_name] = idex
            imgs_path = os.path.join(self.root, class_name)

            for img_path in os.listdir(imgs_path):
                absolute_path = imgs_path + '/' + img_path

                train_data.append((absolute_path, class_dict[class_name]))

        total = len(train_data)  # 总数据量
        classes = len(class_dict)  # 总的类别数

        val_num = round(0.1 * total)  # 验证集的熟练 1：9

        # 从训练集中抽取1：9比例的验证集
        for i in range(val_num):
            train_num = len(train_data)  # 每抽取一次后更新训练集数量
            idnex = random.randint(0, train_num - 1)

            val_data.append(train_data[idnex])
            train_data.pop(idnex)  # 删除已经分配到val里面的数据

        return train_data, val_data, total, classes

    # 打印信息
    def _logs(self, train_num, val_num, train_label, val_label, total, num_classes):
        print("==> ImageNet100 loaded")
        print("Dateset statistics")

        print("   -------------------------------")
        print("   subset  | # labels | # images")
        print("   -------------------------------")
        print("   total   |  {:5d} \t|{:8d}".format(num_classes, total))
        print("   train   |  {:5d} \t|{:8d}".format(train_label, train_num))
        print("   val     |  {:5d} \t|{:8d}".format(val_label, val_num))
        print("   -------------------------------")

    # 提取类别数量
    def _set_label(self, x):
        sava = []
        for i, (path, label) in enumerate(x):
            sava.append(label)

        return len(set(sava))


