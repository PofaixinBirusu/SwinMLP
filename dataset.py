import torch
from torch.utils import data
from util import extract_big_pic_roi, pil2cv, img_pad, cv2pil
import os
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np


# tensor([0.4658, 0.2411, 0.2019])
class Endoscope(data.Dataset):
    def __init__(self, roots, labels, use_cache=False):
        super(Endoscope, self).__init__()
        self.imgs, self.labels = [], []
        if not use_cache:
            for i, root in enumerate(roots):
                path = root
                for _, y, filelist in os.walk(root):
                    # print(y[0], filelist)
                    names = []
                    if len(y) > 0:
                        if len(y) == 1:
                            path += "/"+y[0]
                        else:
                            names += [path+"/"+name for name in y]
                    if len(names) > 0:
                        # print(names)
                        for name in names:
                            for _, _, x in os.walk(name+"/"+"图像/"):
                                self.imgs.append([name+"/"+"图像/"+img for img in x if img.startswith("IMG") and img.endswith(".png")])
                                self.labels.append(labels[i])
                                break
        else:
            self.labels = [0]*15 + [1]*64 + [2]*141 + [3]*145
        # print(self.imgs)
        # 随机取100个病人吧，电脑跑不动那么多
        # idx = np.random.permutation(len(self.imgs))[:100]
        # np.save("idx100.npy", idx)
        # idx = np.load("idx100.npy")
        # self.imgs = [self.imgs[idx[i]] for i in range(idx.shape[0])]
        # self.labels = [self.labels[idx[i]] for i in range(idx.shape[0])]
        cnt = [0 for _ in range(len(set(labels)))]
        for i in range(len(self.labels)):
            cnt[self.labels[i]] += 1
        print(cnt)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.use_cache = use_cache
        self.to_pil = transforms.Compose([
            transforms.ToPILImage()
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        if self.use_cache:
            imgs, label = torch.Tensor(np.load("dataset_cache/%d.npy" % index)), self.labels[index]
            pil_imgs = [self.to_pil(imgs[i]) for i in range(imgs.shape[0])]
            return pil_imgs, label

        imgs_of_patient = self.imgs[index]
        # imgs_of_patient = imgs_of_patient[(len(imgs_of_patient)-32)//2:(len(imgs_of_patient)-32)//2+32]
        # print(len(imgs_of_patient))
        imgs = []
        for path in imgs_of_patient:
            img = Image.open(path)
            img = pil2cv(img)
            h, w = img.shape[0], img.shape[1]
            # print(h, w)
            if h > 1000 or w > 1000:
                img = extract_big_pic_roi(img)
            if img is not None:
                img = cv2pil(img_pad(img, 640, 640, 0))
                img = self.transform(img)
                imgs.append(img)
            else:
                print(path)
        # n x 3 x h x w
        imgs = torch.stack(imgs, dim=0)
        return imgs, self.labels[index]


class HpMutiClsTrain(data.Dataset):
    def __init__(self, ds):
        self.inds = np.load("dataset_train_test_split/train_idx.npy")
        # self.inds = np.concatenate([np.load("dataset_train_test_split/train_idx.npy"), np.load("dataset_train_test_split/test_idx.npy")], axis=0)
        self.ds = ds
        self.transform = transforms.Compose([
            transforms.Resize(384),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # mean: (0.4658, 0.2411, 0.2019)
            # var : (0.0974, 0.0411, 0.0296)
            # std : (0.3121, 0.2028, 0.1720)
            transforms.Normalize(mean=(0.4658, 0.2411, 0.2019), std=(0.3121, 0.2028, 0.1720))
        ])

    def __len__(self):
        return self.inds.shape[0]

    def __getitem__(self, item):
        item = self.inds[item]
        imgs, label = self.ds[item]
        imgs = torch.stack([self.transform(imgs[i]) for i in range(len(imgs))], dim=0)

        return imgs, label


class HpMutiClsTest(data.Dataset):
    def __init__(self, ds):
        self.inds = np.load("dataset_train_test_split/test_idx.npy")
        self.ds = ds
        self.transform = transforms.Compose([
            transforms.Resize(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4658, 0.2411, 0.2019), std=(0.3121, 0.2028, 0.1720))
        ])

    def __len__(self):
        return self.inds.shape[0]

    def __getitem__(self, item):
        item = self.inds[item]
        imgs, label = self.ds[item]
        imgs = torch.stack([self.transform(imgs[i]) for i in range(len(imgs))], dim=0)

        return imgs, label


if __name__ == '__main__':
    # get cache and std
    endscope_dataset = Endoscope(
        roots=["C:/Users/Administrator.DESKTOP-DDF6IV7/Desktop/HPMutiCls/1",
               "C:/Users/Administrator.DESKTOP-DDF6IV7/Desktop/HPMutiCls/2",
               "C:/Users/Administrator.DESKTOP-DDF6IV7/Desktop/HPMutiCls/3",
               "C:/Users/Administrator.DESKTOP-DDF6IV7/Desktop/HPMutiCls/4"],
        labels=[0, 1, 2, 3],
        use_cache=True
    )
    to_pil = transforms.Compose([
        transforms.ToPILImage()
    ])
    x = 0
    cnt = 0
    means = torch.Tensor([0.4658, 0.2411, 0.2019])
    for i in range(0, len(endscope_dataset)):
        # n x 3 x h x w
        imgs, label = endscope_dataset[i]
        # for j in range(imgs.shape[0]):
        #     img = to_pil(imgs[j])
        #     img = pil2cv(img)
        #     cv2.imshow("", img)
        #     cv2.waitKey(0)
        cnt += (imgs.shape[0]*imgs.shape[2]*imgs.shape[3])
        # x = x + imgs.sum(dim=0).sum(dim=1).sum(dim=1)
        x = x + ((imgs.permute([0, 2, 3, 1]).contiguous().view(-1, 3) - means.view(1, 3)) ** 2).sum(dim=0)
        # print(label)
        np.save("./dataset_cache/%d.npy" % i, imgs.numpy())
        print("\r%d / %d" % (i+1, len(endscope_dataset)), end="")
    print()
    # var
    var = x / cnt
    std = torch.sqrt(var)
    print(var, std)

    cls_0_idx = np.random.permutation(15)
    # 12:3
    cls_0_train_idx, cls_0_test_idx = cls_0_idx[:12], cls_0_idx[12:]

    cls_1_idx = np.random.permutation(64) + 15
    # 52:12
    cls_1_train_idx, cls_1_test_idx = cls_1_idx[:52], cls_1_idx[52:]

    cls_2_idx = np.random.permutation(141) + 15 + 64
    # 113:28
    cls_2_train_idx, cls_2_test_idx = cls_2_idx[:113], cls_2_idx[113:]

    cls_3_idx = np.random.permutation(145) + 15 + 64 + 141
    # 116:29
    cls_3_train_idx, cls_3_test_idx = cls_3_idx[:116], cls_3_idx[116:]

    # # split train and test
    # train_idx = np.concatenate([cls_0_train_idx, cls_0_train_idx, cls_0_train_idx, cls_0_train_idx, cls_0_train_idx, cls_0_train_idx, cls_0_train_idx, cls_0_train_idx, cls_0_train_idx, cls_0_train_idx, cls_1_train_idx, cls_1_train_idx, cls_2_train_idx, cls_3_train_idx], axis=0)
    # test_idx = np.concatenate([cls_0_test_idx, cls_1_test_idx, cls_2_test_idx, cls_3_test_idx], axis=0)
    #
    # np.save("./dataset_train_test_split/train_idx.npy", train_idx)
    # np.save("./dataset_train_test_split/test_idx.npy", test_idx)