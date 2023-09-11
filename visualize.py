import numpy as np
import cv2
import torch
from dataset import Endoscope, HpMutiClsTrain, HpMutiClsTest
from models.swin_mlp import SwinMLP_ImgGroupDiagnosis
from torch.utils.data import DataLoader
from util import processbar


gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SwinMLP_ImgGroupDiagnosis(
    n_class=4, in_chans=3,
    dims=(64, 128, 256, 512), depths=(1, 1, 3, 1),
    w_sizes=(8, 8, 8, 6), img_token_dim=256,
    batch_forward=False, forward_batch_size=16,
    attn_layers=6
)
model.to(gpu)
model.load_state_dict(torch.load("./params/swin-hp-muti-cls-heatmap.pth"))
# model.load_state_dict(torch.load("./params/swin-hp-muti-cls.pth"))
model.eval()

endscope_dataset = Endoscope(
    roots=["C:/Users/Administrator.DESKTOP-DDF6IV7/Desktop/HPMutiCls/1",
           "C:/Users/Administrator.DESKTOP-DDF6IV7/Desktop/HPMutiCls/2",
           "C:/Users/Administrator.DESKTOP-DDF6IV7/Desktop/HPMutiCls/3",
           "C:/Users/Administrator.DESKTOP-DDF6IV7/Desktop/HPMutiCls/4"],
    labels=[0, 1, 2, 3],
    use_cache=True
)
train_set, test_set = HpMutiClsTrain(endscope_dataset), HpMutiClsTest(endscope_dataset)
train_loader, test_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True), DataLoader(dataset=test_set, batch_size=1, shuffle=False)


def evaluate():
    processed = 0
    acc, acc1, acc2, acc3, acc4 = 0, 0, 0, 0, 0
    cls_correct_num, cls_num = [0, 0, 0, 0], [0, 0, 0, 0]
    with torch.no_grad():
        for imgs, label in test_loader:
            imgs, label = imgs[0].to(gpu), label.to(gpu)

            pred, imgs_weights, heatmaps = model(imgs)

            cls_num[label.item()] += 1
            cls_correct_num[label.item()] += int(pred.argmax(dim=1).item() == label.item())

            acc1 = 0 if cls_num[0] == 0 else cls_correct_num[0] / cls_num[0]
            acc2 = 0 if cls_num[1] == 0 else cls_correct_num[1] / cls_num[1]
            acc3 = 0 if cls_num[2] == 0 else cls_correct_num[2] / cls_num[2]
            acc4 = 0 if cls_num[3] == 0 else cls_correct_num[3] / cls_num[3]
            acc = (acc1 + acc2 + acc3 + acc4) / 4

            processed += 1

            print(
                "\r测试进度：%s  acc1: %.5f  acc2: %.5f  acc3: %.5f  acc4: %.5f  acc: %.5f" % (
                    processbar(processed, len(test_set)),
                    acc1, acc2, acc3, acc4, acc
                ), end="")
        print()
    return acc


def visualize_heatmap():
    processed = 0
    acc, acc1, acc2, acc3, acc4 = 0, 0, 0, 0, 0
    cls_correct_num, cls_num = [0, 0, 0, 0], [0, 0, 0, 0]
    topk = 3
    with torch.no_grad():
        for imgs, label in test_loader:
            imgs, label = imgs[0].to(gpu), label.to(gpu)

            pred, imgs_weights, heatmaps = model(imgs)
            # visualize
            important_imgs_inds = imgs_weights.topk(k=topk, largest=True, dim=0)[1]
            imgs, heatmaps = imgs[important_imgs_inds], heatmaps[important_imgs_inds]
            for i in range(imgs.shape[0]):
                img_i = (imgs[i] * torch.Tensor([0.3121, 0.2028, 0.1720]).view(3, 1, 1).to(gpu) + torch.Tensor([0.4658, 0.2411, 0.2019]).view(3, 1, 1).to(gpu)) * 255
                img_i = img_i.permute([1, 2, 0]).cpu().numpy().astype(np.uint8)
                img_i = cv2.cvtColor(img_i, cv2.COLOR_RGB2BGR)

                heatmap_i = heatmaps[i].cpu().numpy()
                heatmap_i = heatmap_i / np.max(heatmap_i)
                heatmap_i = cv2.resize(heatmap_i, dsize=(img_i.shape[1], img_i.shape[0]))
                heatmap_i = cv2.applyColorMap(np.uint8(255 * heatmap_i), cv2.COLORMAP_JET)
                heatmap_img_i = cv2.addWeighted(img_i, 1, heatmap_i, 0.6, gamma=1)

                img_i = np.concatenate([img_i, heatmap_img_i, heatmap_i], axis=1)

                cv2.imshow("heatmap", img_i)
                cv2.waitKey(0)

            cls_num[label.item()] += 1
            cls_correct_num[label.item()] += int(pred.argmax(dim=1).item() == label.item())

            acc1 = 0 if cls_num[0] == 0 else cls_correct_num[0] / cls_num[0]
            acc2 = 0 if cls_num[1] == 0 else cls_correct_num[1] / cls_num[1]
            acc3 = 0 if cls_num[2] == 0 else cls_correct_num[2] / cls_num[2]
            acc4 = 0 if cls_num[3] == 0 else cls_correct_num[3] / cls_num[3]
            acc = (acc1 + acc2 + acc3 + acc4) / 4

            processed += 1

            print(
                "\r测试进度：%s  acc1: %.5f  acc2: %.5f  acc3: %.5f  acc4: %.5f  acc: %.5f" % (
                    processbar(processed, len(test_set)),
                    acc1, acc2, acc3, acc4, acc
                ), end="")
        print()
    return acc


if __name__ == '__main__':
    visualize_heatmap()