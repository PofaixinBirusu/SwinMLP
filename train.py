import numpy as np
import torch
from dataset import Endoscope, HpMutiClsTrain, HpMutiClsTest
from models.swin_mlp import SwinMLP_ImgGroupDiagnosis
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from util import processbar
from torch.cuda.amp import autocast, GradScaler

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SwinMLP_ImgGroupDiagnosis(
    n_class=4, in_chans=3,
    dims=(64, 128, 256, 512), depths=(1, 1, 3, 1),
    w_sizes=(8, 8, 8, 6), img_token_dim=256,
    batch_forward=False, forward_batch_size=16,
    attn_layers=6
)
model.to(gpu)
# model.load_state_dict(torch.load("./params/swin-hp-muti-cls-heatmap.pth"))

epoch = 40
lr = 0.0001
min_learning_rate = 0.00001
lr_update_step = 5

accumulation_steps = 12

save_path = "params/swin-hp-muti-cls-heatmap.pth"
# optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=0)
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
loss_fn = CrossEntropyLoss()

endscope_dataset = Endoscope(
    roots=["F:/传/HPMutiCls/1",
           "F:/传/HPMutiCls/2",
           "F:/传/HPMutiCls/3",
           "F:/传/HPMutiCls/4"],
    labels=[0, 1, 2, 3],
    use_cache=True
)
train_set, test_set = HpMutiClsTrain(endscope_dataset), HpMutiClsTest(endscope_dataset)
train_loader, test_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True), DataLoader(dataset=test_set, batch_size=1, shuffle=False)

scaler = GradScaler()


def update_lr(optimizer, gamma=0.5):
    global lr
    lr = max(lr*gamma, min_learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("lr update finished  cur lr: %.5f" % lr)


def evaluate():
    model.eval()
    loss_val = 0
    processed = 0
    acc, acc1, acc2, acc3, acc4 = 0, 0, 0, 0, 0
    cls_correct_num, cls_num = [0, 0, 0, 0], [0, 0, 0, 0]
    with torch.no_grad():
        for imgs, label in test_loader:
            imgs, label = imgs[0].to(gpu), label.to(gpu)

            pred, imgs_weights, heatmaps = model(imgs)
            loss = loss_fn(pred, label)

            loss_val += loss.item()

            cls_num[label.item()] += 1
            cls_correct_num[label.item()] += int(pred.argmax(dim=1).item() == label.item())

            acc1 = 0 if cls_num[0] == 0 else cls_correct_num[0] / cls_num[0]
            acc2 = 0 if cls_num[1] == 0 else cls_correct_num[1] / cls_num[1]
            acc3 = 0 if cls_num[2] == 0 else cls_correct_num[2] / cls_num[2]
            acc4 = 0 if cls_num[3] == 0 else cls_correct_num[3] / cls_num[3]
            acc = (acc1 + acc2 + acc3 + acc4) / 4

            processed += 1

            print(
                "\r测试进度：%s  loss: %.5f  acc1: %.5f  acc2: %.5f  acc3: %.5f  acc4: %.5f  acc: %.5f" % (
                    processbar(processed, len(test_set)), loss.item(),
                    acc1, acc2, acc3, acc4, acc
                ), end="")
        print()
    return acc


def train():
    best_acc = 0

    for epoch_count in range(1, epoch + 1):
        loss_val = 0
        processed = 0
        acc, acc1, acc2, acc3, acc4 = 0, 0, 0, 0, 0
        cls_correct_num = [0, 0, 0, 0]
        cls_num = [0, 0, 0, 0]

        model.train()
        optimizer.zero_grad()
        for imgs, label in train_loader:
            imgs, label = imgs[0].to(gpu), label.to(gpu)
            with autocast():
                pred, imgs_weights, heatmaps = model(imgs)
                loss = loss_fn(pred, label)

            loss_val += loss.item()
            loss = loss / accumulation_steps

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            scaler.scale(loss).backward()
            if (processed + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            processed += 1

            cls_num[label.item()] += 1
            cls_correct_num[label.item()] += int(pred.argmax(dim=1).item() == label.item())

            acc1 = 0 if cls_num[0] == 0 else cls_correct_num[0] / cls_num[0]
            acc2 = 0 if cls_num[1] == 0 else cls_correct_num[1] / cls_num[1]
            acc3 = 0 if cls_num[2] == 0 else cls_correct_num[2] / cls_num[2]
            acc4 = 0 if cls_num[3] == 0 else cls_correct_num[3] / cls_num[3]
            acc = (acc1 + acc2 + acc3 + acc4) / 4

            print(
                "\r进度：%s  loss: %.5f  acc1: %.5f  acc2: %.5f  acc3: %.5f  acc4: %.5f  acc: %.5f" % (
                    processbar(processed, len(train_set)), loss.item(),
                    acc1, acc2, acc3, acc4, acc
                ), end="")
        print("\nepoch: %d  loss: %.5f  acc1: %.5f  acc2: %.5f  acc3: %.5f  acc4: %.5f  acc: %.5f" % (
            epoch_count, loss_val, acc1, acc2, acc3, acc4, acc)
        )

        acc = evaluate()

        if acc > best_acc:
            best_acc = acc
            print("save...")
            torch.save(model.state_dict(), save_path)
            print("save finish !!")

        if epoch_count % lr_update_step == 0:
            update_lr(optimizer, 0.5)


if __name__ == '__main__':
    train()
    # evaluate()
    x = torch.Tensor([1, 2, 3])
