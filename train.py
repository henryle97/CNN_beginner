import time
from dataset.cifar import get_cifar_dataset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision

from utils.utils import AverageMeter, ProgressMeter, accuracy

# CONFIG
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
PRINT_FREQ = 2


def main():

    # DATA
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = get_cifar_dataset(is_train=True, transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_data = get_cifar_dataset(is_train=False, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    n_classes = len(train_data.classes)
    print("Class: ", train_data.classes)

    # MODEL
    model = torchvision.models.vgg19(pretrained=False, num_classes=n_classes)

    # LOSS + OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
        model = model.cuda()

    best_acc = -1e3
    for epoch in range(1, EPOCHS+1):
        train(train_loader, model, criterion, optimizer, epoch)
        acc1 = valid(test_loader, model, criterion, epoch)
        if best_acc < acc1:
            best_acc = acc1
            save_weight(model, "best_weight.pt")

        save_checkpoint(model, optimizer, epoch, "last_model.pt")

    print(best_acc)


def save_weight(model, weight_path):
    torch.save(model.state_dict(), weight_path)


def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch

    }, checkpoint_path)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):

        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            target = target.cuda()  # non_blocking

        # compute output
        output = model(inputs)
        loss = criterion(output, target)   # reduction='mean' --> mean loss for per sample

        # computer gradient and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy + loss

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            progress.display(i)


def valid(test_loader, model, criterion, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.eval()
    end = time.time()
    with torch.no_grad():  # reduce memory consumption for computations
        for i, (inputs, target) in enumerate(test_loader):
            data_time.update(time.time()-end)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                target = target.cuda()

            output = model(inputs)
            loss = criterion(output, target)

            losses.update(loss.item(), inputs.size(0))
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # measure batch time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % PRINT_FREQ == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


if __name__ == "__main__":
    main()
