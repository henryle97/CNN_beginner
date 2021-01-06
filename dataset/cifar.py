from torchvision.datasets import cifar


def get_cifar_dataset(is_train=True, transform=None):
    cifar_dataset = cifar.CIFAR10(root="C:\\Users\\hoanglv10\\PycharmProjects\\CNN\dataset\\data", download=True, train=is_train, transform=transform)

    return cifar_dataset


if __name__ == "__main__":
    train_data = get_cifar_dataset(is_train=True)
    test_data  = get_cifar_dataset(is_train=False)
    print(train_data)