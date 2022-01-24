import os
import torch
from torchvision import datasets, transforms


def ImageNet1k(data_path,train=False):
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'validation')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    if train:
        return datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

    else:
        return datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ]))

def get_dataset(args):
    #prepare datasets
    train_dir=os.path.join(args.data_path, 'train')
    test_dir=os.path.join(args.data_path, 'test')

    pathinput=args.data_path

    if args.dataset=='CIFAR100':
        args.data_path='/scratch/CIFAR100'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset=datasets.CIFAR100(args.data_path, train=True, transform=transform_train, target_transform=None, download=True)
        valid_dataset=datasets.CIFAR100(args.data_path, train=True, transform=transform_test, target_transform=None, download=False)
        test_dataset=datasets.CIFAR100(args.data_path, train=False, transform=transform_test, target_transform=None, download=False)
        args.nclass=100
    elif args.dataset=='CIFAR10':
        args.data_path='/scratch/CIFAR10'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset=datasets.CIFAR10(args.data_path, train=True, transform=transform_train, target_transform=None, download=True)
        valid_dataset=datasets.CIFAR10(args.data_path, train=True, transform=transform_test, target_transform=None, download=True)
        test_dataset=datasets.CIFAR10(args.data_path, train=False, transform=transform_test, target_transform=None, download=True)
        args.nclass=10

    elif args.dataset=='SVHN':
        args.data_path='/scratch/SVHN'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        SVHNtrain_dataset=datasets.SVHN(args.data_path, split='train', transform=transform_train, target_transform=None, download=True)
        SVHNextra_dataset=datasets.SVHN(args.data_path, split='extra', transform=transform_train, target_transform=None, download=True)
        train_dataset=torch.utils.data.ConcatDataset([SVHNtrain_dataset,SVHNextra_dataset])
        valid_dataset=datasets.SVHN(args.data_path, split='train', transform=transform_test, target_transform=None, download=True)
        test_dataset=datasets.SVHN(args.data_path, split='test', transform=transform_test, target_transform=None, download=True)
        args.nclass=10
    elif args.dataset=='MNIST':
        args.data_path='/scratch/MNIST'
        train_dataset=datasets.MNIST(args.data_path, train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
        valid_dataset=datasets.MNIST(args.data_path, train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
        test_dataset=datasets.MNIST(args.data_path, train=False, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
        args.nclass=10
    else:
        args.data_path='/scratch/IMAGENET/data-dir/raw-data'
        if pathinput!='None':
            args.data_path=pathinput
        train_dataset=ImageNet1k(args.data_path, train=True)
        valid_dataset=train_dataset
        test_dataset=ImageNet1k(args.data_path, train=False)
        args.nclass=1000

    kwargs = {'num_workers': 16, 'pin_memory': True} if args.cuda else {}
    #train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size, **kwargs)
    return train_loader,valid_loader,test_loader
