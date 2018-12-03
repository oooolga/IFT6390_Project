import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)

class EMNIST(MNIST):
    """`EMNIST <https://www.nist.gov/itl/iad/image-group/emnist-dataset/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        split (string): The dataset has 6 different splits: ``byclass``, ``bymerge``,
            ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies
            which one to use.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
    splits = ('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')

    def __init__(self, root, split, **kwargs):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        self.split = split
        self.training_file = self._training_file(split)
        self.test_file = self._test_file(split)
        super(EMNIST, self).__init__(root, **kwargs)

    def _training_file(self, split):
        return 'training_{}.pt'.format(split)

    def _test_file(self, split):
        return 'test_{}.pt'.format(split)

    def download(self):
        """Download the EMNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip
        import shutil
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print('Downloading ' + self.url)
        data = urllib.request.urlopen(self.url)
        filename = self.url.rpartition('/')[2]
        raw_folder = os.path.join(self.root, self.raw_folder)
        file_path = os.path.join(raw_folder, filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print('Extracting zip archive')
        with zipfile.ZipFile(file_path) as zip_f:
            zip_f.extractall(raw_folder)
        os.unlink(file_path)
        gzip_folder = os.path.join(raw_folder, 'gzip')
        for gzip_file in os.listdir(gzip_folder):
            if gzip_file.endswith('.gz'):
                print('Extracting ' + gzip_file)
                with open(os.path.join(raw_folder, gzip_file.replace('.gz', '')), 'wb') as out_f, \
                        gzip.GzipFile(os.path.join(gzip_folder, gzip_file)) as zip_f:
                    out_f.write(zip_f.read())
        shutil.rmtree(gzip_folder)

        # process and save as torch files
        for split in self.splits:
            print('Processing ' + split)
            training_set = (
                read_image_file(os.path.join(raw_folder, 'emnist-{}-train-images-idx3-ubyte'.format(split))),
                read_label_file(os.path.join(raw_folder, 'emnist-{}-train-labels-idx1-ubyte'.format(split)))
            )
            test_set = (
                read_image_file(os.path.join(raw_folder, 'emnist-{}-test-images-idx3-ubyte'.format(split))),
                read_label_file(os.path.join(raw_folder, 'emnist-{}-test-labels-idx1-ubyte'.format(split)))
            )
            with open(os.path.join(self.root, self.processed_folder, self._training_file(split)), 'wb') as f:
                torch.save(training_set, f)
            with open(os.path.join(self.root, self.processed_folder, self._test_file(split)), 'wb') as f:
                torch.save(test_set, f)

        print('Done!')


def load_data(batch_size, test_batch_size, data_type):
	datasets = {'CIFAR': load_cifar100_data,
				'EMNIST': load_EMNIST_data,
				'FMNIST': load_fashionMNIST_data}
	return datasets[data_type](batch_size, test_batch_size)

def load_cifar100_data(batch_size, test_batch_size, alpha=0.8):

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


	train_data = dset.CIFAR10(root='data', train=True, download=True, transform=transform_train)
	train_len = int(len(train_data)*alpha)
	train_data.train_data = train_data.train_data[:train_len]
	train_data.train_labels = train_data.train_labels[:train_len]
	valid_data = dset.CIFAR10(root='data', train=True, download=True, transform=transform_train)
	valid_data.train_data = valid_data.train_data[train_len:]
	valid_data.train_labels = valid_data.train_labels[train_len:]

	test_data = dset.CIFAR10(root='data', train=False, download=True, transform=transform_test)

	train_loader = torch.utils.data.DataLoader(
		train_data, batch_size=batch_size, shuffle=True, drop_last=True)
	valid_loader = torch.utils.data.DataLoader(
		valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
	test_loader = torch.utils.data.DataLoader(
		test_data,
		batch_size=test_batch_size, shuffle=True, drop_last=True)

	return train_loader, valid_loader, test_loader

def load_fashionMNIST_data(batch_size, test_batch_size):
	train_data = dset.FashionMNIST('data', train=True, download=True, transform=transforms.ToTensor())
	train_data.train_data = train_data.train_data[:50000]
	train_data.train_labels = train_data.train_labels[:50000]
	valid_data = dset.FashionMNIST('data', train=True, download=True, transform=transforms.ToTensor())
	valid_data.train_data = valid_data.train_data[50000:]
	valid_data.train_labels = valid_data.train_labels[50000:]
	train_loader = torch.utils.data.DataLoader(
		train_data, batch_size=batch_size, shuffle=True, drop_last=True)
	valid_loader = torch.utils.data.DataLoader(
		valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
	test_loader = torch.utils.data.DataLoader(
		dset.FashionMNIST('data', train=False, download=True, transform=transforms.ToTensor()),
		batch_size=test_batch_size, shuffle=True, drop_last=True)

	return train_loader, valid_loader, test_loader


def load_EMNIST_data(batch_size, test_batch_size, alpha=0.9):
	train_data = EMNIST(root='data', train=True, download=True, 
						split='balanced',
						transform=transforms.ToTensor())
	train_len = int(len(train_data)*alpha)
	train_data.train_data = train_data.train_data[:train_len]
	train_data.train_labels = train_data.train_labels[:train_len]
	valid_data = EMNIST(root='data', train=True, download=True,
						split='balanced',
						transform=transforms.ToTensor())
	valid_data.train_data = valid_data.train_data[train_len:]
	valid_data.train_labels = valid_data.train_labels[train_len:]

	test_data = EMNIST(root='data', train=False, download=True,
					   split='balanced',
					   transform=transforms.ToTensor())

	train_loader = torch.utils.data.DataLoader(
		train_data, batch_size=batch_size, shuffle=True, drop_last=True)
	valid_loader = torch.utils.data.DataLoader(
		valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
	test_loader = torch.utils.data.DataLoader(
		test_data,
		batch_size=test_batch_size, shuffle=True, drop_last=True)

	return train_loader, valid_loader, test_loader