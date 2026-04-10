import logging
import numpy as np
from torch.utils.data import Dataset

from utils.data import Compose, iADSBIQ, iADSBImage


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, aug=1, dataset_kwargs=None):
        self.dataset_name = dataset_name
        self.aug = aug
        self.dataset_kwargs = dataset_kwargs or {}
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_accumulate_tasksize(self, task):
        return sum(self._increments[: task + 1])

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False, m_rate=None):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError(f"Unknown data source {source}.")

        if mode == "train":
            trsf = Compose([*self._train_trsf, *self._common_trsf])
        elif mode in {"test", "flip"}:
            trsf = Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError(f"Unknown mode {mode}.")

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            else:
                class_data, class_targets = self._select_rmm(x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate)
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        if data:
            data, targets = np.concatenate(data), np.concatenate(targets)
        else:
            feature_shape = self._train_data.shape[1:]
            data = np.empty((0, *feature_shape), dtype=self._train_data.dtype)
            targets = np.empty((0,), dtype=self._train_targets.dtype)

        dataset = DummyDataset(data, targets, trsf, aug=self.aug if source == "train" and mode == "train" else 1)
        if ret_data:
            return data, targets, dataset
        return dataset

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, **self.dataset_kwargs)
        idata.download_data()

        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        if m_rate != 0:
            selected_idxes = np.random.randint(0, len(idxes), size=int((1 - m_rate) * len(idxes)))
            idxes = np.sort(idxes[selected_idxes])
        return x[idxes], y[idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, aug=1):
        assert len(images) == len(labels), "Data size error!"
        self.aug = aug
        self.images = images
        self.labels = labels
        self.trsf = trsf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = self.images[idx]
        label = self.labels[idx]
        if self.aug == 1:
            return idx, self.trsf(sample.copy()), label
        views = [self.trsf(sample.copy()) for _ in range(self.aug)]
        return idx, *views, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, **dataset_kwargs):
    name = dataset_name.lower()
    if name == "adsb_iq":
        return iADSBIQ(
            data_root=dataset_kwargs.get("data_root"),
            metadata_file=dataset_kwargs.get("metadata_file"),
            iq_len=dataset_kwargs.get("iq_len", 1024),
            num_channels=dataset_kwargs.get("num_channels", 2),
        )
    if name == "adsb_image":
        return iADSBImage(
            data_root=dataset_kwargs.get("data_root"),
            metadata_file=dataset_kwargs.get("metadata_file"),
            train_data_file=dataset_kwargs.get("train_data_file"),
            train_label_file=dataset_kwargs.get("train_label_file"),
            test_data_file=dataset_kwargs.get("test_data_file"),
            test_label_file=dataset_kwargs.get("test_label_file"),
            image_size=dataset_kwargs.get("image_size", 32),
            num_channels=dataset_kwargs.get("num_channels", 3),
        )
    raise NotImplementedError(f"Unknown dataset {dataset_name}.")
