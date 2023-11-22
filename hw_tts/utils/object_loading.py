from operator import xor

from torch.utils.data import ConcatDataset, DataLoader

import hw_tts.augmentations
import hw_tts.datasets
from hw_tts.utils.parse_config import ConfigParser
from hw_tts.utils.collate import collate_fn


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == 'train':
            wave_augs, spec_augs = hw_tts.augmentations.from_configs(configs)
            drop_last = True
        else:
            wave_augs, spec_augs = None, None
            drop_last = False

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(configs.init_obj(
                ds, hw_tts.datasets, config_parser=configs,
                wave_augs=wave_augs, spec_augs=spec_augs))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        bs = params["batch_size"]
        shuffle = True

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=collate_fn,
            shuffle=shuffle, num_workers=num_workers, drop_last=drop_last
        )
        dataloaders[split] = dataloader
    return dataloaders
