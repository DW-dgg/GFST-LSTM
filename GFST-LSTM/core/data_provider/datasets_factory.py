from torch.utils.data import DataLoader
from core.data_provider.radar import Radar


def data_provider(dataset, configs, data_train_path, data_test_path, batch_size,
                  is_training=True,
                  is_shuffle=True):
    if is_training:
        num_workers = configs.num_workers  # 2
        root = data_train_path
    else:
        num_workers = 0
        root = data_test_path
    if dataset == "radar":
        dataset = Radar(root=root, is_train=is_training, factor=configs.factor, input_length=configs.input_length)

    return DataLoader(dataset,
                      pin_memory=True,
                      batch_size=batch_size,
                      shuffle=is_shuffle,
                      num_workers=num_workers)

