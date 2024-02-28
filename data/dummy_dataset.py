from torch.utils.data import Dataset


class DummyDataset(Dataset):
    """
    only for debugging the samplers
    """

    def __init__(self, num_samples, mode='train'):
        self.data = list(range(num_samples))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]