import numpy as np
from torch.utils.data import Dataset, DataLoader


initial_seed = 2021
np.random.seed(initial_seed)


class RandomDataset(Dataset):
    def __getitem__(self, index):
        return np.random.randint(0, 1000, 3)

    def __len__(self):
        return 10


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


dataset = RandomDataset()
dataloader = DataLoader(dataset, batch_size=2,
                        num_workers=2, worker_init_fn=worker_init_fn)

for e in range(2):
    np.random.seed(initial_seed + e)
    for batch in dataloader:
        print(batch)
    print('===='*10)
