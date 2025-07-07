import numpy as np
from typing import Iterator, Tuple, Optional, Union, List

from deep_learning.tensor import Tensor


class Dataset:
    def __init__(self, inputs: Union[np.ndarray, List], targets: Union[np.ndarray, List]):
        """Initialize the dataset with inputs and targets."""
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(targets)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.inputs)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Get a single sample from the dataset."""
        return Tensor(self.inputs[index]), Tensor(self.targets[index])


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False, drop_last: bool = False, random_state: Optional[int] = None):
        """Initialize the DataLoader with a dataset, batch size, and options for shuffling and dropping the last batch."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def __len__(self) -> int:
        """Return the number of batches in the DataLoader."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[Tuple[Tensor, ...]]:
        """Yield batches of data from the dataset."""
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            
            batch_samples = [self.dataset[idx] for idx in batch_indices]
            
            if len(batch_samples) > 0:
                num_elements = len(batch_samples[0])
                
                batched_elements = []
                for element_idx in range(num_elements):
                    element_data = [sample[element_idx].data for sample in batch_samples]
                    stacked_data = np.stack(element_data, axis=0)
                    batched_elements.append(Tensor(stacked_data))
                
                yield tuple(batched_elements)
