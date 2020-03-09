import torch
from torch.utils.data.sampler import BatchSampler


class LenMatchBatchSampler(BatchSampler):
    """
    Custom PyTorch Sampler that generate batches of similar length.
    Used alongside with trim_tensor, it helps speed up training.
    """
    def __iter__(self):

        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            count_zeros = torch.sum(self.sampler.data_source[idx][0] == 0)
            count_zeros = int(count_zeros / 64) 
            if len(buckets[count_zeros]) == 0:  buckets[count_zeros] = []

            buckets[count_zeros].append(idx)

            if len(buckets[count_zeros]) == self.batch_size:
                batch = list(buckets[count_zeros])
                yield batch
                yielded += 1
                buckets[count_zeros] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

        assert len(self) == yielded, "produced an inccorect number of batches. expected %i, but yielded %i" %(len(self), yielded)



def trim_tensors(tokens, min_len=10):
    """
    Trim tensors so that within a batch, padding is shortened.
    This speeds up training for RNNs and Transformers
    
    Arguments:
        tokens {torch tensor} -- Text tokens
    
    Keyword Arguments:
        min_len {int} -- Minimum length to trim to (default: {10})
    
    Returns:
        torch tensor -- trimmed tokens
    """
    max_len = max(torch.max(torch.sum((tokens != 0 ), 1)), min_len)
    return tokens[:, :max_len]