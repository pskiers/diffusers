class SafeIterator:
    """
    A wrapper around a DataLoader (or any iterator) that catches and skips
    exceptions during iteration (like corrupted images in ImageNet).
    """
    def __init__(self, iterable, logger=None):
        self.iterable = iterable
        self.iterator = iter(iterable)
        self.logger = logger

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                return next(self.iterator)
            except StopIteration:
                raise
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"SafeIterator caught error: {e}. Skipping batch.")
                else:
                    print(f"SafeIterator caught error: {e}. Skipping batch.")


class LimitedLoader:
    """
    Wraps a DataLoader to stop iteration after a fixed number of batches.
    For 'short epochs' on huge datasets. Should be compatible with multigpu accelerate logic etc. (I hope).
    """
    def __init__(self, dataloader, limit_batches):
        self.dataloader = dataloader
        self.limit_batches = min(limit_batches, len(dataloader))

    def __len__(self):
        return self.limit_batches

    def __iter__(self):
        iterator = iter(self.dataloader)

        for _ in range(self.limit_batches):
            try:
                yield next(iterator)
            except StopIteration:
                break
