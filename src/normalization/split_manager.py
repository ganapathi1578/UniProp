"""Manage dataset splits."""
import random

def create_splits(train_video_ids: list[str], val_ratio: float = 0.1, seed: int = 42):
    """Split training videos into train and val deterministically."""
    rng = random.Random(seed)
    shuffled = sorted(list(train_video_ids))
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    val_ids = set(shuffled[:n_val])
    train_ids = set(shuffled[n_val:])
    return train_ids, val_ids
