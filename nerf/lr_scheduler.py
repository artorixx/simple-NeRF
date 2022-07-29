import torch
import torch.optim as optim

class LrScheduler():
    """ Implements a learning rate schedule with warum up and decay """
    def __init__(self, peak_lr=5e-4, peak_it=0, decay_rate=0.1, decay_it=500*1000):
        self.peak_lr = peak_lr
        self.peak_it = peak_it
        self.decay_rate = decay_rate
        self.decay_it = decay_it

    def get_cur_lr(self, it):
        if it < self.peak_it:  # Warmup period
            return self.peak_lr * (it / self.peak_it)
        it_since_peak = it - self.peak_it
        return self.peak_lr * (self.decay_rate ** (it_since_peak / self.decay_it))