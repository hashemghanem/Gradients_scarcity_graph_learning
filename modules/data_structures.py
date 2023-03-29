import torch


class PriorityQ:
    def __init__(self, dim=1, device='cpu'):
        self.dim = dim
        self.device = device
        self.q = torch.empty((0, self.dim), device=self.device)

    def push(self, a):
        # To prevent problems later
        a = a.detach()
        idx = torch.searchsorted(self.q[:, 0], a[0])
        # Check if the element (edge) you enter doesn't exist already.
        if idx < self.q.shape[0]:
            nxt = self.q[idx]
            if nxt[1] == a[1] and nxt[2] == a[2]:
                return
            if nxt[1] == a[2] and nxt[2] == a[1]:
                return
        if idx > 0:
            nxt = self.q[idx-1]
            if nxt[1] == a[1] and nxt[2] == a[2]:
                return
            if nxt[1] == a[2] and nxt[2] == a[1]:
                return
        self.q = torch.vstack([self.q[0:idx], a, self.q[idx:]]).contiguous()

    def pop(self, num=1):
        ret = self.q[-num]
        self.q = self.q[:-num]
        return ret

    def isEmpty(self):
        return self.q.shape[0] == 0

    def crop(self, size_to_keep):
        if self.q.shape[0] <= size_to_keep:
            return
        self.q = self.q[-size_to_keep:]

    def get_q(self):
        return self.q

    def reset(self):
        self.q = torch.empty((0, self.dim), device=self.device)
