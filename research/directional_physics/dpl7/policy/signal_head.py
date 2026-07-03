import numpy as np


class LinearICHead:
    def __init__(self, dim=8, lr=0.001):
        self.w = np.zeros(dim)
        self.b = 0.0
        self.lr = lr

    def forward(self, z):
        score = float(np.dot(self.w, z) + self.b)
        direction = float(np.tanh(score))
        confidence = float(abs(direction))
        return {"score": score, "direction": direction, "confidence": confidence}

    def update(self, z_batch, y_batch):
        z = np.array(z_batch)
        y = np.array(y_batch)
        z_c = z - z.mean(axis=0)
        y_c = y - y.mean()
        grad_w = np.dot(z_c.T, y_c) / (len(y) + 1e-8)
        grad_b = float(np.mean(y_c))
        self.w += self.lr * grad_w
        self.b += self.lr * grad_b
        self.w = np.clip(self.w, -5.0, 5.0)
        pred = np.dot(z, self.w) + self.b
        ic = float(np.corrcoef(pred, y)[0, 1]) if len(y) > 2 else 0.0
        return ic if not np.isnan(ic) else 0.0

    def fit(self, z_batch, y_batch, epochs=5):
        best_ic = -1.0
        best_w = self.w.copy()
        best_b = self.b
        for epoch in range(epochs):
            ic = self.update(z_batch, y_batch)
            if ic > best_ic:
                best_ic = ic
                best_w = self.w.copy()
                best_b = self.b
        self.w = best_w
        self.b = best_b
        return best_ic

    def predict(self, z):
        return self.forward(z)

    def reset(self):
        self.w = np.zeros_like(self.w)
        self.b = 0.0
