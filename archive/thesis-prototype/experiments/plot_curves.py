import torch, torch.nn as nn, torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np, matplotlib.pyplot as plt

# ----------------------- DATA -----------------------
X, y = make_moons(n_samples=1200, noise=0.25, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val)

# ----------------------- MODEL -----------------------
class Net(nn.Module):
    def __init__(self, hidden=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )
    def forward(self, x): return self.layers(x)

def accuracy(model, X, y):
    return (model(X).argmax(1) == y).float().mean().item()

# ----------------------- TRAIN SEVERAL NETS -----------------------
def train_net(seed):
    torch.manual_seed(seed)
    model = Net(hidden=np.random.randint(6,30))
    opt = optim.Adam(model.parameters(), lr=np.random.uniform(1e-3,1e-2))
    crit = nn.CrossEntropyLoss()
    val_accs = []
    for epoch in range(40):
        opt.zero_grad()
        loss = crit(model(X_train), y_train)
        loss.backward(); opt.step()
        val_accs.append(accuracy(model, X_val, y_val))
    return np.array(val_accs)

curves = [train_net(seed) for seed in range(10)]
epochs = np.arange(len(curves[0]))

# ----------------------- FORECAST COMPONENTS -----------------------
def rational(x,a=0.95,b=5): return (a*x)/(b+x)
def sigmoid(x,L=0.98,k=6,x0=20): return L/(1+np.exp(-k*(x-x0)/len(epochs)))
def linear(x,m=0.02,c=0.6): return np.clip(m*x+c,0,1)

# ----------------------- SHAPE-AWARE MORPHING -----------------------
def morph_forecast(x, y):
    """Blend rational, sigmoid, and linear using slope/curvature awareness."""
    slope = np.gradient(y, x)
    curvature = np.gradient(slope, x)
    y_r, y_s, y_l = rational(x), sigmoid(x), linear(x)

    wr = 1/(1+np.exp(-5*np.abs(slope))) * (1 - 1/(1+np.exp(-5*np.abs(curvature))))
    ws = 1/(1+np.exp(-3*curvature))
    wl = np.maximum(0.0, 1 - (wr+ws))
    total = wr+ws+wl
    wr, ws, wl = wr/total, ws/total, wl/total
    y_morph = wr*y_r + ws*y_s + wl*y_l
    return y_morph, wr, ws, wl

example_curve = curves[0]
y_morph, wr, ws, wl = morph_forecast(epochs, example_curve)

# ----------------------- PLOTS -----------------------
fig, axs = plt.subplots(2, 2, figsize=(13,7))
axs = axs.flatten()

# Panel A: Real neural curves
for y in curves: axs[0].plot(epochs, y, color="gray", alpha=0.5)
axs[0].set_title("(A) Real Neural Learning Curves")
axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Validation Accuracy")
axs[0].set_ylim(0,1)

# Panel B: Component models
x = np.linspace(0, len(epochs)-1, len(epochs))
axs[1].plot(x, rational(x), 'b', lw=2, label="Rational")
axs[1].plot(x, sigmoid(x), 'orange', lw=2, label="Sigmoid")
axs[1].plot(x, linear(x), 'green', lw=2, label="Linear")
axs[1].legend(); axs[1].set_ylim(0,1)
axs[1].set_title("(B) Forecast Components")

# Panel C: Morphing forecast vs observed
axs[2].plot(epochs, example_curve, color="gray", alpha=0.6, label="Observed Curve")
axs[2].plot(epochs, y_morph, 'k--', lw=3, label="Morphing Forecast")
axs[2].legend(); axs[2].set_ylim(0,1)
axs[2].set_title("(C) Shape-Aware Morphing Forecast")

# Panel D: Weight evolution
axs[3].plot(epochs, wr, 'b', label="Weight Rational")
axs[3].plot(epochs, ws, 'orange', label="Weight Sigmoid")
axs[3].plot(epochs, wl, 'green', label="Weight Linear")
axs[3].set_ylim(0,1)
axs[3].set_title("(D) Adaptive Morphing Weights")
axs[3].set_xlabel("Epoch"); axs[3].legend()

plt.tight_layout()
plt.show()
