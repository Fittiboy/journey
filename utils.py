from matplotlib.axes import Axes


def plot_training_loss(losses: list[float], ax: Axes):
    ax.plot(losses, label="Training loss")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")