
import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Sentinel pushed into the queue to tell the plot thread to exit.
_STOP = object()


@dataclass
class _RunSeries:
    """Accumulated epoch data for one training run."""
    iteration: int
    epochs:     list[int]   = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss:   list[float] = field(default_factory=list)
    val_f1:     list[float] = field(default_factory=list)


# Colour cycle — one colour per run, loops if > 10 runs
_COLOURS = [
    "#4C9BE8", "#E8834C", "#4CE87A", "#E84C4C",
    "#C44CE8", "#E8D84C", "#4CE8D8", "#E84CAB",
    "#8BE84C", "#4C4CE8",
]


class LivePlot:
    """
    Starts a background thread that maintains a live matplotlib window.
    Call update() from the main thread for every epoch. Call close() when
    training finishes to cleanly shut the window.

    Usage:
        plot = LivePlot()
        plot.new_run(iteration=1)
        for epoch_metrics in runner.execute(proposal):
            plot.update(epoch_metrics)
        plot.close()
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._available = False
        self._start()

    # ── public ────────────────────────────────────────────────────────────────

    def new_run(self, iteration: int) -> None:
        """Signal that a new training run is starting."""
        if self._available:
            self._queue.put(("new_run", iteration))

    def update(self, epoch_data: dict) -> None:
        """
        Feed one epoch's metrics to the plot. epoch_data must have keys:
        epoch, train_loss, val_loss, val_f1.
        """
        if self._available:
            self._queue.put(("epoch", epoch_data))

    def close(self) -> None:
        """Shut down the background thread and close the window."""
        if self._available and self._thread is not None:
            self._queue.put(_STOP)
            self._thread.join(timeout=5)
            self._available = False

    # ── private ───────────────────────────────────────────────────────────────

    def _start(self) -> None:
        self._ready = threading.Event()  # add this to __init__ too
        self._thread = threading.Thread(
            target=self._plot_loop,
            daemon=True,
            name="live-plot",
        )
        self._thread.start()
        self._ready.wait(timeout=3.0)  # blocks until plot confirms it's up
        # or gives up after 3s if no display

    def _plot_loop(self) -> None:
        try:
            import matplotlib
            matplotlib.use("TkAgg")          # works on Ubuntu + OBS capture
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except Exception as e:
            logger.warning("matplotlib unavailable — live plot disabled: %s", e)
            return

        try:
            fig = plt.figure(figsize=(12, 5), facecolor="#1a1a2e")
            fig.canvas.manager.set_window_title("Self-Improving ML Agent — Live Training")

            gs = gridspec.GridSpec(1, 2, figure=fig)
            ax_loss = fig.add_subplot(gs[0, 0])
            ax_f1   = fig.add_subplot(gs[0, 1])

            _style_ax(ax_loss, "Loss", "Epoch", "Loss")
            _style_ax(ax_f1,   "Val F1", "Epoch", "F1")

            plt.tight_layout(pad=2.0)
            plt.ion()
            plt.show(block=False)

        except Exception as e:
            logger.warning("Could not create plot window: %s", e)
            return

        self._available = True
        self._ready.set()  # unblocks the main thread

        # all run series accumulated so far
        series: list[_RunSeries] = []
        current: Optional[_RunSeries] = None

        while True:
            try:
                msg = self._queue.get(timeout=0.05)
            except queue.Empty:
                try:
                    fig.canvas.flush_events()
                except Exception:
                    pass
                continue

            if msg is _STOP:
                break

            kind, data = msg

            if kind == "new_run":
                iteration = data
                current = _RunSeries(iteration=iteration)
                series.append(current)

            elif kind == "epoch" and current is not None:
                current.epochs.append(data["epoch"])
                current.train_loss.append(data["train_loss"])
                current.val_loss.append(data["val_loss"])
                current.val_f1.append(data["val_f1"])

                # redraw
                try:
                    ax_loss.cla()
                    ax_f1.cla()
                    _style_ax(ax_loss, "Loss per Epoch",  "Epoch", "Loss")
                    _style_ax(ax_f1,   "Val F1 per Epoch", "Epoch", "F1")

                    for s in series:
                        colour = _COLOURS[(s.iteration - 1) % len(_COLOURS)]
                        label  = f"Run {s.iteration}"
                        ax_loss.plot(
                            s.epochs, s.train_loss,
                            linestyle="--", color=colour, alpha=0.6,
                            linewidth=1.2,
                        )
                        ax_loss.plot(
                            s.epochs, s.val_loss,
                            linestyle="-", color=colour, alpha=0.9,
                            linewidth=1.8, label=label,
                        )
                        ax_f1.plot(
                            s.epochs, s.val_f1,
                            linestyle="-", color=colour, alpha=0.9,
                            linewidth=1.8, marker="o", markersize=4,
                            label=label,
                        )

                    ax_loss.legend(
                        fontsize=7, loc="upper right",
                        facecolor="#16213e", labelcolor="white",
                    )
                    ax_f1.legend(
                        fontsize=7, loc="lower right",
                        facecolor="#16213e", labelcolor="white",
                    )
                    # goal line on F1 subplot
                    try:
                        from config import GOAL_F1
                        ax_f1.axhline(
                            GOAL_F1, color="#4CE87A", linestyle=":",
                            linewidth=1.5, alpha=0.8, label=f"Goal {GOAL_F1}",
                        )
                    except ImportError:
                        pass

                    fig.canvas.draw()
                    fig.canvas.flush_events()

                except Exception as e:
                    logger.debug("Plot redraw error (non-fatal): %s", e)

        try:
            plt.close(fig)
        except Exception:
            pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _style_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply a consistent dark theme to a matplotlib axis."""
    bg = "#16213e"
    fg = "#e0e0e0"
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg, labelsize=8)
    ax.set_title(title,  color=fg, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=fg, fontsize=8)
    ax.set_ylabel(ylabel, color=fg, fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d2d5e")
    ax.grid(True, color="#2d2d5e", linewidth=0.6, linestyle="--", alpha=0.7)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    import math

    logging.basicConfig(level=logging.DEBUG)
    print("Opening live plot — simulating 2 runs × 10 epochs...")

    plot = LivePlot()
    time.sleep(1)

    for run in range(1, 3):
        plot.new_run(iteration=run)
        for ep in range(1, 11):
            noise = (hash((run, ep)) % 100) / 1000
            fake = {
                "epoch":      ep,
                "train_loss": 0.6 / (ep ** 0.4 * run) + noise,
                "val_loss":   0.7 / (ep ** 0.4 * run) + noise,
                "val_f1":     0.70 + 0.02 * ep * run * 0.5 + noise,
            }
            plot.update(fake)
            time.sleep(0.3)

    print("Done. Close the window or wait 3 seconds...")
    time.sleep(3)
    plot.close()
    print("Plot closed cleanly.")
