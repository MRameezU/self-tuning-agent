import logging
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_STOP = object()


@dataclass
class _RunSeries:
    """Accumulated epoch data for one training run."""
    iteration: int
    epochs:     list[int]   = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss:   list[float] = field(default_factory=list)
    val_f1:     list[float] = field(default_factory=list)


_COLOURS = [
    "#4C9BE8", "#E8834C", "#4CE87A", "#E84C4C",
    "#C44CE8", "#E8D84C", "#4CE8D8", "#E84CAB",
    "#8BE84C", "#4C4CE8",
]


class LivePlot:
    """
    Starts a background thread that maintains a live matplotlib window.
    Call update() from the main thread for every epoch. Call save() to
    persist the current figure to disk, then close() to shut down cleanly.

    Usage:
        plot = LivePlot()
        plot.new_run(iteration=1)
        for epoch_metrics in runner.execute(proposal):
            plot.update(epoch_metrics)
        plot.save(Path("outputs/training_curves.png"))
        plot.close()
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._ready             = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._available         = False
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

    def save(self, path: Path) -> None:
        """
        Save the current figure to disk as a PNG. Blocks until the plot
        thread confirms the write is complete — safe to call before close().
        Does nothing if the display was unavailable.
        """
        if not self._available:
            return
        done = threading.Event()
        self._queue.put(("save", (path, done)))
        done.wait(timeout=10)

    def close(self) -> None:
        """Shut down the background thread and close the window cleanly."""
        if self._available and self._thread is not None:
            self._queue.put(_STOP)
            self._thread.join(timeout=5)
            self._available = False

    # ── private ───────────────────────────────────────────────────────────────

    def _start(self) -> None:
        self._thread = threading.Thread(
            target=self._plot_loop,
            daemon=True,
            name="live-plot",
        )
        self._thread.start()
        # block until the plot thread confirms the window is up (or gave up)
        self._ready.wait(timeout=3.0)

    def _plot_loop(self) -> None:
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except Exception as e:
            logger.warning("matplotlib unavailable — live plot disabled: %s", e)
            self._ready.set()
            return

        try:
            fig = plt.figure(figsize=(12, 5), facecolor="#1a1a2e")
            fig.canvas.manager.set_window_title("Self-Improving ML Agent — Live Training")
            gs      = gridspec.GridSpec(1, 2, figure=fig, hspace=0.1)
            ax_loss = fig.add_subplot(gs[0, 0])
            ax_f1   = fig.add_subplot(gs[0, 1])
            _style_ax(ax_loss, "Loss per Epoch",   "Epoch", "Loss")
            _style_ax(ax_f1,   "Val F1 per Epoch", "Epoch", "F1")
            plt.tight_layout(pad=2.0)
            plt.ion()
            plt.show(block=False)
        except Exception as e:
            logger.warning("Could not create plot window: %s", e)
            self._ready.set()
            return

        self._available = True
        self._ready.set()   # unblock _start() in the main thread

        series:  list[_RunSeries]   = []
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
                current = _RunSeries(iteration=data)
                series.append(current)

            elif kind == "epoch" and current is not None:
                current.epochs.append(data["epoch"])
                current.train_loss.append(data["train_loss"])
                current.val_loss.append(data["val_loss"])
                current.val_f1.append(data["val_f1"])
                _redraw(fig, ax_loss, ax_f1, series)

            elif kind == "save":
                out_path, done_event = data
                try:
                    out = Path(out_path)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(
                        out, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor(),
                    )
                    logger.info("Live plot saved → %s", out)
                except Exception as e:
                    logger.warning("Failed to save live plot: %s", e)
                finally:
                    done_event.set()   # unblock main thread regardless

        try:
            plt.close(fig)
        except Exception:
            pass


# ── Drawing ───────────────────────────────────────────────────────────────────

def _redraw(fig, ax_loss, ax_f1, series: list[_RunSeries]) -> None:
    try:
        ax_loss.cla()
        ax_f1.cla()
        _style_ax(ax_loss, "Loss per Epoch",   "Epoch", "Loss")
        _style_ax(ax_f1,   "Val F1 per Epoch", "Epoch", "F1")

        for s in series:
            colour = _COLOURS[(s.iteration - 1) % len(_COLOURS)]
            label  = f"Run {s.iteration}"
            ax_loss.plot(s.epochs, s.train_loss, linestyle="--", color=colour,
                         alpha=0.6, linewidth=1.2)
            ax_loss.plot(s.epochs, s.val_loss,   linestyle="-",  color=colour,
                         alpha=0.9, linewidth=1.8, label=label)
            ax_f1.plot(  s.epochs, s.val_f1,     linestyle="-",  color=colour,
                         alpha=0.9, linewidth=1.8, marker="o", markersize=4,
                         label=label)

        ax_loss.legend(fontsize=7, loc="upper right",
                       facecolor="#16213e", labelcolor="white")
        ax_f1.legend(  fontsize=7, loc="lower right",
                       facecolor="#16213e", labelcolor="white")

        try:
            from config import GOAL_F1
            ax_f1.axhline(GOAL_F1, color="#4CE87A", linestyle=":",
                          linewidth=1.5, alpha=0.8)
        except ImportError:
            pass

        fig.canvas.draw()
        fig.canvas.flush_events()
    except Exception as e:
        logger.debug("Plot redraw error (non-fatal): %s", e)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _style_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    bg = "#16213e"
    fg = "#e0e0e0"
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg, labelsize=8)
    ax.set_title(title,   color=fg, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=fg, fontsize=8)
    ax.set_ylabel(ylabel, color=fg, fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d2d5e")
    ax.grid(True, color="#2d2d5e", linewidth=0.6, linestyle="--", alpha=0.7)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.DEBUG)
    print("Opening live plot...")
    plot = LivePlot()
    for run in range(1, 3):
        plot.new_run(iteration=run)
        for ep in range(1, 11):
            noise = (hash((run, ep)) % 100) / 1000
            plot.update({
                "epoch":      ep,
                "train_loss": 0.6 / (ep ** 0.4 * run) + noise,
                "val_loss":   0.7 / (ep ** 0.4 * run) + noise,
                "val_f1":     0.70 + 0.02 * ep * run * 0.5 + noise,
            })
            time.sleep(0.3)
    plot.save(Path("/tmp/smoke_test_plot.png"))
    print("Saved to /tmp/smoke_test_plot.png")
    time.sleep(2)
    plot.close()
    print("Closed cleanly.")