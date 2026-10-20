"""RANA's optional desktop indicator, isolated from numerical computation.

Executed as a script by RanaStatus, so its UI process does not import PyTorch.
Closing this window only dismisses the indicator. Closing the parent pipe ends
the UI, including when the simulation process exits unexpectedly.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

MESSAGE = "MaoMi-RANA is working in background"
ASSET = Path(__file__).with_name("assets") / "rana_cat.png"


def desktop_possible():
    if os.environ.get("MAOMI_STATUS_WINDOW", "auto").lower() in {"0", "off", "false"}:
        return False
    if os.environ.get("CI", "").lower() in {"true", "1"}:
        return False
    return sys.platform in {"win32", "darwin"} or bool(os.environ.get("DISPLAY"))


class RanaStatus:
    """Best-effort status window; desktop support never changes the simulation."""

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self.process = None
        self.status = "disabled" if not enabled else "unavailable"

    def __enter__(self):
        if not self.enabled or not desktop_possible():
            return self
        try:
            options = {}
            if sys.platform == "win32":
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--pipe"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf8", **options)
            ready = threading.Event()
            def read_status():
                line = self.process.stdout.readline().strip()
                self.status = "shown" if line == "READY" else "unavailable"
                ready.set()
            threading.Thread(target=read_status, daemon=True).start()
            # Bounded startup acknowledgement ensures even a fast job can show
            # its cat. This UI time is reported separately from simulation time.
            if not ready.wait(2.):
                self.status = "starting"
        except (OSError, ValueError):
            self.status = "unavailable"
        return self

    def __exit__(self, *exc):
        proc = self.process
        if proc is None:
            return False
        try:
            if proc.stdin:
                proc.stdin.close()
        except (OSError, BrokenPipeError):
            pass
        try:
            proc.wait(timeout=.75)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=.75)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if proc.stdout:
            proc.stdout.close()
        return False


def sprite_frames(size=224):
    """Render the requested rotating pixel head, retaining transparency."""
    from PIL import Image
    with Image.open(ASSET) as image:
        head = image.convert("RGBA")
    # Nearest-neighbor sampling and rotation keep square game-sprite pixels.
    head.thumbnail((64, 64), Image.Resampling.NEAREST)
    canvas = Image.new("RGBA", (96, 96))
    canvas.alpha_composite(head, ((96-head.width)//2, (96-head.height)//2))
    return [canvas.rotate(-angle, resample=Image.Resampling.NEAREST)
            .resize((size, size), Image.Resampling.NEAREST) for angle in range(0, 360, 12)]


def show_window(*, pipe=False, seconds=None, capture=None):
    import tkinter as tk
    from PIL import ImageTk
    stop = threading.Event()
    if pipe:
        def watch_parent():
            try:
                sys.stdin.read()
            finally:
                stop.set()
        threading.Thread(target=watch_parent, daemon=True).start()
    root = tk.Tk()
    root.title("MaoMi — RANA Mode")
    root.configure(bg="#151b25")
    root.resizable(False, False)
    width, height = 420, 300
    x = max(0, root.winfo_screenwidth()-width-40)
    y = max(0, root.winfo_screenheight()-height-90)
    root.geometry(f"{width}x{height}+{x}+{y}")
    frames = [ImageTk.PhotoImage(frame, master=root) for frame in sprite_frames()]
    head = tk.Label(root, bg="#151b25", image=frames[0], borderwidth=0)
    head.pack(pady=(8, 2))
    tk.Label(root, text=MESSAGE, bg="#151b25", fg="#f4dfb5",
             font=("TkDefaultFont", 11), padx=8, pady=8).pack()
    root.iconphoto(True, frames[0])
    # Dismissing the cat never cancels its parent's numerical job.
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    started = time.monotonic()
    def tick():
        elapsed = time.monotonic()-started
        if stop.is_set() or (seconds is not None and elapsed >= seconds):
            root.destroy()
            return
        head.configure(image=frames[int(elapsed/.08) % len(frames)])
        root.after(40, tick)
    if capture is not None:
        def screenshot():
            from PIL import ImageGrab
            root.update_idletasks()
            box = (root.winfo_rootx(), root.winfo_rooty(),
                   root.winfo_rootx()+root.winfo_width(), root.winfo_rooty()+root.winfo_height())
            ImageGrab.grab(bbox=box).save(capture)
        root.after(350, screenshot)
    root.after(40, tick)
    root.update_idletasks()
    if pipe:
        print("READY", flush=True)
    root.mainloop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipe", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--seconds", type=float, default=None, help="Auto-close a standalone preview")
    p.add_argument("--capture", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    try:
        show_window(pipe=args.pipe, seconds=args.seconds, capture=args.capture)
    except (ImportError, OSError, RuntimeError) as exc:
        if not args.pipe:
            p.exit(1, f"RANA desktop indicator unavailable: {exc}\n")
    except Exception as exc:
        # TclError includes missing displays and unavailable desktop sessions.
        if not args.pipe:
            p.exit(1, f"RANA desktop indicator unavailable: {exc}\n")


if __name__ == "__main__":
    main()
