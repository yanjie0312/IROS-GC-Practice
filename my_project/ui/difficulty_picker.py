# my_project/ui/difficulty_picker.py
"""Pre-run difficulty selection dialog (tkinter)."""
from __future__ import annotations

from typing import Optional

from my_project.experiments.scenarios import list_difficulty_profiles

# (profile_id, UI title, short description)
DIFFICULTY_CHOICES = (
    ("L0_easy",   "L0 · Easy",   "2-room layout, 3 obstacles, 2 targets; no wind or sensor noise"),
    ("L1_mild",   "L1 · Mild",   "3-room layout, 5 obstacles, 3 targets; no wind or sensor noise"),
    ("L2_medium", "L2 · Medium", "2-room layout, 3 obstacles, 2 targets; wind and sensor noise"),
    ("L3_hard",   "L3 · Hard",   "3-room layout, 5 obstacles, 3 targets; wind and sensor noise (500s)"),
)


def pick_difficulty_profile(default: str = "L0_easy") -> Optional[str]:
    """
    Show difficulty picker. Returns selected profile id, or None if cancelled.
    Falls back to default when tkinter is unavailable.
    """
    valid = set(list_difficulty_profiles())
    if default not in valid:
        default = "L0_easy"

    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("[Difficulty] No GUI available; using config default:", default)
        return default

    choice = {"value": default}
    cancelled = {"flag": False}

    root = tk.Tk()
    root.title("Select Mission Difficulty")
    root.resizable(False, False)

    root.update_idletasks()
    w, h = 440, 320
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    main = ttk.Frame(root, padding=16)
    main.pack(fill=tk.BOTH, expand=True)

    ttk.Label(
        main,
        text="Choose a difficulty level, then click Start Simulation",
        font=("Segoe UI", 11, "bold"),
    ).pack(anchor=tk.W, pady=(0, 10))

    var = tk.StringVar(value=default)
    for profile_id, title, desc in DIFFICULTY_CHOICES:
        if profile_id not in valid:
            continue
        row = ttk.Frame(main)
        row.pack(fill=tk.X, pady=4)
        ttk.Radiobutton(
            row,
            text=title,
            variable=var,
            value=profile_id,
        ).pack(anchor=tk.W)
        ttk.Label(row, text=desc, foreground="#555555", font=("Segoe UI", 9)).pack(
            anchor=tk.W, padx=22
        )

    btn_row = ttk.Frame(main)
    btn_row.pack(fill=tk.X, pady=(18, 0))

    def on_start() -> None:
        choice["value"] = var.get()
        root.destroy()

    def on_cancel() -> None:
        cancelled["flag"] = True
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    ttk.Button(btn_row, text="Start Simulation", command=on_start).pack(side=tk.RIGHT, padx=(8, 0))
    ttk.Button(btn_row, text="Cancel", command=on_cancel).pack(side=tk.RIGHT)

    root.mainloop()

    if cancelled["flag"]:
        return None
    return choice["value"]


def resolve_difficulty_profile(
    *,
    cli_difficulty: Optional[str] = None,
    no_prompt: bool = False,
    prompt_enabled: bool = True,
    config_default: str = "L0_easy",
) -> str:
    """
    Resolve difficulty_profile for this run:
    - CLI --difficulty wins;
    - else show picker when enabled;
    - on cancel, use config_default.
    """
    valid = set(list_difficulty_profiles())

    if cli_difficulty is not None:
        key = cli_difficulty.strip()
        if key not in valid:
            raise ValueError(f"Unknown difficulty: {key}. Choices: {', '.join(sorted(valid))}")
        return key

    if no_prompt or not prompt_enabled:
        key = config_default if config_default in valid else "L0_easy"
        return key

    picked = pick_difficulty_profile(config_default)
    if picked is None:
        print(f"[Difficulty] Selection cancelled; using config default: {config_default}")
        return config_default if config_default in valid else "L0_easy"
    return picked
