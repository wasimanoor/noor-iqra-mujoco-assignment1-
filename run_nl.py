from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from nl_interface import parse_command

ROOT = Path(__file__).resolve().parent
PICK_SCRIPT = ROOT / "pickandplace.py"


def _call_pickandplace(args: list[str]) -> int:
    cmd = [sys.executable, str(PICK_SCRIPT)] + args
    print(f"\n>> Running: {' '.join(cmd)}\n")
    p = subprocess.run(cmd)
    return p.returncode


def main() -> None:
    if not PICK_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot find {PICK_SCRIPT}")

    print("Natural Language Runner (MVP)")
    print("Examples:")
    print("  pick red cube place bin")
    print("  pick red_cube place x 0.25 y -0.15")
    print("  open gripper")
    print("  close gripper")
    print("  reset")
    print("  quit\n")

    while True:
        text = input("NL> ").strip()
        if not text:
            continue

        cmd = parse_command(text)

        if cmd["task"] == "quit":
            print("Bye.")
            return

        if cmd["task"] == "help":
            print(
                "\nCommands:\n"
                "  pick <obj> place bin\n"
                "  pick <obj> place x <float> y <float>\n"
                "  open gripper | close gripper\n"
                "  reset | home\n"
                "  quit\n"
            )
            continue

        if cmd["task"] == "unknown":
            print(f"❌ Didn't understand: {cmd.get('text')}")
            continue

        if cmd["task"] == "reset":
            _call_pickandplace(["--reset"])
            continue

        if cmd["task"] == "gripper":
            _call_pickandplace(["--gripper", cmd["mode"]])
            continue

        if cmd["task"] == "pick_place":
            obj = cmd.get("obj", "red_cube")
            target = cmd.get("target", "bin_center")
            _call_pickandplace(["--task", "pick_place", "--obj", obj, "--target", target])
            continue

        if cmd["task"] == "pick_place_xy":
            obj = cmd.get("obj", "red_cube")
            x = cmd["x"]
            y = cmd["y"]
            _call_pickandplace(["--task", "pick_place_xy", "--obj", obj, "--x", str(x), "--y", str(y)])
            continue

        print(f"❌ Unhandled task: {cmd}")


if __name__ == "__main__":
    main()
