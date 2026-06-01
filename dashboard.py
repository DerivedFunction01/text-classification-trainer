import json
import curses
import re
import shutil
import subprocess
import time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def safe_addstr(stdscr, y, x, text):
    height, width = stdscr.getmaxyx()
    if y < 0 or y >= height or x >= width:
        return
    stdscr.addnstr(y, x, text, max(0, width - x - 1))


def probe_gpu_status():
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            device_lines = []
            rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            for index, row in enumerate(rows):
                parts = [part.strip() for part in row.split(",")]
                if len(parts) >= 5:
                    name, util, mem_used, mem_total, temp = parts[:5]
                    device_lines.append(
                        f"GPU {index}: {name} | Util {util}% | Mem {mem_used}/{mem_total} MiB | Temp {temp} C"
                    )
                else:
                    device_lines.append(f"GPU {index}: {row}")
            return ["GPU: NVIDIA", *device_lines]
        return ["GPU: NVIDIA", "Status unavailable"]

    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi is not None:
        json_result = subprocess.run(
            [rocm_smi, "--json"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if json_result.returncode == 0 and json_result.stdout.strip():
            parsed = _parse_rocm_smi_json(json_result.stdout)
            if parsed:
                return ["GPU: AMD / ROCm", *parsed]

        text_result = subprocess.run(
            [rocm_smi],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if text_result.returncode == 0 and text_result.stdout.strip():
            parsed = _parse_rocm_smi_text(text_result.stdout)
            if parsed:
                return ["GPU: AMD / ROCm", *parsed]
        return ["GPU: AMD / ROCm", "Status unavailable"]

    return ["GPU: Not detected (CPU-only)"]


def _parse_rocm_smi_json(payload):
    def walk(obj, path=()):
        if isinstance(obj, dict):
            yield path, obj
            for key, value in obj.items():
                yield from walk(value, path + (str(key),))
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                yield from walk(value, path + (str(index),))

    data = json.loads(payload)
    devices = []
    seen = set()
    for path, item in walk(data):
        lowered = {str(key).lower(): value for key, value in item.items()}
        name = _first_matching(lowered, ("card series", "product name", "gpu name", "name"))
        util = _first_matching(lowered, ("gpu use", "gpu utilization", "utilization"))
        mem_used = _first_matching(lowered, ("vram used memory", "memory used", "used memory"))
        mem_total = _first_matching(lowered, ("vram total memory", "memory total", "total memory"))
        temp = _first_matching(lowered, ("temperature", "temp"))

        if not any(value is not None for value in (name, util, mem_used, mem_total, temp)):
            continue

        label = name or _guess_rocm_label(path, lowered)
        signature = (label, util, mem_used, mem_total, temp)
        if signature in seen:
            continue
        seen.add(signature)
        devices.append(
            _format_device_line(
                label=label,
                util=util,
                mem_used=mem_used,
                mem_total=mem_total,
                temp=temp,
            )
        )
    return devices


def _parse_rocm_smi_text(payload):
    devices = []
    current = {}
    current_label = None

    def flush():
        nonlocal current, current_label
        if not current and not current_label:
            return
        name = _first_matching(current, ("card series", "product name", "gpu name", "name"))
        util = _first_matching(current, ("gpu use", "gpu utilization", "utilization"))
        mem_used = _first_matching(current, ("vram used memory", "memory used", "used memory"))
        mem_total = _first_matching(current, ("vram total memory", "memory total", "total memory"))
        temp = _first_matching(current, ("temperature", "temp"))
        label = name or current_label or f"GPU {len(devices)}"
        if any(value is not None for value in (name, util, mem_used, mem_total, temp)):
            devices.append(
                _format_device_line(
                    label=label,
                    util=util,
                    mem_used=mem_used,
                    mem_total=mem_total,
                    temp=temp,
                )
            )
        current = {}
        current_label = None

    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
            continue

        match = re.match(r"(?:GPU|Card)\[(\d+)\]|\bGPU\s*(\d+)\b", stripped, re.IGNORECASE)
        if match:
            flush()
            current_label = f"GPU {next(group for group in match.groups() if group is not None)}"
            continue

        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key and value and key not in current:
            current[key] = value

    flush()
    return devices


def _first_matching(mapping, needles):
    for key, value in mapping.items():
        if any(needle in key for needle in needles) and value not in (None, ""):
            return value
    return None


def _guess_rocm_label(path, lowered):
    for part in reversed(path):
        if part.isdigit():
            return f"GPU {part}"
    for candidate in ("card", "gpu", "device", "instance"):
        for key in lowered:
            if candidate in key:
                match = re.search(r"(\d+)", str(lowered[key]))
                if match is not None:
                    return f"GPU {match.group(1)}"
    return f"GPU {len(path)}"


def _normalize_memory(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = re.fullmatch(r"(\d+)(?:\s*bytes?)?", text, re.IGNORECASE)
    if digits is not None:
        mib = int(digits.group(1)) / (1024 * 1024)
        return f"{mib:.0f} MiB"
    return text


def _normalize_percent(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not text.endswith("%"):
        text = f"{text}%"
    return text


def _normalize_temp(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not any(unit in text.lower() for unit in ("c", "°")):
        text = f"{text} C"
    return text


def _compact_status_lines(*, name=None, util=None, mem_used=None, mem_total=None, temp=None):
    return [_format_device_line(label=name, util=util, mem_used=mem_used, mem_total=mem_total, temp=temp)]


def _format_device_line(*, label=None, util=None, mem_used=None, mem_total=None, temp=None):
    parts = []
    if label:
        parts.append(str(label))
    if util:
        parts.append(f"Util {_normalize_percent(util)}")
    if mem_used or mem_total:
        used = _normalize_memory(mem_used) or str(mem_used).strip()
        total = _normalize_memory(mem_total) or str(mem_total).strip()
        if used and total:
            parts.append(f"Mem {used}/{total}")
        elif used:
            parts.append(f"Mem {used}")
        elif total:
            parts.append(f"Mem {total}")
    if temp:
        parts.append(f"Temp {_normalize_temp(temp)}")
    return " | ".join(parts)


def load_scalars(acc):
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    data = {}

    for tag in tags:
        events = acc.Scalars(tag)
        if events:
            last = events[-1]
            data[tag] = (last.step, last.value)
    return data


def draw(stdscr, logdir, refresh):
    curses.curs_set(0)
    stdscr.nodelay(True)

    acc = EventAccumulator(logdir, size_guidance={"scalars": 0})
    acc.Reload()

    while True:
        stdscr.erase()
        safe_addstr(stdscr, 0, 0, f"TensorBoard Live Viewer — {logdir}")
        safe_addstr(stdscr, 1, 0, "Press q to quit")

        data = load_scalars(acc)
        gpu_lines = probe_gpu_status()

        row = 3
        safe_addstr(stdscr, row, 0, "GPU")
        row += 1
        for line in gpu_lines:
            safe_addstr(stdscr, row, 0, line)
            row += 1

        row += 1
        safe_addstr(stdscr, row, 0, f"{'TAG':40} {'STEP':>10} {'VALUE':>12}")
        safe_addstr(stdscr, row + 1, 0, "-" * 65)
        row += 2

        for tag, (step, value) in sorted(data.items()):
            safe_addstr(stdscr, row, 0, f"{tag:40} {step:10d} {value:12.6f}")
            row += 1

        stdscr.refresh()
        time.sleep(refresh)

        try:
            if stdscr.getkey() == "q":
                break
        except:
            pass


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str)
    parser.add_argument("--refresh", type=float, default=1.0)
    args = parser.parse_args()

    curses.wrapper(draw, args.logdir, args.refresh)


if __name__ == "__main__":
    main()
