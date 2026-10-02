"""The organism's body: the computer it lives in, and how it takes care of it.

Thinking with a local language model keeps the GPU busy, and a laptop that
runs for days at full load gets hot. The organism already regulates its way of
being curious (the temperament); here it regulates its body:

* a short breath after every heartbeat;
* a longer rest after a stretch of thinking (a work and rest rhythm);
* a thermal guard: when the NVIDIA GPU passes a temperature limit, it stops
  thinking until the GPU has cooled down. The temperature is read with
  ``nvidia-smi``, which comes with the NVIDIA driver on Windows and Linux;
* a power guard: on a laptop it only thinks while the charger is connected.

Every sensor is optional. When a reading is unavailable that guard does not
block, and the time-based rhythm still protects the machine.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..config import BodyConfig


@dataclass
class Rest:
    reason: str  # "rhythm", "cooling" or "battery"
    seconds: float
    detail: str


def parse_gpu_temperatures(text: str) -> float | None:
    """The hottest GPU in ``nvidia-smi --query-gpu=temperature.gpu`` output."""
    readings = []
    for line in text.splitlines():
        try:
            readings.append(float(line.strip()))
        except ValueError:
            continue
    return max(readings) if readings else None


def linux_on_battery(power_supply_dir: Path) -> bool | None:
    """True on battery, False on mains power, None if the machine does not say."""
    try:
        supplies = list(power_supply_dir.iterdir())
    except OSError:
        return None
    mains: list[bool] = []
    for supply in supplies:
        try:
            if (supply / "type").read_text().strip() == "Mains":
                mains.append((supply / "online").read_text().strip() == "1")
        except OSError:
            continue
    if not mains:
        return None
    return not any(mains)


def _windows_on_battery() -> bool | None:
    import ctypes

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    status = SystemPowerStatus()
    try:
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return None
    except Exception:
        return None
    return {0: True, 1: False}.get(status.ACLineStatus)


def _mac_on_battery() -> bool | None:
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if "Battery Power" in out:
        return True
    if "AC Power" in out:
        return False
    return None


class SystemSensors:
    """Reads the GPU temperature and the power source of this computer."""

    def __init__(self, power_supply_dir: Path = Path("/sys/class/power_supply")):
        self.nvidia_smi = shutil.which("nvidia-smi")
        self.power_supply_dir = power_supply_dir

    def gpu_temperature(self) -> float | None:
        if not self.nvidia_smi:
            return None
        no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console flash on Windows
        try:
            out = subprocess.run(
                [self.nvidia_smi, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
                **({"creationflags": no_window} if no_window else {}),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return parse_gpu_temperatures(out.stdout)

    def on_battery(self) -> bool | None:
        if sys.platform == "win32":
            return _windows_on_battery()
        if sys.platform == "darwin":
            return _mac_on_battery()
        return linux_on_battery(self.power_supply_dir)


class Body:
    """Decides when the organism must stop thinking so the computer can rest."""

    def __init__(
        self,
        config: "BodyConfig",
        *,
        sensors: SystemSensors | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.sensors = sensors if sensors is not None else SystemSensors()
        self._sleep = sleep
        self._clock = clock
        self.log: Callable[[str], None] = lambda message: None
        self.on_rest: Callable[[Rest], None] = lambda rest: None
        self._awake_since = clock()
        self._last_check: float | None = None

    def before_thinking(self) -> None:
        """Called before every call to the language model."""
        c = self.config
        if not c.enabled:
            return
        now = self._clock()
        if self._last_check is not None and now - self._last_check < c.check_every_seconds:
            return
        self._last_check = now
        if c.pause_on_battery:
            self._wait_for_power()
        if c.gpu_temperature_guard:
            self._cool_down()

    def after_heartbeat(self) -> None:
        """Called between heartbeats: a breath, or a real rest after a stretch of work."""
        c = self.config
        if not c.enabled:
            return
        if c.work_minutes > 0 and c.rest_minutes > 0 and self._clock() - self._awake_since >= c.work_minutes * 60:
            start = self._clock()
            self.log(f"I have been thinking for {c.work_minutes:g} minutes; resting {c.rest_minutes:g} minutes so the computer can cool down.")
            self._sleep(c.rest_minutes * 60)
            self._rested("rhythm", start, f"a planned rest after {c.work_minutes:g} minutes of thinking")
        elif c.breath_seconds > 0:
            self._sleep(c.breath_seconds)

    def describe(self) -> str:
        """One line for the console: what protects the computer right now."""
        c = self.config
        if not c.enabled:
            return "Body care is OFF: no rests and no temperature or battery guard."
        parts = []
        if c.breath_seconds > 0:
            parts.append(f"breath {c.breath_seconds:g}s after each heartbeat")
        if c.work_minutes > 0 and c.rest_minutes > 0:
            parts.append(f"rest {c.rest_minutes:g} min every {c.work_minutes:g} min")
        if c.gpu_temperature_guard:
            temp = self.sensors.gpu_temperature()
            if temp is None:
                parts.append("GPU temperature not readable here (nvidia-smi not found), so the rest rhythm protects it")
            else:
                parts.append(f"GPU now {temp:.0f}°C, pause at {c.max_gpu_temp_c:g}°C until {c.resume_gpu_temp_c:g}°C")
        if c.pause_on_battery:
            parts.append("thinks only while plugged in")
        return "Body care: " + "; ".join(parts) + "."

    def _cool_down(self) -> None:
        c = self.config
        temp = self.sensors.gpu_temperature()
        if temp is None or temp < c.max_gpu_temp_c:
            return
        start, peak, last_report = self._clock(), temp, self._clock()
        self.log(f"GPU at {temp:.0f}°C: I stop thinking until it cools to {c.resume_gpu_temp_c:g}°C.")
        while temp is not None and temp > c.resume_gpu_temp_c:
            self._sleep(c.check_every_seconds)
            temp = self.sensors.gpu_temperature()
            if temp is not None:
                peak = max(peak, temp)
            if self._clock() - last_report >= 300:
                self.log(f"Still cooling down: GPU at {temp:.0f}°C." if temp is not None else "Still cooling down.")
                last_report = self._clock()
        after = f"{temp:.0f}°C" if temp is not None else "no reading"
        self._rested("cooling", start, f"GPU {peak:.0f}°C -> {after}")

    def _wait_for_power(self) -> None:
        if self.sensors.on_battery() is not True:
            return
        start, last_report = self._clock(), self._clock()
        self.log("Running on battery: I rest until the charger is connected.")
        while self.sensors.on_battery() is True:
            self._sleep(max(5.0, self.config.check_every_seconds))
            if self._clock() - last_report >= 600:
                self.log("Still waiting for the charger.")
                last_report = self._clock()
        self._rested("battery", start, "waited for the charger")

    def _rested(self, reason: str, start: float, detail: str) -> None:
        seconds = self._clock() - start
        self._awake_since = self._clock()
        self.log(f"Rested {_duration(seconds)} ({detail}); thinking again.")
        self.on_rest(Rest(reason, seconds, detail))


def _duration(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds >= 90 else f"{seconds:.0f} s"
