from __future__ import annotations

from pathlib import Path

from curiosity_ai.config import BodyConfig
from curiosity_ai.organism import CuriosityOrganism
from curiosity_ai.organism.body import (
    Body,
    SystemSensors,
    linux_on_battery,
    parse_gpu_temperatures,
    parse_win32_battery_status,
)

from .organism_fakes import ScriptedLLM


class FakeSensors:
    """Replays temperatures and power states; the last value repeats."""

    def __init__(self, temps=(None,), battery=(None,)):
        self.temps, self.battery = list(temps), list(battery)
        self.temp_reads = 0

    def gpu_temperature(self):
        self.temp_reads += 1
        return self.temps.pop(0) if len(self.temps) > 1 else self.temps[0]

    def on_battery(self):
        return self.battery.pop(0) if len(self.battery) > 1 else self.battery[0]


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def make_body(sensors: FakeSensors, **config) -> tuple[Body, FakeTime, list]:
    t = FakeTime()
    body = Body(BodyConfig(**config), sensors=sensors, sleep=t.sleep, clock=t.clock)
    rests = []
    body.on_rest = rests.append
    return body, t, rests


def test_a_hot_gpu_stops_thinking_until_it_has_cooled():
    body, t, rests = make_body(FakeSensors(temps=[85, 82, 71, 64]), max_gpu_temp_c=80, resume_gpu_temp_c=65, check_every_seconds=15)
    body.before_thinking()
    assert t.slept == [15, 15, 15]
    assert [(r.reason, r.detail) for r in rests] == [("cooling", "GPU 85°C -> 64°C")]


def test_a_cool_gpu_does_not_wait():
    body, t, rests = make_body(FakeSensors(temps=[62]))
    body.before_thinking()
    assert t.slept == [] and rests == []


def test_the_temperature_is_read_at_most_every_few_seconds():
    sensors = FakeSensors(temps=[60])
    body, t, _ = make_body(sensors, check_every_seconds=15)
    body.before_thinking()
    body.before_thinking()
    assert sensors.temp_reads == 1
    t.now += 16
    body.before_thinking()
    assert sensors.temp_reads == 2


def test_no_sensor_never_blocks():
    body, t, rests = make_body(FakeSensors(temps=[None], battery=[None]))
    body.before_thinking()
    assert t.slept == [] and rests == []


def test_a_rest_follows_a_stretch_of_thinking():
    body, t, rests = make_body(FakeSensors(), breath_seconds=10, work_minutes=20, rest_minutes=5)
    body.after_heartbeat()
    assert t.slept == [10]
    t.now += 20 * 60
    body.after_heartbeat()
    assert t.slept[-1] == 300
    assert rests[-1].reason == "rhythm"
    body.after_heartbeat()
    assert t.slept[-1] == 10  # the work clock restarted after the rest


def test_a_cooling_rest_also_restarts_the_work_clock():
    body, t, rests = make_body(FakeSensors(temps=[90, 60]), work_minutes=20, rest_minutes=5, breath_seconds=10)
    t.now += 19 * 60
    body.before_thinking()
    assert rests[-1].reason == "cooling"
    t.now += 2 * 60
    body.after_heartbeat()
    assert t.slept[-1] == 10


def test_a_laptop_on_battery_waits_for_the_charger():
    body, t, rests = make_body(FakeSensors(temps=[50], battery=[True, True, True, False]))
    body.before_thinking()
    assert len(t.slept) == 2  # the first reading decides to wait; two more find it still unplugged
    assert rests[-1].reason == "battery"


def test_battery_guard_can_be_turned_off():
    body, t, _ = make_body(FakeSensors(temps=[50], battery=[True]), pause_on_battery=False)
    body.before_thinking()
    assert t.slept == []


def test_body_care_can_be_switched_off():
    body, t, rests = make_body(FakeSensors(temps=[99], battery=[True]), enabled=False)
    body.before_thinking()
    body.after_heartbeat()
    assert t.slept == [] and rests == []
    assert "OFF" in body.describe()


def test_resume_temperature_is_always_below_the_limit():
    assert BodyConfig(max_gpu_temp_c=70, resume_gpu_temp_c=75).resume_gpu_temp_c == 60


def test_describe_reports_whether_the_gpu_can_be_read():
    assert "GPU now 55°C" in make_body(FakeSensors(temps=[55]))[0].describe()
    assert "not readable" in make_body(FakeSensors(temps=[None]))[0].describe()


def test_parse_gpu_temperatures():
    assert parse_gpu_temperatures("65\n71\n") == 71
    assert parse_gpu_temperatures("[N/A]\n") is None
    assert parse_gpu_temperatures("") is None


def test_linux_power_supply_reading(tmp_path: Path):
    (tmp_path / "AC").mkdir()
    (tmp_path / "AC" / "type").write_text("Mains\n")
    (tmp_path / "AC" / "online").write_text("0\n")
    (tmp_path / "BAT0").mkdir()
    (tmp_path / "BAT0" / "type").write_text("Battery\n")
    assert linux_on_battery(tmp_path) is True
    (tmp_path / "AC" / "online").write_text("1\n")
    assert linux_on_battery(tmp_path) is False
    assert linux_on_battery(tmp_path / "missing") is None


def test_without_nvidia_smi_the_temperature_is_simply_unknown(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert SystemSensors().gpu_temperature() is None


def test_the_organism_rests_between_heartbeats_and_notes_it_in_its_diary(config):
    config.organism.body.enabled = True
    t = FakeTime()
    body = Body(config.organism.body, sensors=FakeSensors(temps=[84, 70, 60], battery=[False]), sleep=t.sleep, clock=t.clock)
    org = CuriosityOrganism(config, llm=ScriptedLLM(), body=body)
    org.live(2)
    assert t.slept.count(config.organism.body.breath_seconds) == 1  # between the two heartbeats, not after the last
    diary = (org.home / "diary.md").read_text(encoding="utf-8")
    assert "to let the computer cool down: GPU 84°C -> 60°C" in diary
    assert org.state.heartbeat == 2


def test_windows_battery_codes():
    assert parse_win32_battery_status("1\r\n") is True
    assert parse_win32_battery_status("2\r\n") is False
    assert parse_win32_battery_status("6\r\n") is False
    assert parse_win32_battery_status("") is None  # desktop: no battery
    assert parse_win32_battery_status("10\r\n") is None


def test_inside_wsl_the_charger_is_asked_from_windows(monkeypatch, tmp_path: Path):
    import subprocess

    calls = []

    class Done:
        stdout = "1\r\n"

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return Done()

    monkeypatch.setattr("curiosity_ai.organism.body.sys.platform", "linux")
    monkeypatch.setattr("curiosity_ai.organism.body.running_in_wsl", lambda: True)
    monkeypatch.setattr("shutil.which", lambda name: "/mnt/c/WINDOWS/powershell.exe" if name == "powershell.exe" else None)
    monkeypatch.setattr(subprocess, "run", fake_run)
    sensors = SystemSensors(power_supply_dir=tmp_path)
    assert sensors.in_wsl
    assert sensors.on_battery() is True
    assert sensors.on_battery() is True
    assert len(calls) == 1  # cached: asking Windows is slow
    assert "Win32_Battery" in calls[0][-1]


def test_a_new_life_may_rest_while_it_is_born(config):
    """A laptop on battery waited for the charger while a new life prepared its topic, then crashed: the rest
    was recorded before the organism had a place to record it."""
    config.organism.body.enabled = True
    config.organism.research.topic = "How does lightning illuminate the inner magnetosphere?"
    t = FakeTime()
    body = Body(config.organism.body, sensors=FakeSensors(temps=[None], battery=[True, True, False]), sleep=t.sleep, clock=t.clock)
    org = CuriosityOrganism(config, llm=ScriptedLLM(), body=body)
    assert [r.reason for r in org.rest_log] == ["battery"]
    diary = (org.home / "diary.md").read_text(encoding="utf-8")
    assert diary.startswith("# Diary of") and "until the charger was connected" in diary
