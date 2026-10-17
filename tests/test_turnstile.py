#!/usr/bin/env python3
"""Tests for turnstile.py.

What matters: one GPU job at a time; the owner's hold stops new starts and status says when the GPU is ready;
registered research with its task goes before other work and each class is first come, first served; nothing starts
into a reserved window, while a reserved run is still going or its state cannot be read (an unknown state included),
without the declared GPU memory free, or before its RAM cap fits beside the reserve, workers included; a short run past
its deadline takes a free normal slot or is stopped, with SIGKILL if it ignores SIGTERM; a short run moved to a normal
slot keeps it after its launcher dies, while a stale slot record or a reused pid does not hold a slot; a waiting worker
that is interrupted exits 130; refusals for bad research claims (including no task), long short-runs and out-of-range
RAM caps; a waiting job clears stale tickets; nvidia-smi output with commas in a process name still parses; a record is
replaced whole, so a reader never sees part of one. Status only reads: it changes and removes nothing, so it never
removes a holder record it read as stale even when a new GPU job has replaced it meanwhile; hold names only a holder
that still runs. The grant is held until every process of the job has ended: also when
writing the job's record fails after it has started; with memory groups off, for the processes the job left behind;
and when Turnstile itself was killed and the job's main process has ended while another of its processes lives (in the
job's session, or its memory group). After a failed record write and a killed Turnstile, the GPU lock stays with every
process that kept the descriptor the job inherited. A slot or GPU holder record that cannot be read keeps its grant
taken, and status says so. At its deadline a short run whose normal-slot record cannot be written is stopped; a failed log or terminal
write never keeps a short run from being stopped, nor ends a wait for a job's processes early. With memory groups (the
MemoryGroups class repeats every test with them on): each job runs in its own group with its cap, outside the caller's;
a job that outgrows its cap is stopped alone, with exit 137 and a logged reason; the GPU lock stays with a job whose
Turnstile process was killed; a group whose state cannot be read keeps the grant; the job's peak memory is recorded.
Temporary paths may contain spaces.
Not separately tested: the second admission check made while holding the lock (its race window is too narrow).
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import io
import json
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
TOOL = HERE.parent / "turnstile.py"
sys.path.insert(0, str(HERE.parent))

import turnstile  # noqa: E402

TZ = dt.timezone(dt.timedelta(hours=1))
WIN = [{"name": "morning", "start": "05:55", "end": "06:45"}]
REG = {"projects": [{"id": "project-a", "lane": "lane1", "status": "published"},
                    {"id": "support-tool", "lane": "support", "status": "active"},
                    {"id": "old-project", "lane": "lane0", "status": "closed"}]}


def groups_work() -> bool:
    try:
        return subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "--collect", "--", "true"],
                              capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def at(h, m, day=11):
    return dt.datetime(2026, 10, day, h, m, tzinfo=TZ)


class Broken(io.StringIO):
    """A terminal that has gone away: every write fails."""

    def write(self, _text):
        raise OSError("synthetic: the terminal has gone")


def catches(pid: int, signum: int) -> bool:
    """Whether the process has installed a handler for the signal (SigCgt in /proc/PID/status)."""
    try:
        for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("SigCgt:"):
                return bool(int(line.split()[1], 16) >> (signum - 1) & 1)
    except OSError:
        pass
    return False


class PureFunctions(unittest.TestCase):
    def test_order_classes_then_first_come(self):
        t = [{"id": "a", "class": "other", "enqueued_at": "1"}, {"id": "b", "class": "research", "enqueued_at": "2"},
             {"id": "c", "class": "other", "enqueued_at": "0"}, {"id": "d", "class": "scheduled", "enqueued_at": "3"}]
        self.assertEqual([x["id"] for x in turnstile.order(t)], ["d", "b", "c", "a"])

    def test_window(self):
        self.assertIsNotNone(turnstile.window_block(at(6, 10), WIN, 5, "other"))         # inside the window
        self.assertIsNone(turnstile.window_block(at(5, 0), WIN, 50, "research"))        # ends 05:50, before it opens
        self.assertIsNotNone(turnstile.window_block(at(5, 0), WIN, 60, "research"))     # would run into it
        self.assertIsNone(turnstile.window_block(at(6, 10), WIN, 30, "scheduled"))      # the scheduled runs own it
        self.assertIsNone(turnstile.window_block(at(7, 0), WIN, 30, "other"))
        self.assertIsNotNone(turnstile.window_block(at(23, 0), WIN, 7 * 60, "other"))   # crosses midnight into it
        late = [{"name": "night", "start": "23:30", "end": "00:30"}]
        self.assertIsNotNone(turnstile.window_block(at(0, 10, 12), late, 5, "other"))   # window over midnight

    def test_research_claims(self):
        self.assertTrue(turnstile.research_ok("project-a", REG)[0])
        for bad in (None, "", "support-tool", "old-project", "no-such-project"):
            self.assertFalse(turnstile.research_ok(bad, REG)[0], bad)

    def test_unreadable_reserved_states_block(self):
        windows = [{"name": "sweep", "start": "05:55", "end": "06:45", "units": ["sweep.service"]}]
        unreadable = "the state of the reserved runs cannot be read"
        for stdout, code, expected in (("unknown\n", 1, unreadable), ("maintenance\n", 3, unreadable), ("", 0, unreadable),
                                       ("active\n", 0, "reserved run sweep.service is running"), ("inactive\n", 3, None),
                                       ("failed\n", 3, None)):
            with self.subTest(stdout=stdout):
                done = subprocess.CompletedProcess(["systemctl"], code, stdout, "")
                with mock.patch.object(turnstile.subprocess, "run", return_value=done):
                    self.assertEqual(turnstile.reserved_running(windows), expected)

    def test_an_unreadable_memory_group_keeps_the_grant(self):
        answers = [subprocess.TimeoutExpired(["systemctl"], 20), subprocess.CompletedProcess(["systemctl"], 1, "unknown\n", ""),
                   subprocess.CompletedProcess(["systemctl"], 0, "active\n", ""),
                   subprocess.CompletedProcess(["systemctl"], 3, "inactive\n", "")]
        with mock.patch.object(turnstile.subprocess, "run", side_effect=answers) as run, \
                mock.patch.object(turnstile.time, "sleep"), mock.patch.object(turnstile, "log"), \
                contextlib.redirect_stderr(io.StringIO()):
            turnstile.wait_for_scope("turnstile-test-scope", {"id": "t", "owner": "t"})
        self.assertEqual(run.call_count, 4)  # it returned only once systemd reported the group inactive

    def test_dead_or_reused_pids_do_not_count(self):
        me = os.getpid()
        self.assertTrue(turnstile.job_alive({"pid": me}))
        self.assertTrue(turnstile.job_alive({"pid": me, "pid_started": turnstile.proc_start(me)}))
        self.assertFalse(turnstile.job_alive({"pid": me, "pid_started": "0"}))  # the pid now names another process
        self.assertFalse(turnstile.job_alive({"pid": 999999999, "child_pid": 999999998}))

    def test_records_are_replaced_whole(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "normal-0.json"
            turnstile.write_json(path, {"id": "first", "pid": 1})
            with open(path, encoding="utf-8") as reader:  # opened before the rewrite
                turnstile.write_json(path, {"id": "second", "pid": 2})
                self.assertEqual(json.loads(reader.read())["id"], "first")  # the earlier record stayed whole
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["id"], "second")
            with mock.patch.object(turnstile.os, "replace", side_effect=OSError("synthetic: the rename failed")):
                with self.assertRaises(OSError):
                    turnstile.write_json(path, {"id": "third", "pid": 3})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["id"], "second")
            self.assertEqual([p.name for p in pathlib.Path(folder).iterdir()], ["normal-0.json"])  # no temporary file left

    def test_a_live_session_keeps_a_job_alive(self):
        p = subprocess.Popen(["sleep", "5"], start_new_session=True)
        self.addCleanup(lambda: (p.kill(), p.wait()))
        self.assertTrue(turnstile.job_alive({"pid": 999999999, "session": p.pid}))
        p.kill()
        p.wait()
        self.assertFalse(turnstile.job_alive({"pid": 999999999, "session": p.pid}))

    def test_a_wait_survives_failed_terminal_and_log_writes(self):
        answers = [subprocess.CompletedProcess(["systemctl"], 1, "unknown\n", ""),
                   subprocess.CompletedProcess(["systemctl"], 0, "active\n", ""),
                   subprocess.CompletedProcess(["systemctl"], 3, "inactive\n", "")]
        with mock.patch.object(turnstile.subprocess, "run", side_effect=answers) as run, \
                mock.patch.object(turnstile.time, "sleep"), \
                mock.patch.object(turnstile, "log", side_effect=OSError("synthetic: the state folder is full")), \
                contextlib.redirect_stderr(Broken()):
            turnstile.wait_for_scope("turnstile-test-scope", {"id": "t", "owner": "t"})
        self.assertEqual(run.call_count, 3)  # it went on waiting until systemd reported the group inactive


class Process(unittest.TestCase):
    groups = False

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.free = self.tmp / "free"
        self.free.write_text("12000")
        smi = self.tmp / "fake-smi"
        smi.write_text("#!/bin/sh\ncase \"$1\" in --query-gpu*) echo \"16376, 1000, $(cat %s)\";;\n"
                       "*) echo '4321, /opt/browser/browser --type=gpu-process --x=a,b,c, 134'; echo '777, ollama, 9000';; esac\n"
                       % shlex.quote(str(self.free)))
        smi.chmod(0o755)
        self.meminfo = self.tmp / "meminfo"
        self.set_available(60)
        self.unit_state = self.tmp / "unit-state"
        self.unit_state.write_text("inactive")
        unitctl = self.tmp / "fake-systemctl"
        unitctl.write_text("#!/bin/sh\nshift 2\nfor u in \"$@\"; do cat %s; echo; done\n" % shlex.quote(str(self.unit_state)))
        unitctl.chmod(0o755)
        self.cfg = self.tmp / "turnstile.json"
        self.write_config()
        reg = self.tmp / "registry.json"
        reg.write_text(json.dumps(REG))
        self.env = {**os.environ, "TURNSTILE_STATE": str(self.tmp / "state"), "TURNSTILE_CONFIG": str(self.cfg),
                    "TURNSTILE_SMI": str(smi), "TURNSTILE_REGISTRY": str(reg), "TURNSTILE_MEMINFO": str(self.meminfo),
                    "TURNSTILE_UNITCTL": str(unitctl)}
        self.events = self.tmp / "events"

    def write_config(self, **changes):
        cfg = {"normal_slots": 1, "short_slots": 1, "short_max_minutes": 15, "poll_seconds": 0.1, "memory_margin_mib": 0,
               "memory_groups": self.groups, "job_ram_gib": 1, "worker_ram_gib": 1, "ram_cap_max_gib": 44,
               "ram_reserve_gib": 12, "swap_cap_gib": 0.25, "windows": []}
        self.cfg.write_text(json.dumps({**cfg, **changes}))

    def set_available(self, gib):
        tmp = self.meminfo.with_suffix(".new")
        tmp.write_text(f"MemTotal:       67108864 kB\nMemAvailable:   {int(gib * 1048576)} kB\n")
        tmp.replace(self.meminfo)

    def reap(self, p):
        if p.poll() is None:
            p.kill()
        p.wait()

    def script(self, name, seconds):
        events = shlex.quote(str(self.events))
        return ["--", "sh", "-c", f"echo start {name} $(date +%s.%N) >> {events}; sleep {seconds}; echo end {name} $(date +%s.%N) >> {events}"]

    def job(self, name, cls="other", project=None, mem=100, seconds=0.6):
        cmd = [sys.executable, str(TOOL), "run", "--class", cls, "--mem", str(mem), "--max", "5", "--owner", name]
        if project:
            cmd += ["--project", project, "--task", "packet-under-test"]
        p = subprocess.Popen(cmd + self.script(name, seconds), env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self.reap, p)
        return p

    def worker(self, name, kind, minutes, command):
        cmd = [sys.executable, str(TOOL), "worker", "--kind", kind, "--max", str(minutes), "--owner", name, "--", *command]
        p = subprocess.Popen(cmd, env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self.reap, p)
        return p

    def status(self):
        return subprocess.run([sys.executable, str(TOOL), "status"], env=self.env, capture_output=True, text=True, check=True).stdout

    def cap_job(self, name, seconds=0.3):
        cmd = [sys.executable, str(TOOL), "cap", "--max", "5", "--owner", name] + self.script(name, seconds)
        p = subprocess.Popen(cmd, env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self.reap, p)
        return p

    def wait_for(self, path, seconds=30):
        until = time.time() + seconds
        while not path.exists() and time.time() < until:
            time.sleep(0.05)
        self.assertTrue(path.exists(), path)

    def read_events(self):
        rows = [line.split() for line in self.events.read_text().splitlines()] if self.events.exists() else []
        return [(r[0], r[1], float(r[2])) for r in rows]

    def log_events(self):
        return [json.loads(line) for line in (self.tmp / "state" / "log.jsonl").read_text().splitlines()]

    def test_one_at_a_time(self):
        procs = [self.job(f"j{i}") for i in range(3)]
        for p in procs:
            self.assertEqual(p.wait(timeout=30), 0)
        ev = sorted(self.read_events(), key=lambda e: e[2])
        running = 0
        for kind, _name, _t in ev:
            running += 1 if kind == "start" else -1
            self.assertLessEqual(running, 1)
        self.assertEqual(len(ev), 6)

    def test_research_goes_before_earlier_other(self):
        first = self.job("first", seconds=1.5)
        time.sleep(0.5)
        other = self.job("other")
        time.sleep(0.3)
        research = self.job("research", cls="research", project="project-a")
        for p in (first, other, research):
            self.assertEqual(p.wait(timeout=30), 0)
        starts = [n for k, n, _t in sorted(self.read_events(), key=lambda e: e[2]) if k == "start"]
        self.assertEqual(starts, ["first", "research", "other"])

    def test_hold_stops_new_starts(self):
        subprocess.run([sys.executable, str(TOOL), "hold", "--note", "game"], env=self.env, check=True, capture_output=True)
        p = self.job("held")
        time.sleep(1.0)
        self.assertFalse(self.events.exists())
        self.assertIn("on since", self.status())
        self.assertIn("ready, no Turnstile job is running", self.status())
        subprocess.run([sys.executable, str(TOOL), "release"], env=self.env, check=True, capture_output=True)
        self.assertEqual(p.wait(timeout=30), 0)
        self.assertTrue(self.events.exists())

    def test_waits_for_memory(self):
        self.free.write_text("1000")
        p = self.job("big", mem=8000)
        time.sleep(1.0)
        self.assertFalse(self.events.exists())
        self.free.write_text("12000")
        self.assertEqual(p.wait(timeout=30), 0)

    def test_waits_for_ram(self):
        self.set_available(12.5)  # 0.5 GiB beside the 12 GiB reserve; each job's cap is 1 GiB
        gpu, cpu = self.job("gpu"), self.cap_job("cpu")
        time.sleep(1.0)
        self.assertFalse(self.events.exists())
        self.set_available(20)
        self.assertEqual(gpu.wait(timeout=30), 0)
        self.assertEqual(cpu.wait(timeout=30), 0)
        self.assertEqual(sorted(n for k, n, _t in self.read_events() if k == "end"), ["cpu", "gpu"])

    def test_worker_waits_for_ram(self):
        self.set_available(12.5)  # 0.5 GiB beside the 12 GiB reserve; the worker's cap is 1 GiB
        w = self.worker("w", "normal", 5, self.script("w", 0.2)[1:])
        time.sleep(1.0)
        self.assertFalse(self.events.exists())
        self.set_available(20)
        self.assertEqual(w.wait(timeout=30), 0)
        self.assertTrue(self.events.exists())

    def test_reserved_run_blocks_until_it_finishes(self):
        later = dt.datetime.now() + dt.timedelta(hours=12)  # a one-minute clock window that cannot block here
        self.write_config(windows=[{"name": "sweep", "start": later.strftime("%H:%M"),
                                    "end": (later + dt.timedelta(minutes=1)).strftime("%H:%M"), "units": ["sweep.service", "board.service"]}])
        self.unit_state.write_text("active")  # a scheduled run going late or long
        other = self.job("other")
        time.sleep(1.0)
        self.assertFalse(self.events.exists())
        self.assertIn("Reserved runs: reserved run sweep.service is running", self.status())
        scheduled = self.job("scheduled", cls="scheduled")
        self.assertEqual(scheduled.wait(timeout=30), 0)  # the reserved runs' own class is not kept out
        self.unit_state.write_text("")  # unreadable state also blocks
        time.sleep(0.6)
        self.assertNotIn(("start", "other"), {(k, n) for k, n, _t in self.read_events()})
        self.unit_state.write_text("unknown")  # and so does a state systemd does not report as running or ended
        time.sleep(0.6)
        self.assertNotIn(("start", "other"), {(k, n) for k, n, _t in self.read_events()})
        self.assertIn("Reserved runs: the state of the reserved runs cannot be read", self.status())
        self.unit_state.write_text("inactive")
        self.assertEqual(other.wait(timeout=30), 0)

    def test_research_needs_its_task(self):
        r = subprocess.run([sys.executable, str(TOOL), "run", "--class", "research", "--project", "project-a", "--mem", "1",
                            "--max", "1", "--owner", "x", "--", "true"], env=self.env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)
        self.assertIn("--task", r.stderr)
        r = subprocess.run([sys.executable, str(TOOL), "run", "--class", "research", "--project", "project-a", "--task",
                            "packet-under-test", "--mem", "1", "--max", "1", "--owner", "x", "--", "true"],
                           env=self.env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(any(e["event"] == "enqueued" and e.get("task") == "packet-under-test" for e in self.log_events()))

    def test_short_run_deadline_moves_or_stops(self):
        moved = self.worker("short-a", "short", 0.02, ["sleep", "2.5"])  # deadline 1.2 s, normal slot free
        time.sleep(1.8)
        self.assertIn("Worker slots (normal): 1 of 1 in use", self.status())
        self.assertEqual(moved.wait(timeout=30), 0)
        self.assertTrue(any(e["event"] == "short_run_moved" for e in self.log_events()))
        busy = self.worker("normal-b", "normal", 5, ["sleep", "4"])
        time.sleep(0.5)
        stopped = self.worker("short-c", "short", 0.02, ["sleep", "6"])  # deadline 1.2 s, no normal slot free
        self.assertEqual(stopped.wait(timeout=30), 143)
        self.assertTrue(any(e["event"] == "short_run_stopped" for e in self.log_events()))
        self.assertEqual(busy.wait(timeout=30), 0)

    def test_short_run_that_ignores_sigterm_is_ended_after_the_grace(self):
        self.write_config(stop_grace_seconds=0.5)
        busy = self.worker("busy", "normal", 5, ["sleep", "4"])
        self.wait_for(self.tmp / "state" / "slots" / "normal-0.json")
        survived = self.tmp / "survived"
        stubborn = (f"import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(8)\n"
                    f"open({str(survived)!r}, 'w').close()\n")
        short = self.worker("stubborn", "short", 0.02, [sys.executable, "-c", stubborn])  # deadline 1.2 s, no normal slot free
        self.assertEqual(short.wait(timeout=30), 137)
        self.assertFalse(survived.exists())
        self.assertTrue(any(e["event"] == "short_run_killed" for e in self.log_events()))
        self.assertEqual(busy.wait(timeout=30), 0)

    def test_moved_short_run_keeps_its_normal_slot_after_its_launcher_dies(self):
        short = self.worker("first", "short", 0.01, self.script("first", 3)[1:])  # deadline 0.6 s, normal slot free
        self.wait_for(self.tmp / "state" / "slots" / "normal-0.json")
        short.kill()  # the launcher dies, as with its terminal; its job lives on and still holds normal-0
        short.wait()
        second = self.worker("second", "normal", 5, self.script("second", 0.2)[1:])
        self.assertEqual(second.wait(timeout=60), 0)
        ev = {(k, n): t for k, n, t in self.read_events()}
        self.assertIn(("end", "first"), ev, "the second worker finished while the moved job was still running")
        self.assertGreaterEqual(ev[("start", "second")], ev[("end", "first")])

    def test_stale_slot_record_does_not_hold_the_slot(self):
        slots = self.tmp / "state" / "slots"
        slots.mkdir(parents=True)
        (slots / "normal-0.json").write_text(json.dumps({"id": "normal-0", "owner": "ghost", "pid": 999999999,
                                                          "child_pid": 999999998, "started_at": "2026-10-16T00:00:00+01:00"}))
        w = self.worker("w", "normal", 5, ["true"])
        self.assertEqual(w.wait(timeout=30), 0)

    def test_an_unreadable_slot_record_keeps_its_slot_taken(self):
        slots = self.tmp / "state" / "slots"
        slots.mkdir(parents=True)
        (slots / "normal-0.json").write_text('{"id":')  # a record cut off part-way
        w = self.worker("w", "normal", 5, self.script("w", 0.1)[1:])
        time.sleep(1.0)
        self.assertFalse(self.events.exists())  # the slot stays taken
        self.assertIn("record cannot be read", self.status())
        (slots / "normal-0.json").unlink()
        self.assertEqual(w.wait(timeout=30), 0)
        self.assertTrue(self.events.exists())

    def test_an_unreadable_gpu_holder_record_keeps_gpu_jobs_waiting(self):
        state = self.tmp / "state"
        state.mkdir()
        (state / "holder.json").write_text('{"owner": "cut')
        p = self.job("after")
        time.sleep(1.0)
        self.assertFalse(self.events.exists())
        self.assertIn("holder record cannot be read", self.status())
        (state / "holder.json").unlink()
        self.assertEqual(p.wait(timeout=30), 0)
        self.assertTrue(self.events.exists())

    def orphaned_job_command(self):
        """A job whose main process starts a longer-lived process (holding none of the job's descriptors, as Python's
        subprocess does by default) and ends 0.8 s later; that process records its end 2.5 s after it started."""
        events = shlex.quote(str(self.events))
        grand = f"sleep 2.5; echo end grand $(date +%s.%N) >> {events}"
        return [sys.executable, "-c", f"import subprocess, time\nsubprocess.Popen(['sh', '-c', {grand!r}], stdin=subprocess.DEVNULL, "
                                      f"stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\ntime.sleep(0.8)\n"]

    def kill_launcher_after_start(self, launcher, record):
        """Kill Turnstile itself once the job's record names the job's process, as when its terminal is closed, and
        return once that main process has ended too."""
        until, child = time.time() + 30, None
        while child is None and time.time() < until:
            with contextlib.suppress(OSError, ValueError, KeyError):
                child = json.loads(record.read_text())["child_pid"]
            time.sleep(0.05)
        self.assertIsNotNone(child, "the job never started")
        launcher.kill()
        launcher.wait()
        until = time.time() + 30
        while turnstile.alive(child) and time.time() < until:
            time.sleep(0.05)
        self.assertFalse(turnstile.alive(child))

    def test_a_slot_stays_taken_while_an_orphaned_jobs_process_lives(self):
        first = self.worker("first", "normal", 5, self.orphaned_job_command())
        self.kill_launcher_after_start(first, self.tmp / "state" / "slots" / "normal-0.json")
        second = self.worker("second", "normal", 5, self.script("second", 0.1)[1:])
        self.assertEqual(second.wait(timeout=60), 0)
        ev = {(k, n): t for k, n, t in self.read_events()}
        self.assertIn(("end", "grand"), ev, "the second worker finished while the first job's process was still running")
        self.assertGreaterEqual(ev[("start", "second")], ev[("end", "grand")])

    def test_a_gpu_job_waits_while_an_orphaned_jobs_process_lives(self):
        cmd = [sys.executable, str(TOOL), "run", "--class", "other", "--mem", "100", "--max", "5", "--owner", "first", "--",
               *self.orphaned_job_command()]
        first = subprocess.Popen(cmd, env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self.reap, first)
        self.kill_launcher_after_start(first, self.tmp / "state" / "holder.json")
        second = self.job("second", seconds=0.1)
        self.assertEqual(second.wait(timeout=60), 0)
        ev = {(k, n): t for k, n, t in self.read_events()}
        self.assertIn(("end", "grand"), ev, "the second GPU job finished while the first job's process was still running")
        self.assertGreaterEqual(ev[("start", "second")], ev[("end", "grand")])

    def test_waiting_worker_interrupted_exits_130(self):
        busy = self.worker("busy", "normal", 5, ["sleep", "3"])
        self.wait_for(self.tmp / "state" / "slots" / "normal-0.json")
        waiting = self.worker("waiting", "normal", 5, ["true"])
        until = time.time() + 5
        while not catches(waiting.pid, signal.SIGTERM) and time.time() < until:
            time.sleep(0.05)
        time.sleep(0.2)
        waiting.send_signal(signal.SIGTERM)
        self.assertEqual(waiting.wait(timeout=30), 130)
        self.assertEqual(busy.wait(timeout=30), 0)

    def test_grant_held_until_all_processes_end(self):
        # The straggler lets go of the output pipes, so only Turnstile's own wait can keep this call running.
        events = shlex.quote(str(self.events))
        tail = (f"(sleep 1.5; echo end straggler $(date +%s.%N) >> {events}) </dev/null >/dev/null 2>&1 & "
                f"echo start straggler $(date +%s.%N) >> {events}")
        r = subprocess.run([sys.executable, str(TOOL), "run", "--class", "other", "--mem", "1", "--max", "1", "--owner", "s", "--",
                            "sh", "-c", tail], env=self.env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(("end", "straggler"), {(k, n) for k, n, _t in self.read_events()})
        self.assertIn("GPU: no Turnstile job running", self.status())

    def test_stopping_a_short_run_reaches_the_processes_it_started(self):
        self.write_config(stop_grace_seconds=0.5)
        busy = self.worker("busy", "normal", 5, ["sleep", "4"])
        self.wait_for(self.tmp / "state" / "slots" / "normal-0.json")
        survived = self.tmp / "survived"
        stubborn = (f"import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(8)\n"
                    f"open({str(survived)!r}, 'w').close()\n")
        short = self.worker("parent", "short", 0.02, ["sh", "-c", f"{shlex.quote(sys.executable)} -c {shlex.quote(stubborn)} & wait"])
        short.wait(timeout=30)
        self.assertFalse(survived.exists())  # the background process was ended too, not only the shell
        self.assertTrue(any(e["event"] == "short_run_killed" for e in self.log_events()))
        self.assertEqual(busy.wait(timeout=30), 0)

    def inproc_patches(self, state):
        """Patches for calling cmd_run in this process: its state under state, fake GPU and RAM readings, no adoption (this
        test process must not adopt other tests' processes), and every subprocess.Popen tracked so the job is reaped. The
        test process's signal handlers are restored afterwards."""
        children, popen = [], subprocess.Popen

        def tracked(*a, **kw):
            p = popen(*a, **kw)
            children.append(p)
            return p

        for signum in (signal.SIGINT, signal.SIGTERM):
            self.addCleanup(signal.signal, signum, signal.getsignal(signum))
        self.addCleanup(lambda: [p.wait(timeout=30) for p in children])
        return [mock.patch.object(turnstile, "STATE", state),
                mock.patch.object(turnstile, "config", return_value=json.loads(self.cfg.read_text())),
                mock.patch.object(turnstile, "gpu_memory", return_value={"total": 16376, "used": 0, "free": 16376, "procs": []}),
                mock.patch.object(turnstile, "available_gib", return_value=60.0),
                mock.patch.object(turnstile, "adopt_orphans", return_value=True, create=True),
                mock.patch.object(turnstile.subprocess, "Popen", side_effect=tracked)]

    @staticmethod
    def inproc_args(marker):
        return argparse.Namespace(ram=1, cls="other", project=None, task=None, mem=1, max=1, owner="inproc",
                                  command=["sh", "-c", f"sleep 0.8; echo done > {shlex.quote(str(marker))}"])

    def test_a_failed_record_write_after_the_start_keeps_the_grant(self):
        marker, state, stderr = self.tmp / "job-ended", self.tmp / "inproc-state", io.StringIO()
        write = turnstile.write_json

        def full_disk(path, record):
            if "child_pid" in record:
                raise OSError("synthetic: the state filesystem is full")
            write(path, record)

        with contextlib.ExitStack() as stack:
            for patch in self.inproc_patches(state) + [mock.patch.object(turnstile, "write_json", side_effect=full_disk)]:
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stderr(stderr))
            code = turnstile.cmd_run(self.inproc_args(marker))
        self.assertEqual(code, 0)
        self.assertTrue(marker.exists())  # cmd_run returned only after the job had ended
        self.assertIn("its record could not be written", stderr.getvalue())
        with open(state / "gpu.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # and only then was the grant free again

    def test_any_failure_after_the_start_waits_for_the_job(self):
        marker, state = self.tmp / "job-ended", self.tmp / "inproc-state"
        no_thread = mock.patch.object(turnstile.threading, "Timer", side_effect=RuntimeError("can't start new thread"))
        with contextlib.ExitStack() as stack:
            for patch in self.inproc_patches(state) + [no_thread]:
                stack.enter_context(patch)
            with self.assertRaises(RuntimeError):
                turnstile.cmd_run(self.inproc_args(marker))
        self.assertTrue(marker.exists())  # cmd_run raised only after the job had ended
        with open(state / "gpu.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_a_moved_run_whose_record_cannot_be_written_is_stopped(self):
        state, marker = self.tmp / "inproc-state", self.tmp / "ran-to-the-end"
        write = turnstile.write_json

        def full_disk(path, record):
            if path.name == "normal-0.json":
                raise OSError("synthetic: the state filesystem is full")
            write(path, record)

        args = argparse.Namespace(ram=1, kind="short", max=0.01, owner="moved",
                                  command=["sh", "-c", f"sleep 3; echo done > {shlex.quote(str(marker))}"])
        with contextlib.ExitStack() as stack:
            for patch in self.inproc_patches(state) + [mock.patch.object(turnstile, "write_json", side_effect=full_disk)]:
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            code = turnstile.cmd_worker(args)
        self.assertEqual(code, 143)  # stopped at its deadline, since its normal-slot record could not be written
        self.assertFalse(marker.exists())
        self.assertFalse((state / "slots" / "normal-0.json").exists())
        with open(state / "slots" / "normal-0.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # and the normal slot it had taken is free again

    def test_a_failed_log_or_terminal_write_still_stops_a_short_run(self):
        self.write_config(stop_grace_seconds=0.3)
        for failing in ("overrun", "short_run_stopped", "short_run_killed", "terminal"):
            with self.subTest(failing=failing):
                state, survived = self.tmp / f"inproc-{failing}", self.tmp / f"survived-{failing}"
                (state / "slots").mkdir(parents=True)
                log = turnstile.log

                def failing_log(event, **fields):
                    if event == failing:
                        raise OSError(f"synthetic: the log write for {event} failed")
                    log(event, **fields)

                stubborn = (f"import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(4)\n"
                            f"open({str(survived)!r}, 'w').close()\n")
                args = argparse.Namespace(ram=1, kind="short", max=0.01, owner="stubborn", command=[sys.executable, "-c", stubborn])
                with open(state / "slots" / "normal-0.lock", "a") as held:
                    fcntl.flock(held, fcntl.LOCK_EX)  # no normal slot is free
                    with contextlib.ExitStack() as stack:
                        for patch in self.inproc_patches(state) + [mock.patch.object(turnstile, "log", side_effect=failing_log)]:
                            stack.enter_context(patch)
                        stack.enter_context(contextlib.redirect_stderr(Broken() if failing == "terminal" else io.StringIO()))
                        code = turnstile.cmd_worker(args)
                self.assertEqual(code, 137)
                self.assertFalse(survived.exists())

    def test_refusals(self):
        r = subprocess.run([sys.executable, str(TOOL), "run", "--class", "research", "--project", "support-tool", "--mem", "1",
                            "--max", "1", "--owner", "x", "--", "true"], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        r = subprocess.run([sys.executable, str(TOOL), "worker", "--kind", "short", "--max", "20", "--owner", "x", "--", "true"],
                           env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        for args in (["cap", "--ram", "100"], ["cap", "--ram", "0"], ["run", "--class", "other", "--mem", "1", "--ram", "45"],
                     ["worker", "--kind", "normal", "--ram", "-1"]):
            r = subprocess.run([sys.executable, str(TOOL), *args, "--max", "1", "--owner", "x", "--", "true"],
                               env=self.env, capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 2, args)
            self.assertIn("--ram must be", r.stderr)

    def test_a_stale_ticket_is_ignored_by_status_and_removed_by_a_waiting_job(self):
        q = self.tmp / "state" / "queue"
        q.mkdir(parents=True)
        ghost = q / "0-999999999.json"
        ghost.write_text(json.dumps({"id": "0-999999999", "class": "other", "pid": 999999999, "enqueued_at": "0",
                                     "owner": "ghost", "mem_mib": 1, "max_minutes": 1}))
        r = subprocess.run([sys.executable, str(TOOL), "status"], env=self.env, capture_output=True, text=True, check=True)
        self.assertIn("Waiting: 0", r.stdout)
        self.assertIn("outside Turnstile: browser 134 MiB, ollama 9000 MiB", r.stdout)
        self.assertIn("RAM: 60.0 GiB available, 12 GiB kept in reserve", r.stdout)
        self.assertTrue(ghost.exists())  # status only reads
        self.assertEqual(self.job("after", seconds=0.1).wait(timeout=60), 0)
        self.assertFalse(ghost.exists())  # a waiting job removes a ticket whose launcher has gone

    @staticmethod
    def snapshot(folder):
        return {str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}

    @staticmethod
    def dead_record(**changes):
        return {"pid": 999999999, "owner": "ghost", "id": "old", "class": "other", "started_at": "2026-10-16T09:00:00+01:00",
                "expected_finish": "2026-10-16T09:05:00+01:00", **changes}

    def test_status_changes_nothing(self):
        """Status only reads: the records of jobs that have ended (a GPU holder, a capped job, a waiting ticket) are left
        as they are, and nothing is logged."""
        state = self.tmp / "state"
        for folder in ("queue", "capped"):
            (state / folder).mkdir(parents=True)
        (state / "holder.json").write_text(json.dumps(self.dead_record()))
        (state / "capped" / "0-999999999.json").write_text(json.dumps(self.dead_record(id="0-999999999")))
        (state / "queue" / "0-999999999.json").write_text(json.dumps({"id": "0-999999999", "class": "other", "pid": 999999999,
                                                                      "enqueued_at": "0", "owner": "ghost", "mem_mib": 1,
                                                                      "max_minutes": 1}))
        before = self.snapshot(state)
        out = self.status()
        self.assertEqual(self.snapshot(state), before)
        self.assertIn("GPU: no Turnstile job running", out)
        self.assertIn("Waiting: 0", out)
        self.assertIn("Capped jobs without GPU: 0", out)

    def test_status_never_removes_a_holder_record_it_read_as_stale(self):
        """Status reads the holder record of a job that has ended; before it acts on that, a new GPU job replaces the
        record. The new record must survive: once the new job's launcher and main process are gone, it is all that keeps a
        second GPU job from starting beside the new job's other processes."""
        state = self.tmp / "inproc-state"
        state.mkdir()
        holder = state / "holder.json"
        turnstile.write_json(holder, self.dead_record())
        new = {"pid": os.getpid(), "pid_started": turnstile.proc_start(os.getpid()), "owner": "new", "id": "new",
               "class": "other", "started_at": "2026-10-16T10:00:00+01:00", "expected_finish": "2026-10-16T10:05:00+01:00"}
        read = turnstile.record_state

        def read_then_replaced(path):
            result = read(path)
            if pathlib.Path(path).name == "holder.json":
                turnstile.write_json(holder, new)
            return result

        with contextlib.ExitStack() as stack:
            for patch in self.inproc_patches(state) + [mock.patch.object(turnstile, "record_state", side_effect=read_then_replaced)]:
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(turnstile.cmd_status(None), 0)
        self.assertTrue(holder.exists(), "status removed the new job's holder record")
        self.assertEqual(json.loads(holder.read_text())["id"], "new")

    def test_hold_names_only_a_holder_that_still_runs(self):
        state = self.tmp / "state"
        state.mkdir()
        (state / "holder.json").write_text(json.dumps(self.dead_record()))
        r = subprocess.run([sys.executable, str(TOOL), "hold", "--note", "test"], env=self.env, capture_output=True,
                           text=True, check=True)
        self.assertIn("Nothing is running", r.stdout)
        self.assertNotIn("ghost", r.stdout)

    def test_after_a_failed_record_write_and_a_killed_launcher_inheriting_processes_hold_the_gpu_lock(self):
        """When the job's record cannot be written after the start and Turnstile is then killed, the GPU lock stays with
        every process that kept the descriptor the job inherited: here a process the job forked, after the job's own
        process has ended."""
        grand_pid, primary_pid, ended = self.tmp / "grand.pid", self.tmp / "primary.pid", self.tmp / "grand.end"
        job = (f"import os, time\nfrom pathlib import Path\nif os.fork() == 0:\n"
               f"    Path({str(grand_pid)!r}).write_text(str(os.getpid()))\n    time.sleep(2.5)\n    Path({str(ended)!r}).touch()\n"
               f"else:\n    Path({str(primary_pid)!r}).write_text(str(os.getpid()))\n    time.sleep(0.5)\n")
        wrapper = (f"import importlib.util, sys\nspec = importlib.util.spec_from_file_location('ts', {str(TOOL)!r})\n"
                   "ts = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(ts)\nwrite = ts.write_json\n\n"
                   "def fail_after_start(path, record):\n    if 'child_pid' in record:\n"
                   "        raise OSError('synthetic: the record cannot be written')\n    write(path, record)\n\n"
                   "ts.write_json = fail_after_start\nraise SystemExit(ts.main(sys.argv[1:]))\n")
        launcher = subprocess.Popen([sys.executable, "-c", wrapper, "run", "--class", "other", "--mem", "100", "--max", "5",
                                     "--owner", "first", "--", sys.executable, "-c", job],
                                    env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self.reap, launcher)
        self.wait_for(grand_pid)
        self.wait_for(primary_pid)
        launcher.kill()
        launcher.wait()
        primary, grand = int(primary_pid.read_text()), int(grand_pid.read_text())
        until = time.time() + 30
        while turnstile.alive(primary) and time.time() < until:
            time.sleep(0.05)
        self.assertFalse(turnstile.alive(primary))
        self.assertTrue(turnstile.alive(grand))
        with open(self.tmp / "state" / "gpu.lock", "a") as lock:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.wait_for(ended)
        until = time.time() + 30
        while turnstile.alive(grand) and time.time() < until:
            time.sleep(0.05)
        with open(self.tmp / "state" / "gpu.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # free once that process has ended


@unittest.skipUnless(groups_work(), "systemd-run --user --scope does not work here")
class MemoryGroups(Process):
    groups = True

    def test_job_runs_in_its_own_group(self):
        out = shlex.quote(str(self.tmp / "cgroup"))
        probe = f"cat /proc/self/cgroup > {out}; cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.max >> {out}"
        r = subprocess.run([sys.executable, str(TOOL), "cap", "--max", "1", "--owner", "g", "--", "sh", "-c", probe],
                           env=self.env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        group, cap = (self.tmp / "cgroup").read_text().splitlines()[:2]
        self.assertIn("turnstile-cap-", group)
        self.assertNotEqual(group, pathlib.Path("/proc/self/cgroup").read_text().strip())
        self.assertEqual(int(cap), 1 << 30)

    # This test makes the kernel stop a capped job on purpose, which the desktop reports as "the system is low on
    # memory". It runs only on request (TURNSTILE_LIVE_OOM_TEST=1), after any change to Turnstile's caps.
    @unittest.skipUnless(os.environ.get("TURNSTILE_LIVE_OOM_TEST") == "1",
                         "opt-in: deliberately OOM-kills a capped job; set TURNSTILE_LIVE_OOM_TEST=1")
    def test_runaway_job_is_stopped_alone(self):
        grow = "import time\nb = []\nfor i in range(400):\n    b.append(bytearray(8 << 20)); time.sleep(0.01)\nprint('not stopped')"
        r = subprocess.run([sys.executable, str(TOOL), "cap", "--ram", "0.125", "--max", "1", "--owner", "runaway", "--",
                            sys.executable, "-I", "-c", grow], env=self.env, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 137, r.stderr)
        self.assertNotIn("not stopped", r.stdout)
        self.assertIn("reached its memory cap of 0.125 GiB", r.stderr)
        self.assertTrue(any(e["event"] == "memory_cap_reached" and e["owner"] == "runaway" for e in self.log_events()))

    def test_gpu_lock_stays_with_an_orphaned_job(self):
        first = self.job("orphan", seconds=3)
        deadline = time.time() + 30
        while not self.events.exists() and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(0.3)
        first.kill()  # Turnstile itself dies, as it would with its terminal; the job lives on in its own group
        first.wait()
        r = subprocess.run([sys.executable, str(TOOL), "status"], env=self.env, capture_output=True, text=True, check=True)
        self.assertIn("GPU: running orphan", r.stdout)
        self.assertIn("Turnstile itself has gone, the job still runs", r.stdout)
        second = self.job("next", seconds=0.2)
        self.assertEqual(second.wait(timeout=60), 0)
        ev = {(k, n): t for k, n, t in self.read_events()}
        self.assertGreaterEqual(ev[("start", "next")], ev[("end", "orphan")])

    def test_peak_recorded(self):
        hold = "import time\nb = bytearray(64 << 20)\nfor i in range(0, len(b), 4096): b[i] = 1\ntime.sleep(1.5)"
        r = subprocess.run([sys.executable, str(TOOL), "cap", "--ram", "1", "--max", "1", "--owner", "peak", "--",
                            sys.executable, "-I", "-c", hold], env=self.env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        done = [e for e in self.log_events() if e["event"] == "cap_finished" and e["owner"] == "peak"]
        self.assertGreaterEqual(done[0]["peak_gib"], 0.06)


class ShippedDefaults(unittest.TestCase):
    """The example ships its settings and a registry beside the script. Without TURNSTILE_REGISTRY the research class reads
    the registry beside the script; its example project is admitted and its support tool is refused."""

    def test_default_registry_is_beside_the_script(self):
        env = {k: v for k, v in os.environ.items() if k != "TURNSTILE_REGISTRY"}
        code = "import turnstile; print(turnstile.REGISTRY == turnstile.HERE / 'registry.json')"
        r = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(HERE.parent), env=env, capture_output=True, text=True,
                           timeout=60)
        self.assertEqual(r.stdout.strip(), "True", r.stderr)

    def test_shipped_registry_admits_its_example_project_only(self):
        reg = json.loads((HERE.parent / "registry.json").read_text(encoding="utf-8"))
        self.assertTrue(turnstile.research_ok("example-project", reg)[0])
        self.assertFalse(turnstile.research_ok("example-support-tool", reg)[0])

    def test_shipped_settings_parse(self):
        cfg = json.loads((HERE.parent / "turnstile.json").read_text(encoding="utf-8"))
        self.assertEqual((cfg["normal_slots"], cfg["short_slots"], cfg["windows"]), (6, 2, []))


if __name__ == "__main__":
    unittest.main()
