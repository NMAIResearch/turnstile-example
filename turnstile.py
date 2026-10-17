#!/usr/bin/env python3
"""Turnstile: a shared GPU queue and worker slots for every agent session on one Linux workstation (a worked example).

One GPU job at a time. While the owner holds the GPU (for example to play a game) nothing new starts. Then scheduled runs in
their reserved window, then jobs for a registered research project (named with the task that puts the job in the
project's approved scope), then everything else, first come, first served. A reserved window lasts from its warning to
its end time, and for as long as one of its runs (systemd user units) is still going, including a run that starts late,
or its state cannot be read (any state but a known running, inactive or failed one).
A running job is never pre-empted. Each job declares the GPU memory it needs and a maximum run time; Turnstile starts it
only when that memory is free, and reports, never stops, a GPU job that runs past its maximum. A job's grant is released
only when all of its processes have ended, whatever fails after the job has started and also when Turnstile itself is
killed: a slot or the GPU stays taken while the job's record names a process, memory group or session that still has
processes, and while that record cannot be read. Worker slots: six normal and two
short-run slots machine-wide (short runs: read-only, no GPU, 15 minutes at most, declared by the caller); at its deadline
a short run that is still going also takes a free normal slot, or is stopped when none is free (SIGTERM, then SIGKILL
after stop_grace_seconds), and its short slot stays taken until it ends. A slot whose record names a job that still
lives stays taken, so a short run moved to a normal slot keeps it even if its launcher dies; when that slot's record
cannot be written, the short run is stopped instead. A failed log or terminal write never keeps a short run from being
stopped.

Memory groups: every job Turnstile starts runs in its own memory group (a transient systemd scope) with a hard RAM cap,
outside the terminal that started it, and starts only when its cap fits in available memory beside a reserve. A job
that outgrows its cap is stopped alone, all its processes together, instead of the out-of-memory killer closing the
whole terminal tab with the agent and every job in it. The GPU lock and a worker slot
stay with the job's processes, so a job that outlives its terminal still holds them. `cap` gives a heavy job that needs
no GPU (a render with parallel workers) the same memory group without joining the GPU queue. A memory group whose state
cannot be read counts as still running. With memory_groups off, jobs have no RAM cap; Turnstile then adopts the
processes a job leaves behind (Linux PR_SET_CHILD_SUBREAPER) and keeps the grant until they have ended too; each job
also runs in a session of its own, named in its record, so its grant stays taken after Turnstile is killed while any
process of that session lives. A process that starts a session of its own leaves this tracking.

  turnstile.py status
  turnstile.py hold [--note TEXT]        the owner only
  turnstile.py release                   the owner only
  turnstile.py run --class scheduled|research|other [--project ID --task PACKET] --mem MIB [--ram GIB] --max MINUTES --owner NAME -- COMMAND...
  turnstile.py cap [--ram GIB] --max MINUTES --owner NAME -- COMMAND...
  turnstile.py worker --kind normal|short [--ram GIB] --max MINUTES --owner NAME -- COMMAND...

State: ~/.local/state/turnstile/ (TURNSTILE_STATE overrides). Settings: turnstile.json beside this script
(TURNSTILE_CONFIG overrides). Projects for the research class: registry.json beside this script (TURNSTILE_REGISTRY
overrides). The GPU lock is an advisory flock: it binds only processes that use Turnstile.
Records are replaced whole (a temporary file renamed into place), so a reader never sees part of one. `status` only
reads: it changes and removes nothing, so it can never take a grant from a job; the record of a job that has ended stays
until a new job replaces it, and waiting jobs remove tickets whose launcher has gone.
Exit status: the command's own (128 plus the signal number when a signal ended it, so 137 when its memory cap stopped
it), 2 for a refused request, 130 when interrupted while waiting."""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import datetime as dt
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = Path(os.environ.get("TURNSTILE_STATE", Path.home() / ".local/state/turnstile"))
CONFIG = Path(os.environ.get("TURNSTILE_CONFIG", HERE / "turnstile.json"))
REGISTRY = Path(os.environ.get("TURNSTILE_REGISTRY", HERE / "registry.json"))
SMI = os.environ.get("TURNSTILE_SMI", "nvidia-smi")
UNITCTL = os.environ.get("TURNSTILE_UNITCTL", "systemctl")
MEMINFO = Path(os.environ.get("TURNSTILE_MEMINFO", "/proc/meminfo"))
CGROUPS = Path("/sys/fs/cgroup")
RANK = {"scheduled": 0, "research": 1, "other": 2}
GIB = 1 << 30
PR_SET_CHILD_SUBREAPER = 36


def now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def log(event: str, **fields) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": now().isoformat(timespec="seconds"), "event": event, **fields}, ensure_ascii=False) + "\n")


def note(event: str, **fields) -> None:
    """log(), except that a failed write is dropped: for Turnstile's timers, its waits for a job's processes, its
    signal handlers and the steps that release a grant, which a log line must never stop."""
    with contextlib.suppress(OSError, ValueError, TypeError):
        log(event, **fields)


def say(text: str) -> None:
    """A message for the person at the terminal; a terminal that has gone away never stops Turnstile's own work."""
    with contextlib.suppress(OSError, ValueError):
        print(f"turnstile: {text}", file=sys.stderr, flush=True)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def proc_start(pid: int) -> str | None:
    """The process's start time (clock ticks after boot, field 22 of /proc/PID/stat); None when it cannot be read."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="latin-1")
    except OSError:
        return None
    fields = text[text.rfind(")") + 2:].split()
    return fields[19] if len(fields) > 19 else None


def running(pid: int | None, start: str | None = None) -> bool:
    """A process with this pid lives and, when its start time was recorded, is the same process: a pid the system has
    since given to another process does not count."""
    if not pid or not alive(pid):
        return False
    return start is None or proc_start(pid) in (None, start)


def session_alive(session: int | None) -> bool:
    """Whether a process that is not a zombie still belongs to this session (field 6 of /proc/PID/stat). The system
    gives a session's id to another process only after every process of the session has ended."""
    if not session:
        return False
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = stat.read_text(encoding="latin-1")
        except OSError:
            continue
        fields = text[text.rfind(")") + 2:].split()
        if len(fields) > 3 and fields[3] == str(session) and fields[0] != "Z":
            return True
    return False


def job_alive(record: dict) -> bool:
    """A job lives while Turnstile or the job's own process does, while its session still has processes, or while its
    memory group has processes or cannot be read: a job can outlive a killed terminal, and its other processes can
    outlive its main process."""
    if running(record.get("pid"), record.get("pid_started")) or running(record.get("child_pid"), record.get("child_started")):
        return True
    if session_alive(record.get("session")):
        return True
    return bool(record.get("scope")) and scope_state(record["scope"]) != "ended"


def research_ok(project: str | None, registry: dict) -> tuple[bool, str]:
    """A research claim needs a registered project outside the support lane that is not closed."""
    if not project:
        return False, "the research class needs --project with a registry id"
    for p in registry.get("projects", []):
        if p.get("id") == project:
            if p.get("lane") == "support":
                return False, f"{project} is a support project, not research"
            if p.get("status") == "closed":
                return False, f"{project} is closed"
            return True, ""
    return False, f"{project} is not in the project registry"


def order(tickets: list[dict]) -> list[dict]:
    return sorted(tickets, key=lambda t: (RANK[t["class"]], t["enqueued_at"], t["id"]))


def window_block(at: dt.datetime, windows: list[dict], max_minutes: float, cls: str) -> dict | None:
    """The window a job may not start into: one open now, or one it would overlap given its declared maximum."""
    if cls == "scheduled":
        return None
    end_of_job = at + dt.timedelta(minutes=max_minutes)
    for w in windows:
        for day in (at.date() - dt.timedelta(days=1), at.date(), at.date() + dt.timedelta(days=1)):
            start = dt.datetime.combine(day, dt.time.fromisoformat(w["start"]), at.tzinfo)
            end = dt.datetime.combine(day, dt.time.fromisoformat(w["end"]), at.tzinfo)
            if end <= start:
                end += dt.timedelta(days=1)
            if start <= at < end or (at < start < end_of_job):
                return {"name": w["name"], "start": start.isoformat(timespec="minutes"), "end": end.isoformat(timespec="minutes")}
    return None


RUNNING = ("active", "activating", "deactivating", "reloading", "refreshing")
ENDED = ("inactive", "failed")


def reserved_running(windows: list[dict]) -> str | None:
    """Why the reserved runs block a start now: one of them is still going (it may have started late, or run past its
    window's end), or their state cannot be read. Only the states in RUNNING and ENDED are read; any other answer, such
    as unknown, counts as unreadable."""
    units = [u for w in windows for u in w.get("units", [])]
    if not units:
        return None
    try:
        r = subprocess.run([UNITCTL, "--user", "is-active", *units], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return "the state of the reserved runs cannot be read"
    states = r.stdout.split()
    if len(states) != len(units) or any(s not in RUNNING + ENDED for s in states):
        return "the state of the reserved runs cannot be read"
    running_units = [u for u, s in zip(units, states) if s in RUNNING]
    return f"reserved run {running_units[0]} is running" if running_units else None


def gpu_memory() -> dict:
    """Total, used and free MiB and the compute processes, from nvidia-smi; None values when it cannot be read."""
    try:
        g = subprocess.run([SMI, "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20, check=True).stdout.strip().splitlines()[0]
        total, used, free = (int(x) for x in g.split(","))
        apps = subprocess.run([SMI, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20, check=True).stdout.strip().splitlines()
        procs = []
        for line in apps:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3 and parts[0].isdigit():
                # A process name can itself contain commas (a browser's command line), so the memory is the last field.
                name = ",".join(parts[1:-1]).strip().split()
                procs.append({"pid": int(parts[0]), "name": os.path.basename(name[0]) if name else "?",
                              "mib": int(parts[-1]) if parts[-1].isdigit() else None})
        return {"total": total, "used": used, "free": free, "procs": procs}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"total": None, "used": None, "free": None, "procs": []}


def available_gib() -> float | None:
    """MemAvailable from /proc/meminfo in GiB; None when it cannot be read."""
    try:
        for line in MEMINFO.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1 << 20)
    except (OSError, ValueError, IndexError):
        pass
    return None


def ram_short(cfg: dict, ram: float) -> str | None:
    """Why a job with this cap cannot start yet: it waits until its cap fits in available memory beside the reserve."""
    avail = available_gib()
    reserve = cfg.get("ram_reserve_gib", 12)
    if avail is None:
        return "available memory cannot be read"
    if avail - reserve < ram:
        return f"needs {ram:g} GiB of RAM beside a {reserve:g} GiB reserve, {avail:.1f} GiB available"
    return None


def ram_cap(a, cfg: dict, default_key: str) -> tuple[float | None, str]:
    ram = a.ram if a.ram is not None else cfg.get(default_key, 24)
    top = cfg.get("ram_cap_max_gib", 44)
    if not 0 < ram <= top:
        return None, f"--ram must be above 0 and at most {top:g} GiB, so the rest stays free for the desktop and the agents"
    return ram, ""


def grouped(cfg: dict, unit: str, ram: float, command: list[str]) -> tuple[list[str], str | None]:
    """The command wrapped so systemd-run starts it in its own transient scope under the user manager, with hard RAM
    and swap caps; OOMPolicy=kill stops all of the job's processes together when one of them is killed for memory.
    Returns the command and the scope's unit name (None when memory groups are switched off in the settings)."""
    if not cfg.get("memory_groups", True):
        return command, None
    return ["systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}", "-p", f"MemoryMax={int(ram * 1024)}M",
            "-p", f"MemorySwapMax={int(cfg.get('swap_cap_gib', 4) * 1024)}M", "-p", "OOMPolicy=kill", "--", *command], unit


def groups_problem(cfg: dict) -> str | None:
    """None when memory groups work here, or when they are switched off in the settings and Turnstile can adopt the
    job's processes instead; otherwise why neither works."""
    if not cfg.get("memory_groups", True):
        return None if adopt_orphans() else "memory groups are off and the kernel refused to let Turnstile adopt the job's processes"
    try:
        r = subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "--collect", "-p", "MemoryMax=64M", "--", "true"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"systemd-run cannot start: {exc}"
    return None if r.returncode == 0 else f"systemd-run failed: {r.stderr.strip()[:200] or r.returncode}"


def refuse_without_group(cfg: dict) -> bool:
    why = groups_problem(cfg)
    if why:
        print(f"turnstile: refused: no memory group for the job ({why}). Stop and say so (stop condition); "
              "do not start the job outside Turnstile instead.", file=sys.stderr)
    return bool(why)


def cgroup_of(pid: int) -> Path | None:
    try:
        line = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").strip().splitlines()[0]
    except (OSError, IndexError):
        return None
    return CGROUPS / line.split("::", 1)[-1].lstrip("/")


def ram_now_gib(pid: int | None) -> float | None:
    cg = cgroup_of(pid) if pid else None
    try:
        return int((cg / "memory.current").read_text()) / GIB if cg else None
    except (OSError, ValueError):
        return None


def tickets(clean: bool = True) -> list[dict]:
    """The waiting tickets in queue order. A ticket whose launcher has gone is left out, and with clean (waiting jobs)
    also removed; status lists with clean off, since it only reads."""
    out = []
    for p in sorted((STATE / "queue").glob("*.json")):
        try:
            t = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not alive(t["pid"]):
            if clean:
                p.unlink(missing_ok=True)
                log("stale_ticket_removed", ticket=t["id"], owner=t.get("owner"))
            continue
        out.append(t)
    return order(out)


def read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def record_state(p: Path) -> tuple[str, dict | None]:
    """('none', None) when there is no record, ('ok', record) for a readable one (a JSON object naming a pid), and
    ('unreadable', None) for anything else: a partial, damaged or foreign file is a record that cannot be read."""
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "none", None
    except (OSError, ValueError):
        return "unreadable", None
    try:
        record = json.loads(text)
    except ValueError:
        return "unreadable", None
    if not isinstance(record, dict) or not isinstance(record.get("pid"), int):
        return "unreadable", None
    return "ok", record


def write_json(p: Path, record: dict) -> None:
    """Write a record whole: a temporary file beside it, flushed to disk, then renamed over it, so a reader sees the
    earlier record or the new one, never part of one. A failed write leaves the earlier record and no temporary file."""
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    with contextlib.suppress(OSError):
        folder = os.open(p.parent, os.O_RDONLY)
        try:
            os.fsync(folder)
        finally:
            os.close(folder)


def notify(text: str) -> None:
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", "Turnstile", text], capture_output=True, timeout=10)


def process_table() -> dict[int, tuple[int, str]]:
    """Every process's parent pid and state letter, read from /proc."""
    table = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = stat.read_text(encoding="latin-1")
        except OSError:
            continue
        fields = text[text.rfind(")") + 2:].split()
        if len(fields) > 1 and fields[1].isdigit():
            table[int(stat.parent.name)] = (int(fields[1]), fields[0])
    return table


def descendants(root: int) -> list[int]:
    """The processes below root in the process tree."""
    table, found, frontier = process_table(), set(), {root}
    while frontier:
        frontier = {pid for pid, (parent, _state) in table.items() if parent in frontier} - found
        found |= frontier
    return sorted(found)


def adopt_orphans() -> bool:
    """Make this process the parent of every process its job leaves behind (Linux PR_SET_CHILD_SUBREAPER), so that
    without a memory group Turnstile can still wait for all of them; False when the kernel refuses."""
    try:
        return ctypes.CDLL(None, use_errno=True).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0
    except (OSError, AttributeError):
        return False


def wait_for_adopted(record: dict) -> None:
    """Without a memory group, keep the grant until every process the job left behind has ended. Turnstile adopted them
    (adopt_orphans), so they are its own children, and it reaps each one as it ends."""
    since, said = time.monotonic(), False
    while True:
        me = os.getpid()
        kids = [(pid, state) for pid, (parent, state) in process_table().items() if parent == me]
        if not kids:
            return
        for pid, state in kids:
            if state == "Z":
                with contextlib.suppress(ChildProcessError):
                    os.waitpid(pid, os.WNOHANG)
        if all(state == "Z" for _pid, state in kids):
            # The processes a reaped one left behind are adopted before it can be reaped, so look again at once.
            time.sleep(0.01)
            continue
        if not said and time.monotonic() - since > 1:
            say("the job's main process ended; waiting for the processes it left behind before releasing")
            note("waiting_for_processes", id=record.get("id"), owner=record.get("owner"))
            said = True
        time.sleep(0.2)


def scope_result(unit: str) -> str | None:
    """systemd's verdict on a finished job scope ('oom-kill' when its memory cap stopped it); clears a failed scope."""
    name = f"{unit}.scope"
    try:
        r = subprocess.run(["systemctl", "--user", "show", "-p", "Result", "--value", name], capture_output=True, text=True, timeout=20)
        subprocess.run(["systemctl", "--user", "reset-failed", name], capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None


def scope_state(unit: str) -> str:
    """'running' while the job's memory group still has processes, 'ended' once systemd reports it inactive or failed,
    and 'unknown' when its state cannot be read: a timeout, a bus failure or a state not in RUNNING or ENDED."""
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", f"{unit}.scope"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    state = r.stdout.strip()
    return "running" if state in RUNNING else "ended" if state in ENDED else "unknown"


def signal_job(child: subprocess.Popen, unit: str | None, signum: int) -> None:
    """Signal the whole job: every process in its memory group (through systemd, also when the group's state cannot be
    read), or the job's own process when systemd refuses; without a memory group, the job's own process and every
    process Turnstile adopted from it."""
    if unit and scope_state(unit) != "ended":
        try:
            done = subprocess.run(["systemctl", "--user", "kill", f"--signal={signal.Signals(signum).name}", f"{unit}.scope"],
                                  capture_output=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            done = None
        if done is not None and done.returncode == 0:
            return
    if unit is None:
        for pid in descendants(os.getpid()):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signum)
    elif child.poll() is None:
        child.send_signal(signum)


def wait_for_scope(unit: str, record: dict) -> None:
    """Keep the grant until every process of the job has ended, not only its main process. A memory group whose state
    cannot be read counts as still running, so a systemctl timeout or bus failure never releases the grant early."""
    since, said, unread = time.monotonic(), False, False
    while (state := scope_state(unit)) != "ended":
        if state == "unknown" and not unread:
            say("the state of the job's memory group cannot be read; keeping its grant until it can be")
            note("scope_state_unreadable", id=record.get("id"), owner=record.get("owner"))
            unread = True
        elif state == "running" and not said and time.monotonic() - since > 1:
            say("the job's main process ended; waiting for its other processes before releasing")
            note("waiting_for_processes", id=record.get("id"), owner=record.get("owner"))
            said = True
        time.sleep(0.2)


def wait_until_ended(child: subprocess.Popen, unit: str | None, record: dict) -> None:
    """Return only once every process of the job has ended: its own process, then the rest of its memory group or,
    without one, the processes Turnstile adopted from it."""
    child.wait()
    if unit:
        wait_for_scope(unit, record)
    else:
        wait_for_adopted(record)


def run_child(cmd: list[str], max_minutes: float, record: dict, unit: str | None = None, keep: tuple[int, ...] = (),
              on_start=None, on_overrun=None) -> int:
    """Run the job and return its exit status once all its processes have ended. keep passes lock descriptors to the
    job, so its processes hold the lock even if Turnstile itself is killed; unit names the job's memory group, whose
    peak use is recorded (without one, Turnstile waits for the processes it adopted from the job); on_overrun runs after
    the overrun report at the declared maximum, given a function that signals the whole job and an event that is set
    once the job has ended. A failure after the job has started (writing its record, starting a watcher) never ends the
    wait early, so the grant is never released while the job's processes live."""
    child, pending = None, []

    def forward(signum, _frame):
        if child is None:
            pending.append(signum)
        else:
            signal_job(child, unit, signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    child = subprocess.Popen(cmd, pass_fds=keep, start_new_session=unit is None)
    finished, peak, timer, watcher = threading.Event(), {"bytes": None}, None, None
    try:
        for signum in pending:
            signal_job(child, unit, signum)
        record.update(child_pid=child.pid, child_started=proc_start(child.pid), scope=unit,
                      **({"session": child.pid} if unit is None else {}))
        if on_start:
            try:
                on_start()
            except OSError as error:
                say(f"the job started, but its record could not be written ({type(error).__name__}); it keeps its grant "
                    "until all its processes have ended")

        def watch():
            cg = None
            while not finished.is_set():
                if cg is None:
                    c = cgroup_of(child.pid)
                    cg = c if c is not None and c.name == f"{unit}.scope" else None
                if cg is not None:
                    try:
                        peak["bytes"] = int((cg / "memory.peak").read_text())
                    except (OSError, ValueError):
                        pass
                finished.wait(0.5)

        if unit:
            watcher = threading.Thread(target=watch, daemon=True)
            watcher.start()

        def overrun():
            note("overrun", **{k: record.get(k) for k in ("id", "owner", "max_minutes")})
            if on_overrun:
                on_overrun(lambda signum: signal_job(child, unit, signum), finished)
                return
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                notify(f"{record['owner']}: job {record['id']} is past its declared {record['max_minutes']} min; it keeps running until the owner decides")

        timer = threading.Timer(max_minutes * 60, overrun)
        timer.daemon = True
        timer.start()
        code = child.wait()
    finally:
        wait_until_ended(child, unit, record)
        finished.set()
        if timer:
            timer.cancel()
        if watcher:
            watcher.join(timeout=5)
    code = 128 - code if code < 0 else code
    record["exit"] = code
    record["peak_gib"] = round(peak["bytes"] / GIB, 2) if peak["bytes"] else None
    if unit and scope_result(unit) == "oom-kill":
        record["memory_cap_reached"] = True
        note("memory_cap_reached", **{k: record.get(k) for k in ("id", "owner", "ram_gib", "peak_gib")})
        say(f"the job reached its memory cap of {record.get('ram_gib')} GiB and was stopped, alone; the terminal and other "
            "jobs were not touched. Give it fewer parallel workers or a higher --ram within the settings' maximum.")
    return code


def wait_for_ram(cfg: dict, ram: float) -> None:
    last = None
    while True:
        reason = ram_short(cfg, ram)
        if reason is None:
            return
        if reason != last:
            print(f"turnstile: waiting: {reason}", file=sys.stderr, flush=True)
            last = reason
        time.sleep(cfg.get("poll_seconds", 5))


def cmd_run(a) -> int:
    cfg = config()
    ram, why = ram_cap(a, cfg, "job_ram_gib")
    if ram is None:
        print(f"turnstile: refused: {why}", file=sys.stderr)
        return 2
    if a.cls == "research":
        ok, why = research_ok(a.project, json.loads(REGISTRY.read_text(encoding="utf-8")))
        if ok and not a.task:
            ok, why = False, "the research class needs --task naming the packet or task that puts this job in the project's approved scope"
        if not ok:
            print(f"turnstile: refused: {why}", file=sys.stderr)
            return 2
    if refuse_without_group(cfg):
        return 2
    (STATE / "queue").mkdir(parents=True, exist_ok=True)
    t = {"id": f"{time.time_ns()}-{os.getpid()}", "class": a.cls, "project": a.project, "task": a.task, "mem_mib": a.mem,
         "ram_gib": ram, "max_minutes": a.max, "owner": a.owner, "pid": os.getpid(), "pid_started": proc_start(os.getpid()),
         "enqueued_at": now().isoformat(timespec="microseconds"), "command": " ".join(a.command)[:200]}
    ticket = STATE / "queue" / f"{t['id']}.json"
    write_json(ticket, t)
    log("enqueued", ticket=t["id"], owner=a.owner, cls=a.cls, project=a.project, task=a.task, mem_mib=a.mem, ram_gib=ram,
        max_minutes=a.max)
    lock = open(STATE / "gpu.lock", "a")
    last = None

    def leave(signum, _frame):
        with contextlib.suppress(OSError):
            ticket.unlink(missing_ok=True)
        note("left_queue", ticket=t["id"], owner=a.owner, signal=signum)
        sys.exit(130)

    def blocked() -> str | None:
        if (STATE / "hold.json").exists():
            return "the owner holds the GPU"
        ahead = [x for x in tickets() if x["id"] != t["id"] and (RANK[x["class"]], x["enqueued_at"], x["id"]) < (RANK[a.cls], t["enqueued_at"], t["id"])]
        if ahead:
            return f"{len(ahead)} job(s) ahead"
        w = window_block(now(), cfg.get("windows", []), a.max, a.cls)
        if w:
            return f"reserved window {w['name']} ({w['start']} to {w['end']})"
        if a.cls != "scheduled" and (busy := reserved_running(cfg.get("windows", []))):
            return busy
        held, holder_record = record_state(STATE / "holder.json")
        if held == "unreadable":
            return f"the GPU holder record cannot be read ({STATE / 'holder.json'}); no GPU job starts until it is removed"
        if held == "ok" and job_alive(holder_record):
            return f"the GPU is still in use by {holder_record.get('owner', 'another')}'s job"
        mem = gpu_memory()
        if mem["free"] is None:
            return "GPU memory cannot be read"
        if mem["free"] < a.mem + cfg.get("memory_margin_mib", 512):
            return f"needs {a.mem} MiB plus margin, {mem['free']} MiB free"
        return ram_short(cfg, ram)

    signal.signal(signal.SIGTERM, leave)
    signal.signal(signal.SIGINT, leave)
    while True:
        reason = blocked()
        if reason is None:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                reason = "the GPU is running another job"
            else:
                # Decide again while holding the lock: a hold, an earlier-ranked job or a reserved run may have arrived.
                reason = blocked()
                if reason is None:
                    break
                fcntl.flock(lock, fcntl.LOCK_UN)
        if reason != last:
            print(f"turnstile: waiting: {reason}", file=sys.stderr, flush=True)
            last = reason
        time.sleep(cfg.get("poll_seconds", 5))
    ticket.unlink(missing_ok=True)
    started = now()
    unit = f"turnstile-gpu-{t['id']}"
    cmd, scope = grouped(cfg, unit, ram, a.command)
    record = {**t, "unit": unit, "scope": scope, "started_at": started.isoformat(timespec="seconds"),
              "expected_finish": (started + dt.timedelta(minutes=a.max)).isoformat(timespec="seconds")}
    holder = STATE / "holder.json"
    write_json(holder, record)
    log("started", ticket=t["id"], owner=a.owner, cls=a.cls, project=a.project, ram_gib=ram)
    try:
        return run_child(cmd, a.max, record, unit=scope, keep=(lock.fileno(),), on_start=lambda: write_json(holder, record))
    finally:
        # The job and all its processes have ended (run_child returns or raises only then). Closing the descriptor lets
        # the lock go; an explicit unlock would also take it from a process that inherited the descriptor.
        with contextlib.suppress(OSError):
            holder.unlink(missing_ok=True)
        note("finished", ticket=t["id"], owner=a.owner, minutes=round((now() - started).total_seconds() / 60, 1),
             exit=record.get("exit"), peak_gib=record.get("peak_gib"))
        lock.close()


def cmd_cap(a) -> int:
    cfg = config()
    ram, why = ram_cap(a, cfg, "job_ram_gib")
    if ram is None:
        print(f"turnstile: refused: {why}", file=sys.stderr)
        return 2
    if refuse_without_group(cfg):
        return 2
    jid = f"{time.time_ns()}-{os.getpid()}"

    def leave(signum, _frame):
        note("left_queue", ticket=jid, owner=a.owner, signal=signum)
        sys.exit(130)

    signal.signal(signal.SIGTERM, leave)
    signal.signal(signal.SIGINT, leave)
    wait_for_ram(cfg, ram)
    (STATE / "capped").mkdir(parents=True, exist_ok=True)
    info = STATE / "capped" / f"{jid}.json"
    started = now()
    unit = f"turnstile-cap-{jid}"
    cmd, scope = grouped(cfg, unit, ram, a.command)
    record = {"id": jid, "owner": a.owner, "pid": os.getpid(), "pid_started": proc_start(os.getpid()), "ram_gib": ram,
              "max_minutes": a.max, "unit": unit, "scope": scope, "started_at": started.isoformat(timespec="seconds"),
              "command": " ".join(a.command)[:200]}
    write_json(info, record)
    log("cap_started", id=jid, owner=a.owner, ram_gib=ram, max_minutes=a.max)
    try:
        return run_child(cmd, a.max, record, unit=scope, on_start=lambda: write_json(info, record))
    finally:
        with contextlib.suppress(OSError):
            info.unlink(missing_ok=True)
        note("cap_finished", id=jid, owner=a.owner, minutes=round((now() - started).total_seconds() / 60, 1),
             exit=record.get("exit"), peak_gib=record.get("peak_gib"))


def take_slot(name: str, unreadable: list | None = None):
    """The slot's lock file, open and locked, or None when the slot is taken. A slot is taken while its lock is held,
    while its record names a job that still lives (a short run's launcher can die after moving the run to a normal slot,
    and the run's processes did not inherit that later lock), and while its record cannot be read, which adds the slot's
    name to unreadable."""
    f = open(STATE / "slots" / f"{name}.lock", "a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    state, held = record_state(STATE / "slots" / f"{name}.json")
    if state == "unreadable" or (state == "ok" and job_alive(held)):
        f.close()  # no other descriptor shares this new one, so closing it lets the lock go
        if state == "unreadable" and unreadable is not None:
            unreadable.append(name)
        return None
    return f


def cmd_worker(a) -> int:
    cfg = config()
    if a.kind == "short" and a.max > cfg.get("short_max_minutes", 15):
        print(f"turnstile: refused: a short run is declared for at most {cfg.get('short_max_minutes', 15)} minutes", file=sys.stderr)
        return 2
    ram, why = ram_cap(a, cfg, "worker_ram_gib")
    if ram is None:
        print(f"turnstile: refused: {why}", file=sys.stderr)
        return 2
    if refuse_without_group(cfg):
        return 2
    n = cfg.get("normal_slots", 6) if a.kind == "normal" else cfg.get("short_slots", 2)
    (STATE / "slots").mkdir(parents=True, exist_ok=True)

    def leave(signum, _frame):
        note("left_queue", kind=a.kind, owner=a.owner, signal=signum)
        sys.exit(130)

    signal.signal(signal.SIGTERM, leave)
    signal.signal(signal.SIGINT, leave)
    last = None
    while True:
        reason = ram_short(cfg, ram)
        unreadable = []
        if reason is None:
            for i in range(n):
                f = take_slot(f"{a.kind}-{i}", unreadable)
                if f is None:
                    continue
                info = STATE / "slots" / f"{a.kind}-{i}.json"
                started = now()
                unit = f"turnstile-{a.kind}-{i}-{time.time_ns()}"
                cmd, scope = grouped(cfg, unit, ram, a.command)
                record = {"id": f"{a.kind}-{i}", "owner": a.owner, "pid": os.getpid(), "pid_started": proc_start(os.getpid()),
                          "ram_gib": ram, "max_minutes": a.max, "unit": unit, "scope": scope,
                          "started_at": started.isoformat(timespec="seconds"), "command": " ".join(a.command)[:200]}
                write_json(info, record)
                log("slot_taken", slot=record["id"], owner=a.owner)
                extra = []  # a normal slot taken at a short run's deadline, with its record

                def deadline(signal_job_fn, finished):
                    """A short run still going at its deadline also takes a free normal slot (its short slot stays taken
                    until it ends, so it is never under-counted), or is stopped: when no normal slot is free, or when the
                    normal slot's record cannot be written (without it the slot would not stay taken if Turnstile died). It
                    is asked to end (SIGTERM), then ended (SIGKILL) if it is still going stop_grace_seconds later; each
                    signal goes before its log line, and a failed log or terminal write stops neither. Either way its
                    slots stay taken until all its processes have ended."""
                    try:
                        for j in range(cfg.get("normal_slots", 6)):
                            g = take_slot(f"normal-{j}")
                            if g is None:
                                continue
                            moved = STATE / "slots" / f"normal-{j}.json"
                            try:
                                write_json(moved, {**record, "id": f"normal-{j}", "moved_from": record["id"]})
                            except OSError:
                                g.close()
                                with contextlib.suppress(OSError):
                                    moved.unlink(missing_ok=True)
                                note("short_run_not_moved", slot=f"normal-{j}", owner=a.owner, short_slot=record["id"])
                                break
                            extra.append((g, moved))
                            note("short_run_moved", slot=f"normal-{j}", owner=a.owner, short_slot=record["id"])
                            say(f"the short run is past its {a.max:g} min deadline; it now also holds normal-{j}")
                            return
                    except OSError as error:
                        note("short_run_slot_error", owner=a.owner, short_slot=record["id"], error=type(error).__name__)
                    with contextlib.suppress(OSError, subprocess.SubprocessError):
                        signal_job_fn(signal.SIGTERM)
                    note("short_run_stopped", owner=a.owner, short_slot=record["id"])
                    say("the short run is past its deadline and has no normal slot; it was asked to stop")
                    grace = cfg.get("stop_grace_seconds", 30)
                    if not finished.wait(grace):
                        signal_job_fn(signal.SIGKILL)
                        note("short_run_killed", owner=a.owner, short_slot=record["id"], grace_seconds=grace)
                        say(f"the short run was still going {grace:g} s after it was asked to stop; it was ended")

                try:
                    return run_child(cmd, a.max, record, unit=scope, keep=(f.fileno(),), on_start=lambda: write_json(info, record),
                                     on_overrun=deadline if a.kind == "short" else None)
                finally:
                    # All the job's processes have ended; closing the descriptors lets the slots go.
                    for g, gi in extra:
                        with contextlib.suppress(OSError):
                            gi.unlink(missing_ok=True)
                        g.close()
                    with contextlib.suppress(OSError):
                        info.unlink(missing_ok=True)
                    f.close()
                    note("slot_returned", slot=record["id"], owner=a.owner, minutes=round((now() - started).total_seconds() / 60, 1),
                         exit=record.get("exit"), peak_gib=record.get("peak_gib"))
            reason = f"all {n} {a.kind} slots in use"
            if unreadable:
                reason += (f"; the record of {', '.join(unreadable)} cannot be read, so that slot stays taken until "
                           f"someone removes it from {STATE / 'slots'}")
        if reason != last:
            print(f"turnstile: waiting: {reason}", file=sys.stderr, flush=True)
            last = reason
        time.sleep(cfg.get("poll_seconds", 5))


def ram_note(record: dict) -> str:
    used = ram_now_gib(record.get("child_pid"))
    cap = record.get("ram_gib")
    gone = record.get("pid") and not alive(record["pid"])
    return (f"; RAM {used:.1f} of {cap:g} GiB" if used is not None and cap else f"; RAM cap {cap:g} GiB" if cap else "") + \
        ("; Turnstile itself has gone, the job still runs" if gone else "")


def cmd_status(_a) -> int:
    """Show the state. Status only reads: it changes and removes nothing. A holder record it reads as stale may already
    have been replaced by a new job's record, which is then all that keeps the GPU taken once that job's launcher and
    main process are gone; so a stale record is only left out of the report, and the next GPU job replaces it."""
    cfg = config()
    hold = read_json(STATE / "hold.json")
    held, holder = record_state(STATE / "holder.json")
    if holder and not job_alive(holder):
        holder = None
    print("Owner's hold: " + ((f"on since {hold['since']}" + (f" ({hold['note']})" if hold.get("note") else "")
                            + ("; not ready: the GPU holder record cannot be read" if held == "unreadable"
                               else "; ready, no Turnstile job is running" if not holder
                               else f"; not ready yet, {holder['owner']} is still running"))
                           if hold else "off"))
    if held == "unreadable":
        print(f"GPU: the holder record cannot be read ({STATE / 'holder.json'}); no GPU job starts until it is removed")
    elif holder:
        print(f"GPU: running {holder['owner']} ({holder['class']}{', ' + holder['project'] if holder.get('project') else ''}), "
              f"started {holder['started_at']}, expected to finish by {holder['expected_finish']}{ram_note(holder)}")
    else:
        print("GPU: no Turnstile job running")
    queue = tickets(clean=False) if (STATE / "queue").exists() else []
    print(f"Waiting: {len(queue)}" + "".join(f"\n  {i + 1}. {t['owner']} ({t['class']}{', ' + t['project'] if t.get('project') else ''}), "
                                              f"{t['mem_mib']} MiB, {t['max_minutes']} min, queued {t['enqueued_at'][:19]}" for i, t in enumerate(queue)))
    w = window_block(now(), cfg.get("windows", []), 0, "other")
    nxt = window_block(now(), cfg.get("windows", []), 24 * 60, "other")
    print("Reserved window: " + (f"open now, {w['name']}, until {w['end']}" if w else f"next {nxt['name']}, {nxt['start']} to {nxt['end']}" if nxt else "none"))
    print("Reserved runs: " + (reserved_running(cfg.get("windows", [])) or "none running"))
    mem = gpu_memory()
    if mem["free"] is None:
        print("GPU memory: cannot be read")
    else:
        mine = {holder.get("child_pid"), holder.get("pid")} if holder else set()
        outside = [p for p in mem["procs"] if p["pid"] not in mine]
        print(f"GPU memory: {mem['used']} of {mem['total']} MiB used, {mem['free']} free; outside Turnstile: "
              + (", ".join(f"{p['name']} {p['mib']} MiB" for p in outside) or "none"))
    avail = available_gib()
    print("RAM: " + (f"{avail:.1f} GiB available, {cfg.get('ram_reserve_gib', 12):g} GiB kept in reserve" if avail is not None else "cannot be read"))
    capped = []
    for p in (sorted((STATE / "capped").glob("*.json")) if (STATE / "capped").exists() else []):
        r = read_json(p)
        if r and job_alive(r):
            capped.append(r)
    print(f"Capped jobs without GPU: {len(capped)}" + "".join(f"\n  {r['owner']} since {r['started_at']}{ram_note(r)}" for r in capped))
    for kind, n in (("normal", cfg.get("normal_slots", 6)), ("short", cfg.get("short_slots", 2))):
        paths = sorted((STATE / "slots").glob(f"{kind}-*.json")) if (STATE / "slots").exists() else []
        states = [(p, *record_state(p)) for p in paths]
        lines = [f"\n  {h['id']}: {h['owner']} since {h['started_at']}" for _p, s, h in states if s == "ok" and job_alive(h)]
        lines += [f"\n  {p.stem}: record cannot be read ({p}); the slot stays taken until it is removed"
                  for p, s, _h in states if s == "unreadable"]
        print(f"Worker slots ({kind}): {len(lines)} of {n} in use" + "".join(lines))
    return 0


def cmd_hold(a) -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    write_json(STATE / "hold.json", {"since": now().isoformat(timespec="seconds"), "note": a.note})
    log("hold", note=a.note)
    held, holder = record_state(STATE / "holder.json")
    if held == "unreadable":
        after = f" The GPU holder record cannot be read ({STATE / 'holder.json'}); no GPU job starts until it is removed."
    elif held == "ok" and job_alive(holder):
        after = (f" Still running: {holder.get('owner', 'another')}, expected to finish by "
                 f"{holder.get('expected_finish', 'an unknown time')}.")
    else:
        after = " Nothing is running."
    print("Held. No new GPU job will start." + after)
    return 0


def cmd_release(_a) -> int:
    (STATE / "hold.json").unlink(missing_ok=True)
    log("release")
    print("Released. Queued GPU jobs may start.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="turnstile.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    h = sub.add_parser("hold")
    h.add_argument("--note", default="")
    sub.add_parser("release")
    r = sub.add_parser("run")
    r.add_argument("--class", dest="cls", choices=list(RANK), required=True)
    r.add_argument("--project")
    r.add_argument("--task", help="research class: the packet or task that puts the job in the project's approved scope")
    r.add_argument("--mem", type=int, required=True, help="GPU memory the job needs, MiB")
    c = sub.add_parser("cap")
    w = sub.add_parser("worker")
    w.add_argument("--kind", choices=["normal", "short"], required=True)
    for p in (r, c, w):
        p.add_argument("--ram", type=float, help="the job's RAM cap, GiB (default from the settings)")
        p.add_argument("--max", type=float, required=True, help="declared maximum run time, minutes")
        p.add_argument("--owner", required=True)
        p.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.cmd in ("run", "cap", "worker"):
        if a.command[:1] == ["--"]:
            a.command = a.command[1:]
        if not a.command:
            ap.error("give the command after --")
        if a.max <= 0 or (a.cmd == "run" and a.mem < 0):
            ap.error("--max must be positive and --mem not negative")
    return {"status": cmd_status, "hold": cmd_hold, "release": cmd_release, "run": cmd_run, "cap": cmd_cap, "worker": cmd_worker}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
