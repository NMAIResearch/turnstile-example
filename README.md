# Turnstile (worked example)

`turnstile.py` is a single-file admission queue for several AI agent sessions that share one Linux workstation's GPU and
memory: one GPU job at a time, a fixed number of worker slots, and, with memory groups on (the default), a memory cap
(systemd's `MemoryMax`) set on every job it starts. It is shared as a worked example of one workstation's rules, not as a general scheduler: read it, then adapt the rules or use one of the
tools under "Alternatives".

## Behaviour

- **One GPU job at a time** (`run`): a new GPU job waits while any process of the running one lives, also after
  Turnstile itself has been killed (within the limits below). The owner's hold stops new starts. Then scheduled jobs in a reserved window, then jobs
  for a project listed in `registry.json` (with `--project` and `--task`), then everything else, first come, first served
  within each class. A job starts only when its declared GPU memory (from `nvidia-smi`, plus a margin) is free and its RAM
  cap fits beside a reserve. A running job is never pre-empted; a job past its declared maximum is reported, not stopped.
  The grant is released only once the job's processes have ended, also when something fails after the job has started
  and when Turnstile itself is killed: Turnstile follows them through the job's memory group, or without one through
  the processes it adopts and the job's own session, and a grant whose record cannot be read stays taken. The tests
  check the memory-group path only where systemd user scopes work (see Tests).
- **Reserved windows.** A job that would run into a window waits. While one of the window's systemd user units is still
  running, or its state cannot be read (any answer other than a running state, `inactive` or `failed`), only scheduled
  jobs start, so a late or long scheduled run keeps the GPU.
- **Memory groups.** Every job runs under `systemd-run --user --scope` with `MemoryMax`, `MemorySwapMax` and
  `OOMPolicy=kill`. A test reads the cap back from the job's memory group where user scopes work. systemd documents that
  with `OOMPolicy=kill` the rest of a unit is stopped when the kernel's out-of-memory killer stops one of its
  processes; the live test of that is opt-in (see Tests). A memory group whose state cannot be read counts as still running. `cap` gives a heavy job that
  needs no GPU the same memory group.
- **Worker slots** (`worker --kind normal|short`). Six normal and two short slots by default, for agent runs started from
  any terminal; a worker also waits until its RAM cap fits beside the reserve. A short run is declared for at most 15
  minutes. At its deadline it also takes a free normal slot or, when none is free, is asked to stop (SIGTERM) and ended
  (SIGKILL) `stop_grace_seconds` later; a log or terminal write that fails stops neither signal. A slot whose record
  names a job that still runs stays taken, so a short run moved to a normal slot keeps it even if its launcher dies. If
  that slot's record cannot be written, the short run is stopped instead; a slot record that cannot be read keeps its
  slot taken until someone removes it, and `status` names it.
- **Status** (`status`) shows the hold, the running job, the queue, windows, GPU memory and GPU users outside Turnstile,
  available RAM, capped jobs and slots. It only reads: it changes and removes nothing, so running it never affects a
  grant.

## Requirements

- Linux with a systemd user manager, and systemd 253 or newer for `OOMPolicy=` on scopes. Or set `memory_groups` to
  `false`: jobs then have no memory caps. Turnstile adopts the processes a job leaves behind (Linux
  `PR_SET_CHILD_SUBREAPER`) and starts each job in a session of its own, so it waits for all of them before it lets the
  GPU or a slot go, and if Turnstile itself is killed the grant stays taken while any process of that session lives. A
  job started this way has no controlling terminal.
- Python 3, standard library only.
- `nvidia-smi` for GPU jobs; Turnstile reads the first GPU only. `notify-send` is used for overrun notices when present.

Tested on Linux with Python 3.14.8 and systemd 259.

## Use

```text
python3 turnstile.py status
python3 turnstile.py run --class other --mem 8000 --max 60 --owner alice -- python3 train.py
python3 turnstile.py run --class research --project example-project --task nightly-eval --mem 12000 --max 90 --owner bob -- python3 eval.py
python3 turnstile.py cap --ram 16 --max 30 --owner carol -- ./render.sh
python3 turnstile.py worker --kind normal --max 120 --owner agent-1 -- ./run-agent.sh
python3 turnstile.py hold --note "playing a game"
python3 turnstile.py release
```

`hold` and `release` are meant for the workstation's owner alone; Turnstile does not check who runs them.

Exit status: the job's own (128 plus the signal number when a signal ended it, so 137 for SIGKILL), 2 for a refused
request, 130 when interrupted while waiting.

## Settings (`turnstile.json`)

Slot counts, the short-run limit, the time a short run has to stop at its deadline (`stop_grace_seconds`, 30 by default),
the polling interval, the GPU memory margin, memory groups on or off, default RAM caps, the largest cap a job may ask for,
the RAM kept in reserve and the swap cap. The `windows` list starts empty. A window looks like this (local times; the
units are systemd user services whose running state also keeps the window shut):

```json
"windows": [{"name": "nightly job", "start": "05:55", "end": "06:45", "units": ["nightly-job.service"]}]
```

`TURNSTILE_CONFIG` points at another settings file; `TURNSTILE_STATE` moves the state from `~/.local/state/turnstile`.

## Registry (`registry.json`)

The research class needs `--project` naming a project in `registry.json` whose lane is not `support` and whose status is
not `closed`, and `--task` saying which task puts the job in that project's scope. Replace the two example projects with
your own, or point `TURNSTILE_REGISTRY` at another file.

## Limits

- Advisory: the GPU lock and the slots bind only processes started through Turnstile. `status` shows other GPU users, and
  their memory counts against free memory, but they are not scheduled.
- Not a security boundary: any process of the same user can ignore it.
- Strict priority: the classes have no ageing, so a steady stream of scheduled or research jobs can keep other jobs
  waiting.
- GPU jobs past their declared maximum keep running until someone decides; only short worker runs are stopped, at their
  deadline, when no normal slot is free or its record cannot be written.
- Turnstile follows a job through its memory group, or with memory groups off through its process tree and its session.
  A process the job has another service start (D-Bus activation, or `systemd-run` without `--scope`) is outside all of
  these. With memory groups on, one moved into a scope of its own (`systemd-run --scope`) leaves the job's memory group;
  with them off it stays in the process tree and the session. One that starts a session of its own (`setsid`) leaves
  the session once Turnstile has gone. Turnstile does not wait for processes outside what it follows. The tests do not
  exercise these ways of leaving what Turnstile follows.
- With memory groups off, if Turnstile cannot write a job's record after starting it and is then killed, the grant is
  held only through the lock descriptor the job inherited: by every process of the job that still has it open (the
  job's own process, and the processes it starts unless they close it). A process that has closed it does not hold the
  grant.
- The rules (an owner's hold, a research class, six plus two worker slots) are one workstation's policy. Adapt them.

## Alternatives

- **pueue**: a general queue for shell commands, with groups, per-group parallel limits and groups you can pause; a group
  allowed one parallel task runs its tasks one at a time.
- **GPU Task Spooler** (`justanhduc/task-spooler`): queues CPU and GPU jobs and hands out free GPUs on small servers.
- **worker-q**: a broker for GPU, RAM and CPU work shared by AI coding agents, with declared footprints, priorities and
  time-limited reservations; its notes describe use on Windows.
- **agent-throttle**: a machine-wide semaphore, and a Claude Code hook that blocks test runs started without an explicit
  worker count or above a ceiling.
- **gpu-swap**: one job at a time around a local Ollama server, which can unload its models and restores them afterwards.
- **Slurm**: a full workload manager, with job priorities and reservations.
- **`systemd-run --user --scope -p MemoryMax=...` alone**: if all you need is to start each job in a scope with systemd's
  `MemoryMax` setting; Turnstile starts each job that way when memory groups are on.

Turnstile's approach may suit a workstation where several agent sessions share one GPU and where one readable file is
preferred to a scheduler.

## Tests

```text
python3 -m unittest discover -s tests
```

74 tests, using a fake `nvidia-smi`, a fake `systemctl` for reserved units and temporary state. The memory-group tests
need `systemd-run --user --scope` and are skipped where it does not work. One test deliberately triggers the
out-of-memory killer on a capped job and runs only when `TURNSTILE_LIVE_OOM_TEST=1` is set.

## Review status

An OpenAI GPT-6 model reviewed this code independently in October 2026, and a GPT-6.1 model re-checked it three times.
The review and the first two re-checks found defects; the third found the remaining defects fixed and no new ones.
The reviewer's sandbox could not run systemd user scopes, and processes started by other services were not tested.
This README's wording was revised after the third re-check and has not been re-checked.

## AI assistance

Written with AI assistance (Anthropic Claude Opus 5.5 in Claude Code) under the author's direction.

## Licence

Code under the MIT licence, documentation under CC BY 4.0; see `LICENSE.md`.
