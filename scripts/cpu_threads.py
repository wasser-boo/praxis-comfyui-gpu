#!/usr/bin/env python3
"""Count physical cores available to this Linux process, not logical/SMT CPUs.

Affinity includes taskset/cpuset restrictions. Visible cgroup CPU quotas further
cap the thread budget. No guessing from os.cpu_count() or dividing by two.
"""
import os
from pathlib import Path, PurePosixPath
import re

SYS_CPU = Path('/sys/devices/system/cpu')
CPUINFO = Path('/proc/cpuinfo')
CGROUP = Path('/proc/self/cgroup')
MOUNTINFO = Path('/proc/self/mountinfo')
MAX_THREADS = 512


def cpu_list(text):
    cpus = set()
    for part in text.strip().split(','):
        if not re.fullmatch(r'[0-9]+(?:-[0-9]+)?', part):
            raise ValueError('Invalid CPU topology list')
        bounds = [int(value) for value in part.split('-')]
        first, last = bounds[0], bounds[-1]
        if not 0 <= first <= last < 1_048_576:
            raise ValueError('Invalid CPU topology range')
        cpus.update(range(first, last + 1))
    return frozenset(cpus)


def sysfs_core_count(allowed):
    groups = set()
    for cpu in allowed:
        directory = SYS_CPU / f'cpu{cpu}' / 'topology'
        # Sibling sets identify cores globally, including multiple sockets/dies.
        # Bare core_id can repeat even within a package on some ARM systems.
        for filename in ('core_cpus_list', 'thread_siblings_list'):
            try:
                siblings = cpu_list((directory / filename).read_text())
                if cpu not in siblings:
                    raise ValueError('CPU missing from its own sibling list')
                break
            except (OSError, ValueError):
                continue
        else:
            raise ValueError('Incomplete sysfs CPU topology')
        groups.add(siblings)
    # Reject inconsistent/overlapping groups instead of double-counting a core.
    seen = set()
    for group in groups:
        if seen.intersection(group):
            raise ValueError('Inconsistent CPU sibling topology')
        seen.update(group)
    return len(groups)


def cpuinfo_core_count(allowed):
    cores = {}
    for block in CPUINFO.read_text().split('\n\n'):
        fields = dict(line.split(':', 1) for line in block.splitlines() if ':' in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if 'processor' not in fields:
            continue
        cpu = int(fields['processor'])
        if cpu not in allowed:
            continue
        package, core = int(fields['physical id']), int(fields['core id'])
        if min(package, core) < 0:
            raise ValueError('Unknown CPU topology IDs')
        cores[cpu] = (package, core)
    if set(cores) != set(allowed):
        raise ValueError('Incomplete /proc/cpuinfo topology')
    return len(set(cores.values()))


def unescape_mount(value):
    return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), value)


def quota_directories():
    """Yield (version, directory) for the process and each visible ancestor."""
    try:
        groups = [line.split(':', 2) for line in CGROUP.read_text().splitlines()]
        mounts = MOUNTINFO.read_text().splitlines()
    except OSError:
        return
    membership = {}
    for _, controllers, path in groups:
        if not controllers:
            membership['cgroup2'] = path
        elif 'cpu' in controllers.split(','):
            membership['cgroup'] = path
    for line in mounts:
        left, right = line.split(' - ', 1)
        fields, details = left.split(), right.split()
        kind = details[0]
        if kind not in membership or (kind == 'cgroup' and 'cpu' not in details[2].split(',')):
            continue
        root = PurePosixPath(unescape_mount(fields[3]))
        mount = Path(unescape_mount(fields[4]))
        member = PurePosixPath(membership[kind])
        if not member.is_absolute() or '..' in member.parts:
            raise ValueError('Invalid process cgroup path')
        try:
            relative = member.relative_to(root)
        except ValueError:
            # Cgroup namespaces may report membership relative to the mount root.
            relative = member.relative_to('/')
        current = mount / str(relative)
        while True:
            yield kind, current
            if current == mount:
                break
            current = current.parent


def quota_thread_limit():
    limits = []
    for kind, directory in quota_directories():
        try:
            if kind == 'cgroup2':
                quota, period = (directory / 'cpu.max').read_text().split()
                if quota == 'max':
                    continue
            else:
                quota = (directory / 'cpu.cfs_quota_us').read_text().strip()
                period = (directory / 'cpu.cfs_period_us').read_text().strip()
                if quota == '-1':
                    continue
        except FileNotFoundError:
            continue
        quota, period = int(quota), int(period)
        if quota <= 0 or period <= 0:
            raise ValueError('Invalid cgroup CPU quota')
        # A 2.5-CPU quota gets 2 workers; a sub-core quota still needs 1.
        limits.append(max(1, quota // period))
    return min(limits) if limits else None


def physical_thread_count():
    try:
        allowed = os.sched_getaffinity(0)
        if not allowed:
            raise ValueError('Empty CPU affinity')
        try:
            count = sysfs_core_count(allowed)
        except (OSError, ValueError):
            count = cpuinfo_core_count(allowed)
        quota = quota_thread_limit()
        return min(count, quota if quota is not None else count, MAX_THREADS)
    except (AttributeError, OSError, ValueError, KeyError, IndexError) as error:
        raise ValueError('Cannot detect available physical CPU cores/quota; '
                         'set LLAMA_THREADS explicitly after inspecting the host') from error


if __name__ == '__main__':
    print(physical_thread_count())
