import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import cpu_threads as cpu


class PhysicalThreadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sys = self.root / 'sys'
        self.info = self.root / 'cpuinfo'; self.info.write_text('')
        self.cgroup = self.root / 'cgroup'; self.cgroup.write_text('0::/\n')
        self.mountinfo = self.root / 'mountinfo'
        self.mount = self.root / 'cgroup mount'; self.mount.mkdir()
        self.write_mount()
        (self.mount / 'cpu.max').write_text('max 100000\n')
        for name, value in [('SYS_CPU', self.sys), ('CPUINFO', self.info),
                            ('CGROUP', self.cgroup), ('MOUNTINFO', self.mountinfo)]:
            mock = patch.object(cpu, name, value); mock.start(); self.addCleanup(mock.stop)
        mock = patch.object(cpu.os, 'sched_getaffinity', return_value={0})
        self.affinity = mock.start(); self.addCleanup(mock.stop)
        self.topology([{0}])

    def write_mount(self, root='/', version=2):
        kind = 'cgroup2 cgroup rw' if version == 2 else 'cgroup cgroup rw,cpu,cpuacct'
        escaped = str(self.mount).replace(' ', r'\040')
        self.mountinfo.write_text(f'10 9 0:1 {root} {escaped} rw,nosuid,nodev - {kind}\n')

    def topology(self, groups, legacy=False):
        allowed = set()
        for group in groups:
            allowed.update(group)
            for logical in group:
                directory = self.sys / f'cpu{logical}' / 'topology'
                directory.mkdir(parents=True, exist_ok=True)
                (directory / 'core_cpus_list').unlink(missing_ok=True)
                filename = 'thread_siblings_list' if legacy else 'core_cpus_list'
                (directory / filename).write_text(','.join(str(n) for n in sorted(group)))
                # IDs deliberately repeat: sibling sets, not core_id alone, matter.
                (directory / 'core_id').write_text('0')
                (directory / 'physical_package_id').write_text('0')
        self.affinity.return_value = allowed

    def test_smt_six_real_cores_are_six_not_twelve_threads(self):
        self.topology([{n, n + 6} for n in range(6)])
        self.assertEqual(cpu.physical_thread_count(), 6)

    def test_four_way_smt_is_not_assumed_to_be_two_way(self):
        self.topology([{0, 2, 4, 6}, {1, 3, 5, 7}])
        self.assertEqual(cpu.physical_thread_count(), 2)

    def test_no_smt_and_repeated_core_ids_keep_every_core(self):
        self.topology([{n} for n in range(8)])
        self.assertEqual(cpu.physical_thread_count(), 8)

    def test_affinity_counts_only_available_cores_including_secondary_siblings(self):
        self.topology([{0, 4}, {1, 5}, {2, 6}, {3, 7}])
        for allowed, expected in [({4, 5}, 2), ({0, 4}, 1), ({0, 4, 5}, 2), ({7}, 1)]:
            self.affinity.return_value = allowed
            with self.subTest(allowed=allowed):
                self.assertEqual(cpu.physical_thread_count(), expected)

    def test_legacy_thread_siblings_list_is_supported(self):
        self.topology([{0, 2}, {1, 3}], legacy=True)
        self.assertEqual(cpu.physical_thread_count(), 2)

    def test_cpu_list_ranges_and_bad_lists(self):
        self.assertEqual(cpu.cpu_list('0-2,8,10-11\n'), {0, 1, 2, 8, 10, 11})
        for value in ['', '2-1', '-1', 'a', '1,', '1-2-3', '0-999999999']:
            with self.subTest(value=value), self.assertRaises(ValueError): cpu.cpu_list(value)

    def test_proc_fallback_counts_socket_core_pairs_and_respects_affinity(self):
        self.affinity.return_value = {0, 1, 2, 3, 4, 5}
        self.info.write_text('\n\n'.join(
            f'processor : {n}\nphysical id : {package}\ncore id : {core}'
            for n, package, core in [(0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 1, 0), (4, 1, 0), (5, 1, 1)]))
        self.assertEqual(cpu.physical_thread_count(), 4)
        self.affinity.return_value = {1, 4}
        self.assertEqual(cpu.physical_thread_count(), 2)

    def test_unknown_topology_does_not_guess_from_logical_count(self):
        self.affinity.return_value = {99}
        self.info.write_text('processor : 99\n')
        with patch.object(cpu.os, 'cpu_count', return_value=128) as logical:
            with self.assertRaisesRegex(ValueError, 'LLAMA_THREADS explicitly'):
                cpu.physical_thread_count()
            logical.assert_not_called()

    def test_empty_or_unavailable_affinity_fails_clearly(self):
        self.affinity.return_value = set()
        with self.assertRaisesRegex(ValueError, 'LLAMA_THREADS explicitly'): cpu.physical_thread_count()
        self.affinity.side_effect = OSError('unavailable')
        with self.assertRaisesRegex(ValueError, 'LLAMA_THREADS explicitly'): cpu.physical_thread_count()

    def test_v2_quota_caps_physical_count_and_rounds_down(self):
        self.topology([{n, n + 6} for n in range(6)])
        for quota, expected in [('1200000 100000', 6), ('250000 100000', 2), ('50000 100000', 1), ('max 100000', 6)]:
            (self.mount / 'cpu.max').write_text(quota)
            with self.subTest(quota=quota): self.assertEqual(cpu.physical_thread_count(), expected)

    def test_v2_visible_ancestor_quota_is_honored(self):
        self.topology([{n} for n in range(8)])
        child = self.mount / 'group' / 'worker'; child.mkdir(parents=True)
        (child / 'cpu.max').write_text('600000 100000')
        (child.parent / 'cpu.max').write_text('300000 100000')
        self.cgroup.write_text('0::/group/worker\n')
        self.assertEqual(cpu.physical_thread_count(), 3)
        (self.mount / 'cpu.max').write_text('150000 100000')
        self.assertEqual(cpu.physical_thread_count(), 1)

    def test_nonroot_mount_and_namespaced_membership(self):
        self.topology([{n} for n in range(8)])
        self.write_mount('/docker/container')
        child = self.mount / 'worker'; child.mkdir()
        (child / 'cpu.max').write_text('300000 100000')
        for member in ['/docker/container/worker', '/worker']:
            self.cgroup.write_text(f'0::{member}\n')
            with self.subTest(member=member): self.assertEqual(cpu.physical_thread_count(), 3)

    def test_v1_cpu_quota_and_visible_parent(self):
        self.topology([{n} for n in range(8)])
        self.write_mount('/container', version=1)
        self.cgroup.write_text('4:cpuacct,cpu:/container/worker\n5:cpuset:/other\n')
        child = self.mount / 'worker'; child.mkdir()
        (child / 'cpu.cfs_quota_us').write_text('-1')
        (child / 'cpu.cfs_period_us').write_text('100000')
        (self.mount / 'cpu.cfs_quota_us').write_text('200000')
        (self.mount / 'cpu.cfs_period_us').write_text('100000')
        self.assertEqual(cpu.physical_thread_count(), 2)

    def test_no_visible_quota_keeps_physical_count(self):
        self.topology([{0, 2}, {1, 3}])
        (self.mount / 'cpu.max').unlink()
        self.assertEqual(cpu.physical_thread_count(), 2)
        self.cgroup.unlink()
        self.assertEqual(cpu.physical_thread_count(), 2)

    def test_malformed_quota_requires_manual_selection(self):
        for quota in ['nope', '0 100000', '100000 0', '-1 100000', 'NaN 100000']:
            (self.mount / 'cpu.max').write_text(quota)
            with self.subTest(quota=quota), self.assertRaisesRegex(ValueError, 'LLAMA_THREADS explicitly'):
                cpu.physical_thread_count()

    def test_auto_count_keeps_existing_512_thread_limit(self):
        with patch.object(cpu, 'sysfs_core_count', return_value=1024):
            self.assertEqual(cpu.physical_thread_count(), 512)


if __name__ == '__main__':
    unittest.main()
