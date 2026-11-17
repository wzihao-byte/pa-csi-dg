"""Safety and classification checks; no real experiments or processes are changed."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

PATH = Path(__file__).with_name('monitor_revision_experiments_20260930.py')
spec = importlib.util.spec_from_file_location('health_monitor_under_test', PATH)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / 'coordinator'
        self.directory.mkdir()
        self.bundle = Path(self.temp.name) / 'study'
        self.run = self.bundle / 'runs/job'
        self.run.mkdir(parents=True)
        self.plan = {'studies': [{'bundle': str(self.bundle), 'job_count': 1}]}
        self.protocols = [{'jobs': [{'job_id': 'job', 'epochs': 200}]}]
        self.empty_scan = {'matching_processes': [], 'inaccessible_python_pids': []}

    def events(self, text, timestamp=1950):
        path = self.run / 'events.jsonl'
        path.write_text(text, encoding='utf-8')
        os.utime(path, (timestamp, timestamp))

    def processes(self, coordinator='alive', worker='alive', created=100):
        return {'coordinator': [{'state': coordinator, 'create_time': created}],
                'workers': [{'study_index': 0, 'identities': [{'state': worker, 'create_time': created}]}]}

    def observe(self, processes, verified=False, scan=None):
        checker = Mock(return_value=verified)
        with patch.object(monitor, 'validate', return_value=(self.plan, self.protocols)), \
             patch.object(monitor, 'identities', return_value=processes), \
             patch.object(monitor, 'scan_unregistered_writers', return_value=scan or self.empty_scan), \
             patch.object(monitor, 'load_completion_verifier', return_value=checker):
            result = monitor.snapshot(self.directory, now=2000)
        return result, checker

    def test_dead_worker_old_running_event_is_stopped(self):
        self.events('{"event":"epoch_committed","epoch":136}\n')
        result, checker = self.observe(self.processes('dead', 'dead'))
        self.assertEqual(result['health'], 'stopped')
        self.assertEqual(result['studies'][0]['active_or_failed_jobs'][0]['epoch'], 136)
        self.assertFalse(result['studies'][0]['worker_alive'])
        checker.assert_called_once_with(self.bundle)

    def test_pid_reuse_is_not_alive(self):
        path = self.directory / 'launcher.json'
        path.write_text(json.dumps({'pid': 42, 'create_time': 100}))
        process = Mock()
        process.create_time.return_value = 900
        with patch.object(monitor.psutil, 'Process', return_value=process):
            self.assertEqual(monitor.identity_state(path)['state'], 'pid_reused')
        process.is_running.assert_not_called()

    def test_recent_epoch_does_not_read_mutable_completion_state(self):
        self.events('{"event":"epoch_committed","epoch":137}\n')
        result, checker = self.observe(self.processes())
        self.assertEqual(result['health'], 'running')
        self.assertEqual(result['seconds_since_progress'], 50)
        checker.assert_not_called()

    def test_stale_live_worker_is_suspected_not_stopped(self):
        self.events('{"event":"epoch_committed","epoch":137}\n', timestamp=500)
        result, checker = self.observe(self.processes())
        self.assertEqual(result['health'], 'suspected_stall')
        self.assertTrue(result['needs_attention'])
        checker.assert_not_called()

    def test_partial_event_is_ignored_until_committed(self):
        self.events('{"event":"epoch_committed","epoch":137}\n{"event":"completed"')
        result, _ = self.observe(self.processes())
        self.assertEqual(result['source_completed'], 0)
        self.assertEqual(result['studies'][0]['active_or_failed_jobs'][0]['epoch'], 137)

    def test_clipped_first_line_is_ignored(self):
        path = self.run / 'events.jsonl'
        path.write_text('x' * 200 + '\n{"event":"completed"}\n')
        lines = monitor.log_tail(path, limit=40)['lines']
        self.assertEqual(lines, ['{"event":"completed"}'])

    def test_source_events_alone_do_not_verify_completion(self):
        self.events('{"event":"completed"}\n')
        result, _ = self.observe(self.processes('dead', 'dead'), verified=False)
        self.assertEqual(result['source_completed'], 1)
        self.assertEqual(result['health'], 'stopped')
        self.assertFalse(result['completion_verification']['verified'])

    def test_completed_requires_all_dead_and_frozen_verifier(self):
        self.events('{"event":"completed"}\n')
        result, checker = self.observe(self.processes('dead', 'pid_reused'), verified=True)
        self.assertEqual(result['health'], 'completed')
        checker.assert_called_once_with(self.bundle)

    def test_orphan_worker_is_actionable_without_repair(self):
        self.events('{"event":"epoch_committed","epoch":137}\n')
        result, checker = self.observe(self.processes('dead', 'alive'))
        self.assertEqual(result['health'], 'coordinator_lost')
        checker.assert_not_called()

    def test_unknown_or_unregistered_process_blocks_completion(self):
        result, checker = self.observe(self.processes('dead', 'dead'), scan={
            'matching_processes': [{'pid': 99}], 'inaccessible_python_pids': []})
        self.assertEqual(result['health'], 'unknown')
        checker.assert_not_called()
        result, checker = self.observe(self.processes('unknown', 'dead'))
        self.assertEqual(result['health'], 'unknown')
        checker.assert_not_called()

    def test_evaluation_stdout_progress_prevents_false_stall(self):
        self.events('{"event":"completed"}\n', timestamp=500)
        path = self.directory / '0.stdout.log'
        path.write_text('Target evaluation progressing\n')
        os.utime(path, (1990, 1990))
        result, checker = self.observe(self.processes())
        self.assertEqual(result['health'], 'running')
        self.assertEqual(result['studies'][0]['phase'], 'evaluation_or_final_verification')
        checker.assert_not_called()

    def test_retry_log_activity_is_not_training_progress(self):
        self.events('{"event":"epoch_committed","epoch":137}\n', timestamp=500)
        path = self.directory / '0_atomic_retry.jsonl'
        path.write_text('{"event":"retry"}\n')
        os.utime(path, (1990, 1990))
        result, _ = self.observe(self.processes())
        self.assertEqual(result['health'], 'suspected_stall')

    def test_same_identity_is_deduplicated_but_distinct_writers_alert(self):
        self.events('{"event":"epoch_committed","epoch":137}\n')
        processes = self.processes()
        processes['workers'][0]['identities'] *= 2
        result, _ = self.observe(processes)
        self.assertEqual(result['health'], 'running')
        processes['workers'][0]['identities'].append({'state': 'alive', 'pid': 99, 'create_time': 200})
        result, checker = self.observe(processes)
        self.assertEqual(result['health'], 'duplicate_writers')
        self.assertTrue(result['needs_attention'])
        checker.assert_not_called()

    def test_two_live_studies_alert_even_with_recent_progress(self):
        self.events('{"event":"epoch_committed","epoch":137}\n')
        processes = self.processes()
        processes['workers'].append({'study_index': 1, 'identities': [{'state': 'alive', 'pid': 101, 'create_time': 150}]})
        result, checker = self.observe(processes)
        self.assertEqual(result['health'], 'duplicate_writers')
        checker.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
