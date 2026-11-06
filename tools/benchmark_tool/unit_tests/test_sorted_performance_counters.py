# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import unittest
from contextlib import redirect_stdout
from datetime import timedelta
from types import SimpleNamespace

from openvino import ProfilingInfo
from openvino.tools.benchmark.utils.utils import print_perf_counters_sort


class TestSortedPerformanceCounters(unittest.TestCase):
    @staticmethod
    def counter(name, real_time, cpu_time, status=ProfilingInfo.Status.EXECUTED):
        return SimpleNamespace(node_name=name, node_type='Test', exec_type='test',
                               status=status, real_time=real_time, cpu_time=cpu_time)

    def render(self, counters, sort_flag='sort'):
        output = io.StringIO()
        with redirect_stdout(output):
            result = print_perf_counters_sort([counters], sort_flag=sort_flag)
        return result, output.getvalue()

    def test_zero_time_counters(self):
        counter = self.counter('optimized_out', timedelta(), timedelta(), ProfilingInfo.Status.NOT_RUN)
        for sort_flag in ('sort', 'no_sort', 'simple_sort'):
            with self.subTest(sort_flag=sort_flag):
                result, output = self.render([counter], sort_flag)
                self.assertIn('Total time:       0.000 milliseconds', output)
                if sort_flag == 'simple_sort':
                    self.assertEqual(len(result), 0)
                else:
                    self.assertEqual(result[0][5], 0)

    def test_whole_seconds_preserved_in_counters_and_totals(self):
        counters = [self.counter('short', timedelta(milliseconds=750), timedelta(milliseconds=250)),
                    self.counter('long', timedelta(milliseconds=1500), timedelta(seconds=1))]
        result, output = self.render(counters)
        self.assertEqual([row[0] for row in result], ['long', 'short'])
        self.assertEqual([row[3] for row in result], [1500000, 750000])
        self.assertEqual([row[4] for row in result], [1000000, 250000])
        self.assertEqual([row[5] for row in result], [0.6667, 0.3333])
        self.assertIn('Total time:       2250.000 milliseconds', output)
        self.assertIn('Total CPU time:   1250.000 milliseconds', output)

    def test_subsecond_counters_with_a_whole_second_total(self):
        counters = [self.counter('first', timedelta(milliseconds=500), timedelta(milliseconds=100)),
                    self.counter('second', timedelta(milliseconds=500), timedelta(milliseconds=100))]
        result, output = self.render(counters)
        self.assertEqual([row[5] for row in result], [0.5, 0.5])
        self.assertIn('Total time:       1000.000 milliseconds', output)


if __name__ == '__main__':
    unittest.main()
