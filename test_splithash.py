#!/usr/bin/env python3
"""splithash 的单元测试（标准库 unittest）。"""

import io
import os
import subprocess
import sys
import tempfile
import unittest

import splithash as sh

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "samples")


def run_cli(*args, cwd=ROOT, env=None):
    return subprocess.run(
        [sys.executable, "splithash.py", *args],
        cwd=cwd, env=env, capture_output=True, text=True)


class RatioAndSaltTest(unittest.TestCase):
    def test_valid_ratio(self):
        self.assertEqual(sh.parse_ratio("80:10:10"), [80, 10, 10])
        self.assertEqual(sh.parse_ratio("1:1:98"), [1, 1, 98])

    def test_bad_ratio(self):
        for text in ("80:10", "80:10:10:0", "80:10:9", "0:50:50",
                     "a:10:10", "80:10:-10", "80:10:１０", ""):
            with self.assertRaises(sh.InputError, msg=text):
                sh.parse_ratio(text)

    def test_bad_salt(self):
        for text in ("", "a\x00b"):
            with self.assertRaises(sh.InputError, msg=repr(text)):
                sh.parse_salt(text)


class AssignmentTest(unittest.TestCase):
    def test_known_buckets(self):
        splitter = sh.Splitter(sh.parse_salt("s-01"), sh.parse_ratio("80:10:10"))
        expected = {"u-0001": "train", "u-0002": "valid", "u-0003": "train",
                    "u-0004": "test", "u-0005": "train"}
        for key, name in expected.items():
            idx = splitter.index_of(key.encode("utf-8"))
            self.assertEqual(sh.SPLIT_NAMES[idx], name, key)

    def test_assignment_is_pure_function(self):
        splitter = sh.Splitter(sh.parse_salt("s-x"), sh.parse_ratio("70:20:10"))
        before = [splitter.index_of(f"k-{i}".encode()) for i in range(500)]
        splitter2 = sh.Splitter(sh.parse_salt("s-x"), sh.parse_ratio("70:20:10"))
        after = [splitter2.index_of(f"k-{i}".encode()) for i in range(500)]
        self.assertEqual(before, after)

    def test_ratio_changes_boundaries(self):
        key = b"u-0001"
        wide = sh.Splitter(sh.parse_salt("s-01"), sh.parse_ratio("90:5:5"))
        narrow = sh.Splitter(sh.parse_salt("s-01"), sh.parse_ratio("60:20:20"))
        self.assertEqual(wide.index_of(key), 0)
        self.assertEqual(narrow.index_of(key), 1)
        self.assertEqual(sh.Splitter(sh.parse_salt("s-01"),
                                     sh.parse_ratio("1:1:98")).index_of(key), 2)


class InputValidationTest(unittest.TestCase):
    def write(self, data):
        fd, path = tempfile.mkstemp(suffix=".tsv")
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        self.addCleanup(os.unlink, path)
        return path

    def records(self, data):
        return list(sh.iter_records(self.write(data)))

    def test_valid(self):
        rows = self.records("键\t类别\n".encode("utf-8"))
        self.assertEqual(rows[0][2:], ("键", "类别"))

    def test_no_trim(self):
        rows = self.records(b"a \tb\n")
        self.assertEqual(rows[0][2], "a ")

    def test_bad_inputs(self):
        cases = {
            "BOM": b"\xef\xbb\xbfk\tc\n",
            "CRLF": b"k\tc\r\n",
            "missing LF": b"k\tc",
            "empty line": b"k\tc\n\n",
            "extra TAB": b"k\tc\textra\n",
            "no TAB": b"kc\n",
            "empty key": b"\tc\n",
            "empty label": b"k\t\n",
            "control char": b"k\x01\tc\n",
            "DEL": b"k\x7f\tc\n",
            "bad utf-8": b"k\xff\tc\n",
        }
        for name, data in cases.items():
            with self.assertRaises(sh.InputError, msg=name):
                self.records(data)


class SamplesTest(unittest.TestCase):
    def stats_of(self, data, salt, ratio):
        splitter = sh.Splitter(sh.parse_salt(salt), sh.parse_ratio(ratio))
        return sh.compute_stats(f"{SAMPLES}/data/{data}", splitter)

    def test_check_ok(self):
        result = run_cli("check", "samples")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok=11")

    def test_split_matches_expected_bytes(self):
        result = run_cli("split", "samples/data/01-balanced.tsv", "s-01", "80:10:10")
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(f"{SAMPLES}/expected/01-balanced.map.tsv", "rb") as stream:
            self.assertEqual(result.stdout.encode(), stream.read())

    def test_stats_matches_expected_bytes(self):
        result = run_cli("stats", "samples/data/04-dup-keys.tsv", "s-04-08", "80:10:10")
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(f"{SAMPLES}/expected/04-dup-keys.stats.txt", "rb") as stream:
            self.assertEqual(result.stdout.encode(), stream.read())

    def test_dup_keys_same_split(self):
        splitter = sh.Splitter(sh.parse_salt("s-04-08"), sh.parse_ratio("80:10:10"))
        seen = {}
        for _line, key_b, _key, _label in sh.iter_records(f"{SAMPLES}/data/04-dup-keys.tsv"):
            idx = splitter.index_of(key_b)
            if key_b in seen:
                self.assertEqual(seen[key_b], idx)
            seen[key_b] = idx
        stats = self.stats_of("04-dup-keys.tsv", "s-04-08", "80:10:10")
        self.assertEqual(stats.total, 120)
        self.assertEqual(stats.unique_keys, 36)

    def test_stability_append_and_drop(self):
        for name, a, b, salt, ratio, exp in (
                ("05", "05-base.tsv", "05-append.tsv", "s-05", "70:20:10",
                 "05-append.stability.txt"),
                ("06", "06-full.tsv", "06-dropped.tsv", "s-06-02", "75:15:10",
                 "06-drop.stability.txt")):
            splitter = sh.Splitter(sh.parse_salt(salt), sh.parse_ratio(ratio))
            counts = sh.stability_counts(f"{SAMPLES}/data/{a}",
                                         f"{SAMPLES}/data/{b}", splitter)
            with open(f"{SAMPLES}/expected/{exp}", encoding="utf-8") as stream:
                expected = sh.parse_stability_expected(stream.read())
            self.assertEqual(counts, expected, name)
            self.assertEqual(counts[3], 0, f"{name}: 已有键归属发生变化")

    def test_scale_tolerances(self):
        stats = self.stats_of("07-scale.tsv", "s-07", "80:10:10")
        targets = (80.0, 10.0, 10.0)
        for index, target in enumerate(targets):
            pct = stats.split_counts[index] * 100 / stats.total
            self.assertAlmostEqual(pct, target, delta=2.0,
                                   msg=f"{sh.SPLIT_NAMES[index]} 占比偏差超限")
        for label, counts in stats.categories.items():
            cat_total = counts[0]
            if cat_total < 200:
                continue
            global_pct = cat_total * 100 / stats.total
            for index in range(3):
                split_n = stats.split_counts[index]
                if split_n == 0:
                    continue
                in_split = counts[1 + index] * 100 / split_n
                self.assertAlmostEqual(in_split, global_pct, delta=3.0,
                                       msg=f"{label} 在 {sh.SPLIT_NAMES[index]} 偏差超限")

    def test_long_tail_categories_present(self):
        stats = self.stats_of("03-long-tail.tsv", "s-03", "90:5:5")
        self.assertEqual(len(stats.categories), 92)
        singletons = [c for c in stats.categories.values() if c[0] == 1]
        self.assertEqual(len(singletons), 90)


class DeterminismTest(unittest.TestCase):
    def test_same_output_across_hash_seeds(self):
        outputs = []
        for seed in ("0", "1", "2"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            result = run_cli("stats", "samples/data/02-skewed.tsv", "s-02-01",
                             "80:10:10", env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.append(result.stdout)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])

    def test_row_order_does_not_change_assignment(self):
        splitter = sh.Splitter(sh.parse_salt("s-05"), sh.parse_ratio("70:20:10"))
        map_base = sh.key_split_map(f"{SAMPLES}/data/05-base.tsv", splitter)
        map_append = sh.key_split_map(f"{SAMPLES}/data/05-append.tsv", splitter)
        for value, name in map_base.items():
            if value in map_append:
                self.assertEqual(name, map_append[value])


class CliTest(unittest.TestCase):
    def test_usage_error(self):
        result = run_cli("split", "only-one-arg")
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr.startswith("error: "))

    def test_input_error_exit_code(self):
        fd, path = tempfile.mkstemp(suffix=".tsv")
        with os.fdopen(fd, "wb") as stream:
            stream.write(b"bad-line-without-tab\n")
        self.addCleanup(os.unlink, path)
        result = run_cli("stats", path, "s", "80:10:10")
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr.startswith("error: "))

    def test_report_sections(self):
        fd, path = tempfile.mkstemp(suffix=".html")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        result = run_cli("report", "samples", path)
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(path, encoding="utf-8") as stream:
            page = stream.read()
        self.assertEqual(page.count("<h2>"), 11)
        self.assertNotIn("http://", page)
        self.assertNotIn("https://", page)
        self.assertIn("width:80.5000%", page)
        self.assertIn("归属变化数：0", page)


if __name__ == "__main__":
    unittest.main()
