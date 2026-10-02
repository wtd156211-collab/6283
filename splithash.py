#!/usr/bin/env python3
"""稳定数据划分：键 -> sha256 -> 固定桶 -> 份。

归属是 (盐值, 比例, 键) 的纯函数，与数据规模、行序、其它记录无关。
逐行单遍处理，除唯一键集合与类别计数器外不保留历史。

用法：
    python3 splithash.py split  <数据.tsv> <盐值> <比例>
    python3 splithash.py stats  <数据.tsv> <盐值> <比例>
    python3 splithash.py check  <样例根目录>
    python3 splithash.py report <样例根目录> <输出.html>
"""

import hashlib
import html
import sys

BUCKETS = 10000
SPLIT_NAMES = ("train", "valid", "test")
SPLIT_NAMES_B = (b"train", b"valid", b"test")
FLUSH_LINES = 8192

USAGE = ("用法: splithash.py split <数据.tsv> <盐值> <比例> | "
         "stats <数据.tsv> <盐值> <比例> | check <样例根目录> | "
         "report <样例根目录> <输出.html>")


class InputError(Exception):
    """用法或输入数据错误（退出码 2）。"""


def parse_salt(text):
    if not text:
        raise InputError("盐值不能为空")
    if "\x00" in text:
        raise InputError("盐值不能含 NUL")
    return text.encode("utf-8")


def parse_ratio(text):
    parts = text.split(":")
    if len(parts) != 3:
        raise InputError(f"比例须为 w1:w2:w3 三个正整数且和为 100: {text!r}")
    weights = []
    for part in parts:
        if not (part.isascii() and part.isdigit()):
            raise InputError(f"比例须为正整数: {text!r}")
        value = int(part)
        if value <= 0:
            raise InputError(f"比例须为正整数: {text!r}")
        weights.append(value)
    if sum(weights) != 100:
        raise InputError(f"比例之和须为 100: {text!r}")
    return weights


def iter_records(path):
    """逐行产出 (原始行, 键字节, 键, 类别标签)，校验输入格式。"""
    try:
        stream = open(path, "rb")
    except OSError as exc:
        raise InputError(f"无法读取数据文件: {exc}") from None
    with stream:
        for lineno, raw in enumerate(stream, 1):
            if not raw.endswith(b"\n"):
                raise InputError(f"{path}:{lineno}: 行尾缺少 LF 换行")
            line = raw[:-1]
            if lineno == 1 and line.startswith(b"\xef\xbb\xbf"):
                raise InputError(f"{path}:1: 文件含 BOM")
            if line.count(b"\t") != 1:
                raise InputError(f"{path}:{lineno}: 每行须恰好一个 TAB")
            key_b, label_b = line.split(b"\t")
            if not key_b or not label_b:
                raise InputError(f"{path}:{lineno}: 键与类别标签均须非空")
            if min(key_b) < 0x20 or b"\x7f" in key_b \
                    or min(label_b) < 0x20 or b"\x7f" in label_b:
                raise InputError(f"{path}:{lineno}: 含控制字符")
            try:
                key = key_b.decode("utf-8")
                label = label_b.decode("utf-8")
            except UnicodeDecodeError:
                raise InputError(f"{path}:{lineno}: 非有效 UTF-8")
            yield line, key_b, key, label


class Splitter:
    """(盐值, 比例) 决定的纯函数映射：键 -> 桶 -> 份。"""

    def __init__(self, salt_bytes, weights):
        self._base = hashlib.sha256(salt_bytes + b"\x00")
        self._cut1 = 100 * weights[0]
        self._cut2 = 100 * (weights[0] + weights[1])

    def value_of(self, key_bytes):
        digest = self._base.copy()
        digest.update(key_bytes)
        return int.from_bytes(digest.digest()[:8], "big")

    def index_of_value(self, value):
        bucket = value % BUCKETS
        if bucket < self._cut1:
            return 0
        if bucket < self._cut2:
            return 1
        return 2

    def index_of(self, key_bytes):
        return self.index_of_value(self.value_of(key_bytes))


def split_lines(path, splitter):
    """产出映射行（bytes，不含换行），按输入顺序。"""
    for line, key_b, _key, _label in iter_records(path):
        yield line + b"\t" + SPLIT_NAMES_B[splitter.index_of(key_b)]


def write_split(path, splitter, out):
    buf = []
    for row in split_lines(path, splitter):
        buf.append(row)
        if len(buf) >= FLUSH_LINES:
            out.write(b"\n".join(buf) + b"\n")
            buf.clear()
    if buf:
        out.write(b"\n".join(buf) + b"\n")


class Stats:
    """单遍累加：总行数、唯一键、每份行数、每类别行数、每类别x份行数。"""

    def __init__(self):
        self.total = 0
        self.seen = set()
        self.split_counts = [0, 0, 0]
        self.categories = {}

    def add(self, value, split_index, label):
        self.total += 1
        self.seen.add(value)
        self.split_counts[split_index] += 1
        counts = self.categories.get(label)
        if counts is None:
            counts = self.categories[label] = [0, 0, 0, 0]
        counts[0] += 1
        counts[1 + split_index] += 1

    @property
    def unique_keys(self):
        return len(self.seen)


def compute_stats(path, splitter):
    stats = Stats()
    for _line, key_b, _key, label in iter_records(path):
        value = splitter.value_of(key_b)
        stats.add(value, splitter.index_of_value(value), label)
    return stats


def pct_text(count, total):
    if total == 0:
        return "0.0000%"
    return format(count * 100 / total, ".4f") + "%"


def format_stats(stats):
    lines = [f"total\t{stats.total}\t{stats.unique_keys}"]
    for index, name in enumerate(SPLIT_NAMES):
        count = stats.split_counts[index]
        lines.append(f"split\t{name}\t{count}\t{pct_text(count, stats.total)}")
    for label in sorted(stats.categories):
        total, train, valid, test = stats.categories[label]
        lines.append(f"category\t{label}\t{total}\t{train}\t{valid}\t{test}")
    return "\n".join(lines) + "\n"


def key_split_map(path, splitter):
    """唯一键（以哈希值计）-> 份名。"""
    result = {}
    for _line, key_b, _key, _label in iter_records(path):
        value = splitter.value_of(key_b)
        result[value] = SPLIT_NAMES[splitter.index_of_value(value)]
    return result


def stability_counts(path_a, path_b, splitter):
    """返回 (保留键, 只在A, 只在B, 归属变化)，按唯一键计。"""
    map_a = key_split_map(path_a, splitter)
    map_b = key_split_map(path_b, splitter)
    common = set(map_a) & set(map_b)
    changed = sum(1 for value in common if map_a[value] != map_b[value])
    return len(common), len(map_a) - len(common), len(map_b) - len(common), changed


def parse_stability_expected(text):
    values = {}
    for line in text.splitlines():
        name, _, num = line.partition("\t")
        values[name] = int(num)
    return (values["保留键"], values["只在A"], values["只在B"], values["归属变化"])


def read_manifest(path, fields):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as stream:
        lines = stream.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) != fields:
            raise InputError(f"清单行须为 {fields} 列: {line!r}")
        rows.append(parts)
    return rows


def load_case(root, row):
    name, data, salt, ratio, map_rel, stats_rel = row
    splitter = Splitter(parse_salt(salt), parse_ratio(ratio))
    return name, f"{root}/{data}", splitter, map_rel, stats_rel


def check_case(root, row):
    """核对单条样例，返回 (是否通过, Stats)。"""
    name, data_path, splitter, map_rel, stats_rel = load_case(root, row)
    ok = True
    if map_rel != "-":
        produced = b"\n".join(split_lines(data_path, splitter)) + b"\n"
        with open(f"{root}/{map_rel}", "rb") as stream:
            if produced != stream.read():
                ok = False
    stats = compute_stats(data_path, splitter)
    with open(f"{root}/{stats_rel}", "rb") as stream:
        if format_stats(stats).encode("utf-8") != stream.read():
            ok = False
    return ok, stats


def check_stability(root, row):
    """核对一组稳定性对，返回 (是否通过, 四元组)。"""
    name, data_a, data_b, salt, ratio, expected_rel = row
    splitter = Splitter(parse_salt(salt), parse_ratio(ratio))
    counts = stability_counts(f"{root}/{data_a}", f"{root}/{data_b}", splitter)
    with open(f"{root}/{expected_rel}", "r", encoding="utf-8") as stream:
        expected = parse_stability_expected(stream.read())
    return counts == expected, counts


def cmd_check(root):
    cases = read_manifest(f"{root}/cases.tsv", 6)
    pairs = read_manifest(f"{root}/stability.tsv", 6)
    mismatches = []
    total = 0
    for row in cases:
        ok, _stats = check_case(root, row)
        total += 1
        if not ok:
            mismatches.append(row[0])
    for row in pairs:
        ok, _counts = check_stability(root, row)
        total += 1
        if not ok:
            mismatches.append(row[0])
    if mismatches:
        for name in mismatches:
            print(f"mismatch={name}")
        return 1
    print(f"ok={total}")
    return 0


HTML_HEAD = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>稳定数据划分报告</title>
<style>
body{font-family:"Segoe UI","Microsoft YaHei",sans-serif;margin:2em auto;max-width:960px;color:#222}
h1{border-bottom:3px solid #2c6fbb;padding-bottom:.3em}
h2{margin-top:2em;border-bottom:1px solid #ccc;padding-bottom:.2em}
table{border-collapse:collapse;margin:.6em 0}
th,td{border:1px solid #bbb;padding:.3em .8em;text-align:right}
th:first-child,td:first-child{text-align:left}
tr:nth-child(even){background:#f4f7fb}
.bar{background:#e3e8ef;border-radius:3px;height:1em;width:240px;display:inline-block;vertical-align:middle}
.fill{background:#2c6fbb;height:100%;border-radius:3px}
.pass{color:#1a7f37;font-weight:bold}
.fail{color:#c0392b;font-weight:bold}
.summary td{text-align:left}
.mono{font-family:Consolas,monospace}
</style>
</head>
<body>
"""


def esc(text):
    return html.escape(text, quote=True)


def render_case_section(name, stats, ok):
    parts = [f"<h2>{esc(name)}</h2>"]
    verdict = '<span class="pass">通过</span>' if ok else '<span class="fail">不符</span>'
    parts.append(f"<p>行数 {stats.total} ｜ 唯一键 {stats.unique_keys} ｜ 核对结论：{verdict}</p>")
    parts.append("<table><tr><th>份</th><th>行数</th><th>占比</th><th>占比条</th></tr>")
    for index, split_name in enumerate(SPLIT_NAMES):
        count = stats.split_counts[index]
        pct = pct_text(count, stats.total)
        parts.append(
            f'<tr><td>{split_name}</td><td>{count}</td><td class="mono">{pct}</td>'
            f'<td><span class="bar"><span class="fill" style="width:{pct}"></span></span></td></tr>')
    parts.append("</table>")
    parts.append("<table><tr><th>类别</th><th>总行数</th><th>train</th><th>valid</th><th>test</th></tr>")
    for label in sorted(stats.categories):
        total, train, valid, test = stats.categories[label]
        parts.append(f"<tr><td>{esc(label)}</td><td>{total}</td><td>{train}</td>"
                     f"<td>{valid}</td><td>{test}</td></tr>")
    parts.append("</table>")
    return "\n".join(parts)


def render_stability_section(name, counts, expected, ok):
    retained, only_a, only_b, changed = counts
    verdict = '<span class="pass">通过</span>' if ok else '<span class="fail">不符</span>'
    rows = "".join(
        f"<tr><td>{label}</td><td>{value}</td><td>{exp}</td></tr>"
        for (label, value), exp in zip(
            zip(("保留键", "只在A", "只在B", "归属变化"), counts), expected))
    return (f"<h2>{esc(name)}（稳定性）</h2>\n"
            f"<p>核对结论：{verdict} ｜ 追加/删除前后已有记录的归属变化数：{changed}</p>\n"
            f"<table><tr><th>指标</th><th>实际</th><th>期望</th></tr>{rows}</table>")


def cmd_report(root, out_path):
    cases = read_manifest(f"{root}/cases.tsv", 6)
    pairs = read_manifest(f"{root}/stability.tsv", 6)
    sections = []
    total_rows = 0
    passed = 0
    checks = 0
    for row in cases:
        ok, stats = check_case(root, row)
        checks += 1
        passed += ok
        total_rows += stats.total
        sections.append(render_case_section(row[0], stats, ok))
    for row in pairs:
        ok, counts = check_stability(root, row)
        checks += 1
        passed += ok
        with open(f"{root}/{row[5]}", "r", encoding="utf-8") as stream:
            expected = parse_stability_expected(stream.read())
        sections.append(render_stability_section(row[0], counts, expected, ok))
    summary = (f"<h1>稳定数据划分报告</h1>\n"
               f'<table class="summary">'
               f"<tr><th>样例数</th><td>{len(cases)}（另有 {len(pairs)} 组稳定性对）</td></tr>"
               f"<tr><th>总行数</th><td>{total_rows}</td></tr>"
               f"<tr><th>通过数</th><td>{passed} / {checks}</td></tr></table>")
    page = HTML_HEAD + summary + "\n" + "\n".join(sections) + "\n</body>\n</html>\n"
    with open(out_path, "w", encoding="utf-8", newline="") as stream:
        stream.write(page)
    return 0


def main(argv):
    if len(argv) < 2:
        print(f"error: {USAGE}", file=sys.stderr)
        return 2
    command = argv[1]
    try:
        if command == "split" and len(argv) == 5:
            splitter = Splitter(parse_salt(argv[3]), parse_ratio(argv[4]))
            write_split(argv[2], splitter, sys.stdout.buffer)
            return 0
        if command == "stats" and len(argv) == 5:
            splitter = Splitter(parse_salt(argv[3]), parse_ratio(argv[4]))
            sys.stdout.write(format_stats(compute_stats(argv[2], splitter)))
            return 0
        if command == "check" and len(argv) == 3:
            return cmd_check(argv[2])
        if command == "report" and len(argv) == 4:
            return cmd_report(argv[2], argv[3])
        raise InputError(USAGE)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
