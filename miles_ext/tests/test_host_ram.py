from miles_ext.obs import host_ram


def test_parsers_and_peak_line():
    status = host_ram.parse_kb_lines("VmRSS:\t 2097152 kB\nRssAnon:\t1048576 kB\nThreads: 9\n", ("VmRSS", "RssAnon"))
    assert status == {"VmRSS": 2 * host_ram.GIB, "RssAnon": host_ram.GIB}
    assert host_ram.parse_cgroup_stat("anon 10\nfile 20\nkernel 5\n") == {"cg_anon": 10, "cg_file": 20}
    peak = host_ram.merge_peak({"VmRSS": 3, "Cached": 1}, {"VmRSS": 2, "Dirty": 4})
    assert peak == {"VmRSS": 3, "Cached": 1, "Dirty": 4}
    line = host_ram.format_line(0, {"VmRSS": 2 * host_ram.GIB})
    assert line == "[host ram] rank=0 peak over 60s (GiB): VmRSS=2.0"


def test_node_sample_reads_this_host():
    sample = host_ram.node_sample()
    assert sample["MemTotal"] > 0 and "tmp_used" in sample
