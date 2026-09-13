"""Tests for solaris.utils.logging."""


# --- Logger names --------------------------------------------------------------------


def test_get_logger_keeps_independent_names():
    from solaris.utils import get_logger

    records = []
    a = get_logger("alpha")
    b = get_logger("beta")
    sink = a.add(lambda m: records.append(m.record["extra"].get("name")), level="INFO")
    try:
        a.info("from a")
        b.info("from b")
    finally:
        a.remove(sink)
    assert records == ["alpha", "beta"]
