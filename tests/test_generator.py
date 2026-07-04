"""Tests for the (honest) prediction generators."""

from src.lotto_ds import generator as gen


def test_lotto_sets_are_valid():
    for method in gen.LOTTO_METHODS:
        sets = gen.generate_lotto(method, count=5, seed=1)
        assert len(sets) == 5
        for s in sets:
            nums = s["numbers"]
            assert len(nums) == 6 and len(set(nums)) == 6
            assert all(1 <= n <= 45 for n in nums)
            assert nums == sorted(nums)
            assert s["sum"] == sum(nums)


def test_lotto_all_methods_payload():
    out = gen.generate_all_lotto(count=3, seed=0)
    assert set(out["methods"]) == set(gen.LOTTO_METHODS)
    assert out["disclaimer"]


def test_pension_sets_are_valid():
    for method in gen.PENSION_METHODS:
        sets = gen.generate_pension(method, count=5, seed=2)
        assert len(sets) == 5
        for s in sets:
            assert 1 <= s["group"] <= 5
            assert len(s["digits"]) == 6
            assert all(0 <= d <= 9 for d in s["digits"])


def test_unknown_method_raises():
    import pytest
    with pytest.raises(ValueError):
        gen.generate_lotto("nope")
    with pytest.raises(ValueError):
        gen.generate_pension("nope")


def test_records_handles_both_hit_schemas():
    from src.lotto_ds import records as rec
    # old schema (hits) and recent lotto schema (hits_1st) both parse
    assert rec._method_hits({"hits": [1, 0, 2]}) == [1, 0, 2]
    assert rec._method_hits({"hits_1st": [0, 1]}) == [0, 1]
    assert rec._method_hits({"sets_generated": []}) == []
