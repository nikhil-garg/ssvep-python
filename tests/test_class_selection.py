from ssvep_toolkit.evaluation.class_selection import (
    factorial_class_sets,
    harmonic_collisions,
    select_class_frequencies,
)


def test_compact_harmonic_aware_sets_are_consecutive_and_collision_minimal():
    expected = {
        2: tuple(range(8, 10)),
        4: tuple(range(8, 12)),
        8: tuple(range(8, 16)),
        16: tuple(range(16, 32)),
        32: tuple(range(28, 60)),
    }
    for count, frequencies in expected.items():
        assert select_class_frequencies(count) == frequencies
        assert len(frequencies) == count
        assert max(frequencies) - min(frequencies) == count - 1


def test_collision_audit_reports_only_unavoidable_32_class_pairs():
    for count in (2, 4, 8, 16):
        assert harmonic_collisions(select_class_frequencies(count)) == ()
    collisions = harmonic_collisions(select_class_frequencies(32))
    assert [(x.source_hz, x.harmonic, x.target_hz) for x in collisions] == [
        (28, 2, 56), (29, 2, 58)
    ]


def test_low_contiguous_is_available_as_spacing_ablation():
    assert select_class_frequencies(4, strategy="low_contiguous") == (8, 9, 10, 11)


def test_targeted_fixed_spacing_sets_avoid_exact_second_and_third_harmonics():
    four = select_class_frequencies(
        4, strategy="fixed_spacing_harmonic_aware", spacing_hz=4, start_hz=17,
    )
    sixteen = select_class_frequencies(
        16, strategy="fixed_spacing_harmonic_aware", spacing_hz=2, start_hz=17,
    )
    assert four == (17, 21, 25, 29)
    assert sixteen == tuple(range(17, 48, 2))
    assert set(four).issubset(sixteen)
    assert harmonic_collisions(four) == ()
    assert harmonic_collisions(sixteen) == ()


def test_factorial_class_design_keeps_spacing_span_and_collisions_explicit():
    designs = factorial_class_sets(
        (4,), (2, 4), available_hz=(8, 30), starts_hz=(13, 17), maximum_collisions=0,
    )
    assert {(item.start_hz, item.spacing_hz) for item in designs} == {(13, 2), (13, 4), (17, 2), (17, 4)}
    assert all(item.span_hz == item.spacing_hz * 3 for item in designs)
    assert all(item.harmonic_collisions == () for item in designs)
