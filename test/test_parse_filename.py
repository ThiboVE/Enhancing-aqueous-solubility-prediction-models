from pathlib import Path

from ml_enhance import parse_filename


def test_without_size() -> None:
    fold_id = 5
    size = 1.0
    file = Path(f"./t/test_filtered_id={fold_id}_results.pkl")

    results = parse_filename(file)

    assert results["fold_id"] == fold_id, f"fold_id was {results['fold_id']}, but should be {fold_id}"
    assert results["size"] == size, f"size was {results['size']}, but should be {size}"


def test_with_size() -> None:
    fold_id = 5
    size = 0.1
    file = Path(f"./t/test_filtered_id={fold_id}_size={size}_results.pkl")

    results = parse_filename(file)

    assert results["fold_id"] == fold_id, f"fold_id was {results['fold_id']}, but should be {fold_id}"
    assert results["size"] == size, f"size was {results['size']}, but should be {size}"


def test_size_1() -> None:
    fold_id = 5
    size = 1
    file = Path(f"./t/test_filtered_id={fold_id}_size={size}_results.pkl")

    results = parse_filename(file)

    assert results["fold_id"] == fold_id, f"fold_id was {results['fold_id']}, but should be {fold_id}"
    assert results["size"] == size, f"size was {results['size']}, but should be {size}"
