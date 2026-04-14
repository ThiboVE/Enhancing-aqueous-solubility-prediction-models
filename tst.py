from pathlib import Path


def main() -> None:
    path = Path("1_KRR_topo_relevant_no_norm/results")
    for obj in path.glob("*"):
        name = obj.name
        parts = name.split("_")
        parts[6] = f"id={parts[6]}"
        new_name = "_".join(parts)

        obj.rename(obj.parent / new_name)


if __name__ == "__main__":
    main()
