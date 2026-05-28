# import re
# from collections.abc import Sequence
# from pathlib import Path


# def group_ranges(nums: Sequence[int]) -> list[str]:
#     if not nums:
#         return []

#     ranges = []
#     start = prev = nums[0]

#     for n in nums[1:]:
#         if n == prev + 1:
#             prev = n
#         else:
#             ranges.append(f"{start}-{prev}" if start != prev else str(start))
#             start = prev = n

#     ranges.append(f"{start}-{prev}" if start != prev else str(start))

#     return ranges


# def main() -> None:
#     path = Path("run_dft_rerun/results")

#     files: list[Path] = list(path.glob("*"))

#     ids = [int(re.search(r"\d+", file.stem).group(0)) for file in files]

#     failed = [id for id in range(1000) if id not in ids]

#     print(",".join(group_ranges(failed)))


# if __name__ == "__main__":
#     main()
