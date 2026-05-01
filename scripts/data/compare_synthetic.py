import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["source_text", "target_text", "source_lang", "target_lang"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).fillna("")


def _row_keys(df: pd.DataFrame) -> set:
    missing = [col for col in KEY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}")
    return set(map(tuple, df[KEY_COLUMNS].astype(str).values.tolist()))


def _counts(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="int64")
    return df[column].astype(str).value_counts().sort_index()


def _print_counts(title: str, old: pd.Series, new: pd.Series):
    labels = sorted(set(old.index) | set(new.index))
    if not labels:
        return
    print(f"\n{title}")
    print("value,old,new,delta")
    for label in labels:
        old_count = int(old.get(label, 0))
        new_count = int(new.get(label, 0))
        print(f"{label},{old_count},{new_count},{new_count - old_count}")


def compare(old_path: Path, new_path: Path, sample_size: int):
    old_df = _read_csv(old_path)
    new_df = _read_csv(new_path)

    old_keys = _row_keys(old_df)
    new_keys = _row_keys(new_df)
    added = new_keys - old_keys
    removed = old_keys - new_keys
    common = old_keys & new_keys

    print(f"OLD: {old_path} ({len(old_df)} rows)")
    print(f"NEW: {new_path} ({len(new_df)} rows)")
    print(f"rows_delta: {len(new_df) - len(old_df)}")
    print(f"unique_old: {len(old_keys)}")
    print(f"unique_new: {len(new_keys)}")
    print(f"common_rows: {len(common)}")
    print(f"added_rows: {len(added)}")
    print(f"removed_rows: {len(removed)}")

    old_sources = set(old_df.get("source_text", pd.Series(dtype=str)).astype(str))
    new_sources = set(new_df.get("source_text", pd.Series(dtype=str)).astype(str))
    print(f"common_sources: {len(old_sources & new_sources)}")
    print(f"added_sources: {len(new_sources - old_sources)}")
    print(f"removed_sources: {len(old_sources - new_sources)}")

    _print_counts("target_lang_counts", _counts(old_df, "target_lang"), _counts(new_df, "target_lang"))
    _print_counts("origin_counts", _counts(old_df, "origin"), _counts(new_df, "origin"))

    if sample_size > 0 and added:
        print("\nadded_sample")
        added_df = pd.DataFrame(list(added), columns=KEY_COLUMNS).head(sample_size)
        print(added_df.to_csv(index=False).strip())

    if sample_size > 0 and removed:
        print("\nremoved_sample")
        removed_df = pd.DataFrame(list(removed), columns=KEY_COLUMNS).head(sample_size)
        print(removed_df.to_csv(index=False).strip())


def main():
    parser = argparse.ArgumentParser(description="Compare two synthetic translation CSV datasets.")
    parser.add_argument("old_csv", type=Path)
    parser.add_argument("new_csv", type=Path)
    parser.add_argument("--sample-size", type=int, default=5)
    args = parser.parse_args()
    compare(args.old_csv, args.new_csv, args.sample_size)


if __name__ == "__main__":
    main()
