"""
Build a cleaned copy of YUKTHA_FINAL_MASTER_DATA_2026-09-02_ORGANIZED.

Read-only on the source tree. Writes a new sibling folder:
    YUKTHA_CLEAN_2026-09-07/

Changes made in the copy:
  - strip the 16-hex-char content-hash prefix ("<hash>__") from filenames
  - collapse "02_HISTORICAL_MARKET_DATA" -> "02_MARKET_DATA"
  - collapse the two inconsistent "other material" schemes
    (03_OTHER_SOURCE_MATERIAL/Other, 03_SOURCE_REPORTS/Reports) into:
        03_SOURCE_REPORTS/          (PDFs, de-duplicated by content hash)
        04_OTHER_SOURCE_MATERIAL/   (everything else from those two folders)
  - drop the redundant "Other"/"Reports" leaf directory (parent name already
    says what it is)
  - de-duplicate byte-identical files that used to live in two places
  - rewrite each company's SOURCE_INDEX.csv to point at the new paths
  - copy 00_MASTER_AUDIT_AND_METADATA unchanged
"""
import csv
import hashlib
import re
import shutil
from pathlib import Path

SRC_ROOT = Path(r"D:\code\GITHUB\THESIS\YUKTHA_FINAL_MASTER_DATA_2026-09-02_ORGANIZED")
DST_ROOT = Path(r"D:\code\GITHUB\THESIS\YUKTHA_CLEAN_2026-09-07")

HASH_PREFIX_RE = re.compile(r"^[0-9a-f]{16}__")


def strip_hash_prefix(name: str) -> str:
    return HASH_PREFIX_RE.sub("", name)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_dest(dest_dir: Path, filename: str, src_path: Path, seen_hashes: dict) -> Path | None:
    """Return the destination path to copy to, or None if this is a byte-identical
    duplicate of a file already placed at that name in that directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidate = dest_dir / filename
    key = (dest_dir, filename)

    if not candidate.exists() and key not in seen_hashes:
        seen_hashes[key] = file_sha256(src_path)
        return candidate

    existing_hash = seen_hashes.get(key)
    if existing_hash is None and candidate.exists():
        existing_hash = file_sha256(candidate)
        seen_hashes[key] = existing_hash

    new_hash = file_sha256(src_path)
    if new_hash == existing_hash:
        return None  # true duplicate, skip copy

    stem, dot, ext = filename.rpartition(".")
    disambiguated = f"{stem}_{new_hash[:8]}.{ext}" if dot else f"{filename}_{new_hash[:8]}"
    seen_hashes[(dest_dir, disambiguated)] = new_hash
    return dest_dir / disambiguated


def process_company(company_dir: Path, dst_company_dir: Path, path_map: dict):
    seen_hashes: dict = {}

    # 01_FINANCIAL_DATA/{Annual,Quarterly,Interim,Other} -> unchanged layout
    fin_src = company_dir / "01_FINANCIAL_DATA"
    if fin_src.is_dir():
        for sub in sorted(p for p in fin_src.iterdir() if p.is_dir()):
            for f in sorted(sub.glob("*")):
                if not f.is_file():
                    continue
                clean_name = strip_hash_prefix(f.name)
                dest_dir = dst_company_dir / "01_FINANCIAL_DATA" / sub.name
                dest = unique_dest(dest_dir, clean_name, f, seen_hashes)
                if dest:
                    shutil.copy2(f, dest)
                    path_map[f] = dest

    # 02_HISTORICAL_MARKET_DATA/{Daily,Weekly,Monthly} -> 02_MARKET_DATA/{same}
    mkt_src = company_dir / "02_HISTORICAL_MARKET_DATA"
    if mkt_src.is_dir():
        for sub in sorted(p for p in mkt_src.iterdir() if p.is_dir()):
            for f in sorted(sub.glob("*")):
                if not f.is_file():
                    continue
                clean_name = strip_hash_prefix(f.name)
                dest_dir = dst_company_dir / "02_MARKET_DATA" / sub.name
                dest = unique_dest(dest_dir, clean_name, f, seen_hashes)
                if dest:
                    shutil.copy2(f, dest)
                    path_map[f] = dest

    # 03_OTHER_SOURCE_MATERIAL/Other + 03_SOURCE_REPORTS/Reports
    # -> split by extension into 03_SOURCE_REPORTS (pdf) / 04_OTHER_SOURCE_MATERIAL (rest)
    other_dirs = [
        company_dir / "03_OTHER_SOURCE_MATERIAL" / "Other",
        company_dir / "03_SOURCE_REPORTS" / "Reports",
    ]
    for other_dir in other_dirs:
        if not other_dir.is_dir():
            continue
        for f in sorted(other_dir.glob("*")):
            if not f.is_file():
                continue
            clean_name = strip_hash_prefix(f.name)
            if f.suffix.lower() == ".pdf":
                dest_dir = dst_company_dir / "03_SOURCE_REPORTS"
            else:
                dest_dir = dst_company_dir / "04_OTHER_SOURCE_MATERIAL"
            dest = unique_dest(dest_dir, clean_name, f, seen_hashes)
            if dest:
                shutil.copy2(f, dest)
                path_map[f] = dest


def new_category_subcategory(dest_path: Path, dst_company_dir: Path):
    rel = dest_path.relative_to(dst_company_dir)
    parts = rel.parts
    top = parts[0]
    sub = parts[1] if top in ("01_FINANCIAL_DATA", "02_MARKET_DATA") and len(parts) > 1 else ""
    return top, sub


def rewrite_source_index(company_dir: Path, dst_company_dir: Path, path_map: dict):
    src_index = company_dir / "SOURCE_INDEX.csv"
    if not src_index.is_file():
        return

    with src_index.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    thesis_root_rel = dst_company_dir.parent.parent  # .../YUKTHA_CLEAN_2026-09-07

    out_rows = []
    for row in rows:
        old_organized = row["organized_path"]
        old_abs = SRC_ROOT.parent / old_organized  # organized_path is relative to SRC_ROOT.parent? check below
        # organized_path in the source csv is relative to SRC_ROOT itself
        old_abs = SRC_ROOT / old_organized
        dest = path_map.get(old_abs)
        if dest is None:
            # file was a duplicate that got skipped (byte-identical copy already placed
            # under a different original name) -- point at the surviving copy by name match
            continue
        new_organized = str(dest.relative_to(DST_ROOT)).replace("\\", "/")
        top, sub = new_category_subcategory(dest, dst_company_dir)
        row["organized_path"] = new_organized
        row["category"] = top
        row["subcategory"] = sub
        out_rows.append(row)

    dst_index = dst_company_dir / "SOURCE_INDEX.csv"
    with dst_index.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def main():
    if DST_ROOT.exists():
        raise SystemExit(f"Destination already exists, refusing to overwrite: {DST_ROOT}")

    DST_ROOT.mkdir(parents=True)

    # copy master audit metadata unchanged
    audit_src = SRC_ROOT / "00_MASTER_AUDIT_AND_METADATA"
    if audit_src.is_dir():
        shutil.copytree(audit_src, DST_ROOT / "00_MASTER_AUDIT_AND_METADATA")

    for region in ("01_INDIAN_COMPANIES", "02_NON_INDIAN_COMPANIES"):
        region_src = SRC_ROOT / region
        if not region_src.is_dir():
            continue
        for company_dir in sorted(p for p in region_src.iterdir() if p.is_dir()):
            dst_company_dir = DST_ROOT / region / company_dir.name
            path_map: dict = {}
            process_company(company_dir, dst_company_dir, path_map)
            rewrite_source_index(company_dir, dst_company_dir, path_map)
            print(f"done: {region}/{company_dir.name} ({len(path_map)} files)")

    print("\nAll done ->", DST_ROOT)


if __name__ == "__main__":
    main()
