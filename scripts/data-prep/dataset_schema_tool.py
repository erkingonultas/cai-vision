""" HOW TO USE 

    # 0) Make sure classes.csv is synced with category.txt
    `python dataset_schema_tool.py init-from-category`

    # 1) Put new images under: new-data/<class_name>/*.jpg
    #    (class_name can match display_name_en, slug, or any synonym — case/space/underscore-insensitive)

    # 2) Ingest them + renumber targets to 1.jpg..N.jpg
    `python dataset_schema_tool.py ingest-new-data`

    #    If a <class_name> in new-data doesn't exist yet, either:
    #    A) create it manually:
    `python dataset_schema_tool.py add-class "New Dish Name"`
    #       (then re-run ingest), or

    #    B) let the tool auto-create:
    `python dataset_schema_tool.py ingest-new-data --create-missing`

    # 3) Sanity-check the whole dataset
    `python dataset_schema_tool.py validate`

    # 4) Optional: deduplicate within a class and exclude known cross-class dupes
    `python dataset_schema_tool.py build-manifests --val 0.10 --test 0.15 --seed 1337 --dedup-in-class`

    # 5) Default behavior (no in-class dedup):
    `python dataset_schema_tool.py build-manifests`

"""

#!/usr/bin/env python3

import random
import argparse, csv, hashlib, os, re, sys, shutil
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path("../../datasets/cai-vision-dataset")
CAT_FILE = DATA_ROOT / "category.txt"
CLASSES_CSV = DATA_ROOT / "classes.csv"
MULTILABEL_OVERRIDES = DATA_ROOT / "multilabel_overrides.csv"
IMG_DIR = DATA_ROOT / "images"
NEW_DATA_DIR = DATA_ROOT / "new-data"

SLUG_RX = re.compile(r"[^a-z0-9\-]+")
IMG_EXTS = {".jpg", ".jpeg"}  # keep tight per your dataset; add ".png" if needed

def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = SLUG_RX.sub("", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s

def normalize_name(s: str) -> str:
    # normalize user folder names like "Chicken_Caesar  Salad" -> "chicken caesar salad"
    s = s.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def read_category_txt():
    """
    Expects lines like:
    1 Chicken Caesar Salad
    2 Menemen
    3 Lentil Soup
    """
    items = []
    if not CAT_FILE.exists():
        print(f"[ERROR] {CAT_FILE} not found.", file=sys.stderr)
        sys.exit(1)
    with CAT_FILE.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                print(f"[ERROR] category.txt line {ln}: expected '<id> <name>'", file=sys.stderr)
                sys.exit(1)
            try:
                cid = int(parts[0])
            except:
                print(f"[ERROR] category.txt line {ln}: invalid id '{parts[0]}'", file=sys.stderr)
                sys.exit(1)
            name = parts[1].strip()
            items.append((cid, name))
    return items

def load_existing_classes_csv():
    rows = {}
    if CLASSES_CSV.exists():
        with CLASSES_CSV.open("r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                r["class_id"] = int(r["class_id"])
                rows[r["class_id"]] = r
    return rows

def write_classes_csv(rows_by_id):
    fieldnames = [
        "class_id","slug","display_name_en","synonyms_en","status","parent_id","notes"
    ]
    with CLASSES_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for cid in sorted(rows_by_id.keys()):
            r = rows_by_id[cid]
            # normalize required fields
            r.setdefault("synonyms_en","")
            r.setdefault("status","active")
            r.setdefault("parent_id","")
            r.setdefault("notes","")
            w.writerow({
                "class_id": cid,
                "slug": r.get("slug", slugify(r["display_name_en"])),
                "display_name_en": r["display_name_en"],
                "synonyms_en": r["synonyms_en"],
                "status": r["status"],
                "parent_id": r["parent_id"],
                "notes": r["notes"],
            })

def cmd_init_from_category(args):
    cats = read_category_txt()
    existing = load_existing_classes_csv()

    # Build/merge
    out = dict(existing)
    name_seen = set()
    for cid, name in cats:
        if cid in out:
            # preserve existing row; update display name if changed (notes will log)
            if out[cid]["display_name_en"] != name:
                out[cid]["notes"] = (out[cid].get("notes","") + f" | renamed from '{out[cid]['display_name_en']}'").strip(" |")
                out[cid]["display_name_en"] = name
                out[cid]["slug"] = slugify(name)
        else:
            out[cid] = {
                "class_id": cid,
                "slug": slugify(name),
                "display_name_en": name,
                "synonyms_en": "",
                "status": "active",
                "parent_id": "",
                "notes": "",
            }
        if out[cid]["display_name_en"] in name_seen:
            print(f"[WARN] Duplicate display name: {name}", file=sys.stderr)
        name_seen.add(out[cid]["display_name_en"])

    write_classes_csv(out)
    print(f"[OK] Wrote {CLASSES_CSV} with {len(out)} classes.")

def cmd_validate(args):
    # 1) classes.csv presence & fields
    rows = load_existing_classes_csv()
    if not rows:
        print("[ERROR] classes.csv not found or empty. Run init-from-category first.", file=sys.stderr)
        sys.exit(1)

    # 2) Ensure all folders have class rows and vice versa
    # folders: numeric names at root
    print("Checking folder structure...")
    folder_ids = set()
    if IMG_DIR.exists():
        for p in IMG_DIR.iterdir():
            if p.is_dir() and p.name.isdigit():
                folder_ids.add(int(p.name))
    class_ids = set(rows.keys())

    missing_class = folder_ids - class_ids
    missing_folder = class_ids - folder_ids

    if missing_class:
        print(f"[ERROR] Folders without schema rows: {sorted(missing_class)}", file=sys.stderr)
    if missing_folder:
        print(f"[WARN] Schema has classes with no folder: {sorted(missing_folder)}")

    # 3) Per-row checks
    print("Per-row checks...")
    slugs = set()
    names = set()
    for cid, r in rows.items():
        if r["status"] not in ("active","deprecated","merged"):
            print(f"[ERROR] class_id {cid} has invalid status '{r['status']}'", file=sys.stderr)
        s = r["slug"]
        if not s:
            print(f"[ERROR] class_id {cid} empty slug", file=sys.stderr)
        if s in slugs:
            print(f"[ERROR] duplicate slug '{s}'", file=sys.stderr)
        slugs.add(s)
        dn = r["display_name_en"].strip()
        if not dn:
            print(f"[ERROR] class_id {cid} empty display_name_en", file=sys.stderr)
        if dn in names:
            print(f"[WARN] duplicate display_name_en '{dn}' (allowed but discouraged)")
        names.add(dn)

    # 4) Image cross-label duplicates (same file content in multiple folders)
    # We compute a quick hash; if the same hash appears under different class folders -> report & suggest moving to multilabel_overrides
    print("Image cross-label duplicate checks...")
    hash_to_labels = defaultdict(set)
    problems = []
    for fid in folder_ids:
        folder = IMG_DIR / str(fid)
        for img in folder.glob("*.jpg"):
            try:
                with img.open("rb") as f:
                    h = hashlib.sha1(f.read()).hexdigest()
                hash_to_labels[h].add(fid)
            except Exception as e:
                print(f"[WARN] failed to hash {img}: {e}", file=sys.stderr)
    for h, labels in hash_to_labels.items():
        if len(labels) > 1:
            problems.append((h, sorted(labels)))

    if problems:
        print(f"[WARN] Found {len(problems)} cross-labeled duplicate images (same content in multiple class folders).")
        print("       For single-label training, add them to multilabel_overrides.csv and exclude from train.")
        # Create or append a CSV with guidance rows (no file paths because same content exists in multiple places)
        if not MULTILABEL_OVERRIDES.exists():
            with MULTILABEL_OVERRIDES.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["sha1","class_ids","notes"])
        with MULTILABEL_OVERRIDES.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for h, labels in problems:
                w.writerow([h, "|".join(map(str, labels)), "duplicate content in multiple class folders"])

    if missing_class or problems:
        print("[DONE] Validation completed with findings (see above).", file=sys.stderr)
        sys.exit(2)
    else:
        print("[OK] Validation passed: schema and folders are consistent, no cross-label dupes blocking single-label training.")

def cmd_add_class(args):
    rows = load_existing_classes_csv()
    if not rows:
        print("[ERROR] classes.csv not found. Run init-from-category first.", file=sys.stderr)
        sys.exit(1)
    next_id = max(rows.keys()) + 1
    name = args.name.strip().lower()
    rows[next_id] = {
        "class_id": next_id,
        "slug": slugify(name),
        "display_name_en": name,
        "synonyms_en": "",
        "status": "active",
        "parent_id": "",
        "notes": "",
    }
    write_classes_csv(rows)
    # create folder
    (IMG_DIR / str(next_id)).mkdir(parents=True, exist_ok=True)
    # append to category.txt for traceability
    with CAT_FILE.open("a", encoding="utf-8") as f:
        f.write(f"\n{next_id}   {name}")
    print(f"[OK] Added class {next_id}: {name}")

def build_name_index(rows_by_id):
    """
    Build a case-insensitive index from display_name, slug, and synonyms to class_id.
    """
    idx = {}
    for cid, r in rows_by_id.items():
        names = set()
        names.add(normalize_name(r["display_name_en"]))
        names.add(normalize_name(r.get("slug","")))
        syns = (r.get("synonyms_en") or "").strip()
        if syns:
            for s in syns.split("|"):
                if s.strip():
                    names.add(normalize_name(s))
        for n in names:
            if n:
                if n in idx and idx[n] != cid:
                    # warn but last one wins (extremely unlikely)
                    print(f"[WARN] name index collision: '{n}' -> {idx[n]} / {cid}", file=sys.stderr)
                idx[n] = cid
    return idx

def sha1_of_file(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def index_existing_hashes():
    """
    Build a global hash index across images/<class_id>/...
    Returns: dict[sha1] = (class_id, path)
    """
    idx = {}
    if not IMG_DIR.exists(): 
        return idx
    for class_dir in IMG_DIR.iterdir():
        if not class_dir.is_dir() or not class_dir.name.isdigit():
            continue
        cid = int(class_dir.name)
        for img in class_dir.iterdir():
            if img.is_file() and img.suffix.lower() in IMG_EXTS:
                try:
                    h = sha1_of_file(img)
                    idx[h] = (cid, img)
                except Exception as e:
                    print(f"[WARN] Failed to hash {img}: {e}", file=sys.stderr)
    return idx

def safe_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))

def natural_key(path: Path):
    # helps stable ordering for renumbering
    s = path.stem
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p for p in parts]

def renumber_class_folder(cid: int):
    """
    Rename all images under images/<cid>/ to 1.jpg..N.jpg (stable order),
    using a temp directory to avoid name collisions.
    """
    class_dir = IMG_DIR / str(cid)
    if not class_dir.exists(): 
        return
    files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not files:
        return
    files.sort(key=natural_key)

    tmp_dir = class_dir / "__tmp_renumber__"
    tmp_dir.mkdir(exist_ok=True)

    # First move everything into tmp with stable numeric order
    for i, p in enumerate(files, 1):
        tmp_target = tmp_dir / f"{i:08d}{p.suffix.lower()}"
        shutil.move(str(p), str(tmp_target))

    # Then move back renamed to 1..N.jpg
    i = 1
    for p in sorted(tmp_dir.iterdir()):
        final = class_dir / f"{i}.jpg"
        shutil.move(str(p), str(final))
        i += 1

    tmp_dir.rmdir()

def cmd_ingest_new_data(args):
    """
    Ingest images from new-data/<class_name>/* into images/<class_id>/*,
    skip exact duplicates, log cross-class duplicates, and renumber to 1..N.jpg.
    """
    rows = load_existing_classes_csv()
    if not rows:
        print("[ERROR] classes.csv not found. Run init-from-category first.", file=sys.stderr)
        sys.exit(1)

    name_idx = build_name_index(rows)
    global_hash_idx = index_existing_hashes()

    if not NEW_DATA_DIR.exists():
        print(f"[ERROR] {NEW_DATA_DIR} not found.", file=sys.stderr)
        sys.exit(1)

    added_counts = defaultdict(int)
    skipped_dupe_same = 0
    cross_class_dupe = 0
    created_classes = []

    if not MULTILABEL_OVERRIDES.exists():
        with MULTILABEL_OVERRIDES.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["sha1","class_ids","notes"])

    # Walk new-data
    for cdir in NEW_DATA_DIR.iterdir():
        if not cdir.is_dir():
            continue
        raw_name = cdir.name
        key = normalize_name(raw_name)
        target_cid = name_idx.get(key)

        # Create new class if requested and unknown
        if target_cid is None:
            if args.create_missing:
                # create a new class with display name = raw_name (nicely capitalized)
                disp = " ".join(w.capitalize() for w in normalize_name(raw_name).split())
                # choose next id
                existing_ids = list(rows.keys())
                next_id = (max(existing_ids) + 1) if existing_ids else 1
                rows[next_id] = {
                    "class_id": next_id,
                    "slug": slugify(disp),
                    "display_name_en": disp,
                    "synonyms_en": "",
                    "status": "active",
                    "parent_id": "",
                    "notes": "auto-created via ingest-new-data",
                }
                write_classes_csv(rows)
                with CAT_FILE.open("a", encoding="utf-8") as f:
                    f.write(f"\n{next_id} {disp.lower()}")
                (IMG_DIR / str(next_id)).mkdir(parents=True, exist_ok=True)
                target_cid = next_id
                name_idx = build_name_index(rows)  # refresh index
                created_classes.append((next_id, disp))
                print(f"[OK] Created new class {next_id} for '{raw_name}'")
            else:
                print(f"[ERROR] Unknown class name '{raw_name}'. Use --create-missing to auto-create.", file=sys.stderr)
                continue

        class_dir = IMG_DIR / str(target_cid)
        class_dir.mkdir(parents=True, exist_ok=True)

        # Build per-class existing hashes to speed up same-class dupe detection
        existing_files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        existing_hashes_this_class = set()
        for p in existing_files:
            try:
                existing_hashes_this_class.add(sha1_of_file(p))
            except Exception as e:
                print(f"[WARN] Failed hashing {p}: {e}", file=sys.stderr)

        # Ingest files
        to_renumber = False
        for img in cdir.iterdir():
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                continue
            try:
                h = sha1_of_file(img)
            except Exception as e:
                print(f"[WARN] Failed to hash {img}: {e}", file=sys.stderr)
                continue

            # Skip if exact same image already in this class
            if h in existing_hashes_this_class:
                skipped_dupe_same += 1
                continue

            # If same hash exists in another class, log multilabel
            if h in global_hash_idx:
                other_cid, other_path = global_hash_idx[h]
                if other_cid != target_cid:
                    cross_class_dupe += 1
                    with MULTILABEL_OVERRIDES.open("a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow([h, f"{other_cid}|{target_cid}", f"ingest conflict: {img} equals {other_path.name}"])
                    # Still place it in the target class (keep both), but it will be excluded later by your pipelines if needed.

            # Place with a temporary name; final renumbering happens after the loop
            tmp_name = class_dir / f"__ingest__{h}{img.suffix.lower()}"
            safe_move(img, tmp_name)
            global_hash_idx[h] = (target_cid, tmp_name)
            existing_hashes_this_class.add(h)
            added_counts[target_cid] += 1
            to_renumber = True

        # Renumber class folder to 1..N.jpg if anything was added
        if to_renumber:
            renumber_class_folder(target_cid)

    # Cleanup: remove empty subfolders under new-data
    for cdir in list(NEW_DATA_DIR.iterdir()):
        if cdir.is_dir():
            try:
                next(cdir.iterdir())
            except StopIteration:
                cdir.rmdir()

    # Summary
    total_added = sum(added_counts.values())
    print(f"[SUMMARY] Added {total_added} images across {len(added_counts)} classes.")
    for cid in sorted(added_counts.keys()):
        print(f"  - class {cid}: +{added_counts[cid]}")
    if skipped_dupe_same:
        print(f"[INFO] Skipped {skipped_dupe_same} same-class exact duplicates.")
    if cross_class_dupe:
        print(f"[INFO] Logged {cross_class_dupe} cross-class duplicates to {MULTILABEL_OVERRIDES.name}.")
    if created_classes:
        print("[INFO] Auto-created classes:")
        for cid, name in created_classes:
            print(f"  - {cid}: {name}")

def load_multilabel_sha1():
    bad = set()
    if MULTILABEL_OVERRIDES.exists():
        with MULTILABEL_OVERRIDES.open("r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                h = (r.get("sha1") or "").strip()
                if h: bad.add(h)
    return bad

def cmd_build_manifests(args):
    rows = load_existing_classes_csv()
    if not rows:
        print("[ERROR] classes.csv not found. Run init-from-category first.", file=sys.stderr); sys.exit(1)
    if not IMG_DIR.exists():
        print(f"[ERROR] {IMG_DIR} not found.", file=sys.stderr); sys.exit(1)

    random.seed(args.seed)

    # SHA1s to exclude entirely (cross-class dupes etc.)
    exclude_sha1 = load_multilabel_sha1()

    # Gather items: per class list of (path, sha1)
    per_class = defaultdict(list)
    total_images = 0
    for class_dir in IMG_DIR.iterdir():
        if not class_dir.is_dir() or not class_dir.name.isdigit(): continue
        cid = int(class_dir.name)
        for img in class_dir.iterdir():
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS: continue
            try:
                h = sha1_of_file(img)
            except Exception as e:
                print(f"[WARN] Failed to hash {img}: {e}", file=sys.stderr); continue
            if h in exclude_sha1:  # excluded due to multilabel/cross-class duplicate
                continue
            per_class[cid].append((img.as_posix(), h))
            total_images += 1

    # Optional: dedup within a class
    if args.dedup_in_class:
        for cid, lst in per_class.items():
            seen = set(); deduped = []
            for p, h in lst:
                if h in seen: continue
                seen.add(h); deduped.append((p, h))
            per_class[cid] = deduped

    # Stratified split
    val_ratio = max(0.0, min(0.9, args.val))
    test_ratio = max(0.0, min(0.9, args.test))
    assert val_ratio + test_ratio < 1.0 + 1e-9, "val+test must be < 1.0"

    train_rows, val_rows, test_rows = [], [], []
    tiny_classes = []

    for cid, items in sorted(per_class.items()):
        if not items: 
            continue
        items = list(items)
        # deterministic shuffle
        random.Random(args.seed + cid).shuffle(items)

        n = len(items)
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))
        # ensure at least 1 train if possible
        if n - (n_val + n_test) <= 0:
            # reduce val/test to leave 1 for train when feasible
            overflow = (n_val + n_test) - (n - 1)
            while overflow > 0 and (n_val > 0 or n_test > 0):
                if n_val >= n_test and n_val > 0:
                    n_val -= 1
                elif n_test > 0:
                    n_test -= 1
                overflow -= 1
        n_train = n - n_val - n_test
        if n_train <= 0:
            tiny_classes.append((cid, n))
            # put all into train as last resort
            n_train, n_val, n_test = n, 0, 0

        split_train = items[:n_train]
        split_val   = items[n_train:n_train+n_val]
        split_test  = items[n_train+n_val:]

        train_rows += [(p, cid, h) for (p,h) in split_train]
        val_rows   += [(p, cid, h) for (p,h) in split_val]
        test_rows  += [(p, cid, h) for (p,h) in split_test]

    # Write CSVs
    def write_csv(path: Path, rows):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filepath","class_id","sha1"])
            for p, cid, h in rows:
                w.writerow([p, cid, h])

    out_dir = DATA_ROOT
    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "val.csv",   val_rows)
    write_csv(out_dir / "test.csv",  test_rows)

    print(f"[OK] Wrote manifests: train({len(train_rows)}), val({len(val_rows)}), test({len(test_rows)}).")
    if tiny_classes:
        print("[INFO] Tiny classes put fully in train (no val/test):")
        for cid, n in tiny_classes:
            print(f"  - class {cid}: {n} images")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="CAI Vision Dataset label schema, ingestion & manifests tool")
    sp = ap.add_subparsers()

    p1 = sp.add_parser("init-from-category", help="Generate/merge classes.csv from category.txt")
    p1.set_defaults(func=cmd_init_from_category)

    p2 = sp.add_parser("validate", help="Validate classes.csv vs images/; detect cross-labeled dupes")
    p2.set_defaults(func=cmd_validate)

    p3 = sp.add_parser("add-class", help="Append a new class with the next id and create its folder")
    p3.add_argument("name", type=str, help="Display name in English")
    p3.set_defaults(func=cmd_add_class)

    p4 = sp.add_parser("ingest-new-data", help="Move new-data/<class_name>/* into images/<class_id>/* and renumber to 1..N.jpg")
    p4.add_argument("--create-missing", action="store_true", help="Auto-create classes for unknown <class_name> dirs")
    p4.set_defaults(func=cmd_ingest_new_data)

    p5 = sp.add_parser("build-manifests", help="Create train/val/test CSVs from images/, excluding multilabel overrides")
    p5.add_argument("--val", type=float, default=0.10, help="Validation ratio (default: 0.10)")
    p5.add_argument("--test", type=float, default=0.15, help="Test ratio (default: 0.15)")
    p5.add_argument("--seed", type=int, default=42, help="Deterministic split seed (default: 42)")
    p5.add_argument("--dedup-in-class", action="store_true", help="Drop same-SHA1 duplicates within each class")
    p5.set_defaults(func=cmd_build_manifests)


    args = ap.parse_args()
    if not hasattr(args, "func"):
        ap.print_help()
        sys.exit(1)
    args.func(args)

if __name__ == "__main__":
    main()
