"""Load the Memotion CSVs, detect columns, match images, write processed tables."""
import glob, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

IMG_PREF = ["image_name", "image", "img", "file_name", "filename", "image_url"]
LABEL_PREF = ["overall_sentiment", "overall", "sentiment", "label"]
ID_PREF = ["image_id", "id", "index", "unnamed: 0"]


def find_csv(split):
    hits = glob.glob(os.path.join(config.CSV_DIR, "**", f"{split}.csv"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{split}.csv not found under {config.CSV_DIR}")
    return hits[0]


def read_csv(path):
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def pick(cols, preferred, contains=()):
    low = {c.lower().strip(): c for c in cols}
    for p in preferred:
        if p in low:
            return low[p]
    for lc, c in low.items():
        if any(k in lc for k in contains):
            return c
    return None


def map_label(v):
    """Map a raw sentiment value to 0/1/2 (-1 if missing or unknown)."""
    if pd.isna(v):
        return -1
    s = str(v).strip().lower()
    try:
        return config.NUMERIC_MAP.get(int(float(s)), -1)
    except ValueError:
        pass
    if "neg" in s:
        return 0
    if "neu" in s:
        return 1
    if "pos" in s:
        return 2
    return -1


def build_split(split, verbose=True, image_split=None):
    df = read_csv(find_csv(split))
    img_col = pick(df.columns, IMG_PREF, ("image", "img"))
    lab_col = pick(df.columns, LABEL_PREF, ("sentiment",))
    id_col = pick(df.columns, ID_PREF, ("index",))
    corr = pick(df.columns, ["corrected_text", "text_corrected"], ("corrected",))
    ocr = pick(df.columns, ["ocr_extracted_text", "ocr_text", "text"], ("ocr", "text"))
    if img_col is None or (corr is None and ocr is None):
        raise ValueError(f"[{split}] could not detect image/text columns in {df.columns.tolist()}")
    text = df[corr] if corr else df[ocr]
    if corr and ocr and corr != ocr:
        text = text.fillna(df[ocr])
    image_names = df[img_col].fillna("").astype(str).str.strip()
    # Memotion 3 tables may contain image URLs rather than filenames. The
    # supplied CSV row index matches the released <index>.jpg files.
    if id_col and image_names.str.contains(r"://", regex=True).any():
        def index_name(value):
            if pd.isna(value):
                return ""
            try:
                number = float(value)
                return str(int(number)) if number.is_integer() else str(value).strip()
            except (TypeError, ValueError):
                return str(value).strip()
        image_names = df[id_col].map(index_name)
    out = pd.DataFrame({"image_name": image_names, "text": text.fillna("").astype(str)})
    out["label"] = df[lab_col].map(map_label) if lab_col else -1
    if lab_col and verbose and pd.api.types.is_numeric_dtype(df[lab_col]):
        print(f"[{split}] WARNING: sentiment is numeric; check config.NUMERIC_MAP against the docs.")

    image_split = image_split or split

    def resolve(name):
        if not name:
            return None
        cands = [name] if os.path.splitext(name)[1] else [name + ".jpg", name + ".png", name + ".jpeg"]
        for c in cands:
            p = os.path.join(config.IMG_DIRS[image_split], os.path.basename(c))
            if os.path.exists(p):
                return p
        return None

    out["image_path"] = out["image_name"].map(resolve)
    missing = int(out["image_path"].isna().sum())
    out = out.dropna(subset=["image_path"]).reset_index(drop=True)
    out["label_name"] = out["label"].map(lambda i: config.LABELS[i] if i >= 0 else "")
    if verbose:
        print(f"[{split}] cols: image={img_col} id={id_col} text={corr or ocr} label={lab_col} | rows={len(out)} missing_images={missing}")
        if lab_col:
            print(out["label_name"].value_counts().to_string())
    return out[["image_path", "text", "label", "label_name"]]
