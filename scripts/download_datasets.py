"""
数据集下载脚本
用法:
  python scripts/download_datasets.py --dataset all
  python scripts/download_datasets.py --dataset goodbooks
  python scripts/download_datasets.py --dataset amazon_movies

数据源:
  - ML-1M: 已内置在 data/raw/ml1m/，无需下载
  - Goodbooks-10k: GitHub (72 MB)
  - Amazon Movies: Stanford SNAP (~179 MB)
"""

import sys, os, argparse, urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

RAW_DIR = "data/raw"


def download_file(url, dest, desc="downloading"):
    print(f"  [{desc}] {os.path.basename(dest)} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024*1024)
        print(f"    Done: {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        return False


def download_goodbooks():
    dest = os.path.join(RAW_DIR, "goodbooks")
    os.makedirs(dest, exist_ok=True)

    files = {
        "ratings.csv": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv",
        "books.csv": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv",
        "book_tags.csv": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/book_tags.csv",
        "tags.csv": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/tags.csv",
    }
    for fname, url in files.items():
        out = os.path.join(dest, fname)
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            print(f"  SKIP {fname}: {os.path.getsize(out)} bytes")
            continue
        download_file(url, out, "Goodbooks")
    return True


def download_amazon():
    dest = os.path.join(RAW_DIR, "amazon_movies")
    os.makedirs(dest, exist_ok=True)

    url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Movies_and_TV.csv"
    out = os.path.join(dest, "ratings_Movies_and_TV.csv")
    if os.path.exists(out) and os.path.getsize(out) > 1000000:
        print(f"  SKIP: {os.path.getsize(out)/(1024*1024):.1f} MB already downloaded")
        return True
    return download_file(url, out, "Amazon Movies (~179MB)")


def download_book_crossing():
    """Book-Crossing dataset (Kaggle) — 手动下载说明"""
    print("  Book-Crossing 需要从 Kaggle 下载:")
    print("    1. pip install kagglehub")
    print("    2. python -c \"import kagglehub; kagglehub.dataset_download('somnambwl/bookcrossing-dataset')\"")
    print("    3. 将 Users.csv, Books.csv, Ratings.csv 复制到 data/raw/book_crossing/")
    print("  或直接运行: python scripts/download_datasets.py --dataset book_crossing")
    dest = os.path.join(RAW_DIR, "book_crossing")
    os.makedirs(dest, exist_ok=True)
    try:
        import kagglehub
        path = kagglehub.dataset_download("somnambwl/bookcrossing-dataset")
        import shutil
        for f in ["Users.csv", "Books.csv", "Ratings.csv"]:
            src = os.path.join(path, f)
            dst = os.path.join(dest, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            print(f"  {f}: {os.path.getsize(dst)} bytes")
        return True
    except ImportError:
        print("  [SKIP] kagglehub not installed. Run: pip install kagglehub")
        return False
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def main():
    p = argparse.ArgumentParser(description="Download datasets")
    p.add_argument("--dataset", type=str, default="all",
                   choices=["all", "goodbooks", "amazon_movies", "book_crossing"])
    args = p.parse_args()

    print("=" * 50)
    print("Dataset Download")
    print("=" * 50)
    print(f"  ML-1M: already in data/raw/ml1m/ (included)\n")

    if args.dataset in ("all", "goodbooks"):
        print("[Goodbooks-10k]")
        download_goodbooks()

    if args.dataset in ("all", "amazon_movies"):
        print("\n[Amazon Movies & TV]")
        download_amazon()

    if args.dataset in ("all", "book_crossing"):
        print("\n[Book-Crossing]")
        download_book_crossing()

    print(f"\nDone.")


if __name__ == "__main__":
    main()
