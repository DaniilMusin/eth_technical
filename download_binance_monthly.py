#!/usr/bin/env python3
# simple_binance_downloader.py
# ───────────────────────────
# Просто нажмите ▶ Run. Никаких аргументов.

# ─── НАСТРОЙКИ ──────────────────────────────────────────────────────────
SYMBOL      = "ETHUSDT"   # пара
INTERVAL    = "15m"       # тайм-фрейм
START_YEAR  = 2018        # первый год включительно
END_YEAR    = 2025        # последний год включительно
OUT_DIR     = "binance_data"  # куда класть ZIP и CSV
# ────────────────────────────────────────────────────────────────────────

import os, sys, time, zipfile, io, subprocess, importlib
from typing import Optional, List

def _ensure(pkg: str):
    try:
        return importlib.import_module(pkg)
    except ImportError:
        print(f"[INFO] Устанавливаю {pkg} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(pkg)

requests = _ensure("requests")
pd       = _ensure("pandas")

def download_zip(symbol: str, tf: str, y: int, m: int, folder: str, tries=3) -> Optional[str]:
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{tf}/{symbol}-{tf}-{y}-{m:02d}.zip"
    os.makedirs(folder, exist_ok=True)
    path = f"{folder}/{symbol}-{tf}-{y}-{m:02d}.zip"
    if os.path.exists(path):
        print(f"[SKIP] {path}")
        return path
    for a in range(1, tries + 1):
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                if r.status_code == 404:
                    print(f"[MISS] {url}")
                    return None
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
            print(f"[ OK ] {path}")
            return path
        except Exception as e:
            print(f"[ERR] попытка {a}/{tries}: {e}")
            time.sleep(2)
    print(f"[FAIL] {url}")
    return None

def load_csv(zip_path: str):
    with zipfile.ZipFile(zip_path) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(
                f, header=None,
                names=["open_time","open","high","low","close","volume",
                       "close_time","quote_vol","trades","tb_base","tb_quote","ignore"]
            )

def main():
    all_zips: List[str] = []
    for y in range(START_YEAR, END_YEAR + 1):
        for m in range(1, 13):
            zp = download_zip(SYMBOL, INTERVAL, y, m, OUT_DIR)
            if zp:
                all_zips.append(zp)

    if not all_zips:
        print("[ABORT] Нет данных — проверьте годы/символ.")
        return

    frames = [load_csv(z) for z in all_zips]
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates("open_time")
            .sort_values("open_time"))

    out_csv = f"{OUT_DIR}/{SYMBOL}-{INTERVAL}-{START_YEAR}-{END_YEAR}.csv.gz"
    df.to_csv(out_csv, index=False, compression="gzip")
    print(f"\n[DONE] {out_csv} готов, строк: {len(df):,}")

if __name__ == "__main__":
    main()
