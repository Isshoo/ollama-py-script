import pandas as pd
import json
import re
import os
from ollama import chat

# Konfigurasi
INPUT_FILE = "testing/kesehatan_balanced.xlsx"
OUTPUT_CLEAN_FILE = "testing/kesehatan_cleaned.xlsx"
TOPIK = "Kesehatan"  # Ubah jadi "Kesehatan" jika memproses dataset kesehatan
MODEL = "qwen2.5:14b"

PROMPT_CLEAN = """
Anda adalah pakar kurasi data teks Twitter/X Bahasa Indonesia untuk topik "Kebijakan Efisiensi Anggaran di Sektor {topik}".
Tugas Anda adalah menilai apakah tweet berikut LAYAK dipertahankan untuk analisis emosi.

Kriteria LAYAK (true):
1. Memiliki makna atau emosi yang jelas (termasuk emosi netral yang bersifat informatif/berita).

Kriteria TIDAK LAYAK (false):
2. Hanya berisi spam link, kumpulan hashtag promosi tanpa teks pembentuk kalimat, atau karakter acak/rusak.
3. Kata-kata terlalu pendek, kata-kata acak tanpa konteks emosi yang jelas.

Tweet: "{tweet}"
Label Emosi Saat Ini: {label}

Jawab HANYA dengan format JSON berikut (tanpa penjelasan tambahan):
{{
  "layak_disimpan": true_atau_false
}}
"""

def check_relevance(tweet, label):
    prompt = PROMPT_CLEAN.format(topik=TOPIK, tweet=tweet, label=label)
    try:
        response = chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response["message"]["content"]
        
        # Ekstrak JSON
        match = re.search(r"\{.*?\}", content, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return bool(result.get("layak_disimpan", True))
    except Exception as e:
        print(f" Err LLM: {e}")
    return True # Jika error, amankan data dulu (anggap layak)

def clean_dataset():
    print(f"=== MEMULAI PEMBERSIHAN DATA TOPIK: {TOPIK} ===")
    
    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} tidak ditemukan!")
        return

    df = pd.read_excel(INPUT_FILE)
    total = len(df)
    
    # Checkpoint
    start_index = 0
    existing_results = []
    if os.path.exists(OUTPUT_CLEAN_FILE):
        existing_df = pd.read_excel(OUTPUT_CLEAN_FILE)
        existing_results = existing_df.to_dict("records")
        start_index = len(existing_results)
        print(f"Checkpoint ditemukan! Melanjutkan dari baris {start_index}...")

    results = existing_results.copy()

    for i, row in df.iterrows():
        if i < start_index:
            continue

        tweet = str(row.get("full_text", "")).strip()
        label = str(row.get("fix_label", "")).strip()

        # Jika dari awal label kosong atau tweet kosong, langsung skip
        if not tweet or tweet == "nan" or not label:
            print(f"[{i+1}/{total}] Discarded (Empty text/label)")
            continue

        # Tanya Qwen apakah layak
        is_layak = check_relevance(tweet, label)

        if is_layak:
            results.append(row.to_dict())
            print(f"[{i+1}/{total}] KEEP    | Label: {label}")
        else:
            print(f"[{i+1}/{total}] DISCARD | Tweet dinilai tidak relevan/spam")

        # Auto-save setiap 20 baris
        if (i + 1) % 20 == 0:
            pd.DataFrame(results).to_excel(OUTPUT_CLEAN_FILE, index=False)
            print(f">> Berhasil menyimpan checkpoint sementara. Total bersih saat ini: {len(results)}")

    # Final Save
    final_df = pd.DataFrame(results)
    final_df.to_excel(OUTPUT_CLEAN_FILE, index=False)
    print(f"\n✅ Selesai! Data bersih disimpan di: {OUTPUT_CLEAN_FILE}")
    print(f"Jumlah data sebelum dibersihkan: {total}")
    print(f"Jumlah data setelah dibersihkan: {len(final_df)}")

if __name__ == "__main__":
    clean_dataset()