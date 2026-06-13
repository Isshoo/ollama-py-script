import pandas as pd
import json
import re
import os
from ollama import chat

FILE_PATH = "dataset.xlsx"
LABELS = [
    "senang", "percaya", "terkejut",
    "netral", "takut", "sedih", "marah"
]
MODEL = "qwen2.5:7b"

PROMPT = """
Anda adalah ahli analisis tweet Bahasa Indonesia.
Tentukan label emosi dominan dari tweet berikut.

Label emosi yang tersedia:
senang, percaya, terkejut, netral, takut, sedih, marah

Beberapa contoh definisi label emosi (bukan aturan baku, hanya panduan):
senang = bahagia, puas, bersyukur, gembira
percaya = yakin, optimis, percaya, mendukung, harapan
terkejut = kaget, heran, tidak menyangka
netral = informatif, tidak menunjukkan emosi kuat atau jelas
takut = cemas, khawatir, takut, waswas, panik
sedih = kecewa, sedih, putus asa, miris, menyesal
marah = kesal, geram, marah, jengkel, kesal

Jawab HANYA format JSON (tanpa penjelasan apapun):
{{
  "label": "salah satu label emosi di atas"
}}

Tweet:
{tweet}
"""

def extract_json(text):
    try:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {"label": ""}

def analyze_tweet(tweet):
    prompt = PROMPT.format(tweet=tweet)
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    result = extract_json(response["message"]["content"])
    label = str(result.get("label", "")).lower().strip()
    
    if label not in LABELS:
        label = ""
        
    return label

def process_sheet(sheet_name, output_file):
    print(f"\n{'='*50}")
    print(f"Processing sheet : {sheet_name}")
    print(f"Model            : {MODEL}")
    print(f"{'='*50}")

    df = pd.read_excel(FILE_PATH, sheet_name=sheet_name)
    total = len(df)
    print(f"Total data: {total} baris")

    # --- CHECKPOINT: lanjut dari baris terakhir jika file sudah ada ---
    start_index = 0
    existing_results = []
    if os.path.exists(output_file):
        existing_df = pd.read_excel(output_file)
        existing_results = existing_df.to_dict("records")
        start_index = len(existing_results)
        print(f"Checkpoint ditemukan! Melanjutkan dari baris {start_index}...")

    results = existing_results.copy()

    for i, row in df.iterrows():
        if i < start_index:
            continue

        tweet = str(row.get("full_text", "")).strip()
        old_label = row.get("text_label", "")

        if not tweet or tweet == "nan":
            print(f"[{i+1}/{total}] Skipped (empty tweet)")
            continue

        try:
            label = analyze_tweet(tweet)

            results.append({
                "full_text": tweet,
                "old_label": old_label,
                "fix_label": label,
            })
            
            print(f"[{i+1}/{total}] Processed | Label: {label if label else 'TIDAK VALID'}")

            # Auto-save setiap 50 baris
            if (i + 1) % 50 == 0:
                pd.DataFrame(results).to_excel(output_file, index=False)
                labeled = sum(1 for r in results if r["fix_label"])
                print(f"[{i+1}/{total}] Saved | Labeled: {labeled}")

        except Exception as e:
            print(f"[ERROR] Row {i}: {e}")
            results.append({
                "full_text": tweet,
                "old_label": old_label,
                "fix_label": "",
            })

    # Final save — simpan semua baris
    final_df = pd.DataFrame(results)
    final_df.to_excel(output_file, index=False)

    # Statistik akhir
    total_processed = len(final_df)
    labeled = (final_df["fix_label"] != "").sum()
    empty_label = (final_df["fix_label"] == "").sum()

    print(f"\n{'─'*40}")
    print(f"Sheet: {sheet_name}")
    print(f"Total baris    : {total_processed}")
    print(f"Labeled        : {labeled} ({labeled/total_processed*100:.1f}%)")
    print(f"Label kosong   : {empty_label} (model tidak mengembalikan label valid)")
    print(f"Output         : {output_file}")
    print(f"{'─'*40}")

# Buat folder output jika belum ada
os.makedirs("output_solo", exist_ok=True)

# Jalankan
process_sheet("Pendidikan", "output_solo/pendidikan_labeled.xlsx")
process_sheet("Kesehatan", "output_solo/kesehatan_labeled.xlsx")
print("\n✅ SELESAI")