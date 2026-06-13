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
Analisis tweet berikut.

Aturan RELEVANSI:
YA jika terkait: pendidikan, kesehatan, efisiensi anggaran, kebijakan efisiensi anggaran, dana pendidikan, dana kesehatan,
pemotongan dana pendidikan, pemotongan dana kesehatan, efisiensi anggaran pemerintah, efisiensi anggaran pendidikan, efisiensi anggaran kesehatan.
TIDAK jika: spam, iklan, percakapan acak, topik tidak berkaitan.

Aturan EMOSI_JELAS:
YA jika tweet mengandung emosi yang dapat dikenali.
TIDAK jika: hanya informasi, hanya link, terlalu pendek, tidak ada emosi jelas.

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
  "relevance": "YA atau TIDAK",
  "emotion_clear": "YA atau TIDAK",
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
    return {"relevance": "TIDAK", "emotion_clear": "TIDAK", "label": ""}

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
    return {
        "relevance": str(result.get("relevance", "TIDAK")).upper().strip(),
        "emotion_clear": str(result.get("emotion_clear", "TIDAK")).upper().strip(),
        "label": label
    }

def should_keep(result):
    """
    SIMPAN jika model menilai relevan ATAU memiliki emosi yang jelas.
    BUANG jika tidak relevan DAN tidak ada emosi jelas.
    """
    return result["relevance"] == "YA" or result["emotion_clear"] == "YA"

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
            pred = analyze_tweet(tweet)
            keep = should_keep(pred)

            results.append({
                "full_text": tweet,
                "old_label": old_label,
                "relevance": pred["relevance"],
                "emotion_clear": pred["emotion_clear"],
                "keep": keep,
                "fix_label": pred["label"] if keep else "",
            })
            
            print(f"[{i+1}/{total}] Processed | Relevance: {pred['relevance']} | Emotion Clear: {pred['emotion_clear']} | Label: {pred['label']} | Keep: {keep}")

            # Auto-save setiap 50 baris
            if (i + 1) % 50 == 0:
                pd.DataFrame(results).to_excel(output_file, index=False)
                kept = sum(1 for r in results if r["keep"])
                labeled = sum(1 for r in results if r["fix_label"])
                print(f"[{i+1}/{total}] Saved | Kept: {kept} | Labeled: {labeled}")

        except Exception as e:
            print(f"[ERROR] Row {i}: {e}")
            results.append({
                "full_text": tweet,
                "old_label": old_label,
                "relevance": "ERROR",
                "emotion_clear": "ERROR",
                "keep": False,
                "fix_label": "",
            })

    # Final save — hanya baris keep=True
    final_df = pd.DataFrame(results)
    kept_df = final_df[final_df["keep"] == True].copy()
    kept_df.to_excel(output_file, index=False)

    # Statistik akhir
    total_kept = len(kept_df)
    total_removed = total - total_kept
    labeled = (kept_df["fix_label"] != "").sum()
    empty_label = (kept_df["fix_label"] == "").sum()

    print(f"\n{'─'*40}")
    print(f"Sheet: {sheet_name}")
    print(f"Total awal     : {total}")
    print(f"Data disimpan  : {total_kept} ({total_kept/total*100:.1f}%)")
    print(f"Data dibuang   : {total_removed} ({total_removed/total*100:.1f}%)")
    print(f"Labeled        : {labeled}")
    print(f"Label kosong   : {empty_label} (model tidak mengembalikan label valid)")
    print(f"Output         : {output_file}")
    print(f"{'─'*40}")

# Jalankan
process_sheet("Pendidikan", "output_solo/pendidikan_labeled.xlsx")
process_sheet("Kesehatan", "output_solo/kesehatan_labeled.xlsx")
print("\n✅ SELESAI")