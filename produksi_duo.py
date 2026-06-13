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
MODELS = {
    "qwen": "qwen2.5:7b",
    "llama": "llama3.1:8b"
}

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
{
  "relevance": "YA atau TIDAK",
  "emotion_clear": "YA atau TIDAK",
  "label": "salah satu label emosi di atas"
}

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

def analyze_tweet(tweet, model):
    prompt = PROMPT.format(tweet=tweet)
    response = chat(
        model=model,
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

def should_keep(qwen, llama):
    """
    SIMPAN jika minimal salah satu model menilai:
    - relevan dengan topik, ATAU
    - memiliki emosi yang jelas
    BUANG hanya jika KEDUA model setuju data tidak relevan DAN tidak ada emosi.
    """
    qwen_keep = qwen["relevance"] == "YA" or qwen["emotion_clear"] == "YA"
    llama_keep = llama["relevance"] == "YA" or llama["emotion_clear"] == "YA"
    return qwen_keep or llama_keep  # cukup salah satu model bilang simpan

def process_sheet(sheet_name, output_file):
    print(f"\n{'='*50}")
    print(f"Processing sheet: {sheet_name}")
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
            continue  # skip baris yang sudah diproses

        tweet = str(row.get("full_text", "")).strip()
        old_label = row.get("text_label", "")

        if not tweet or tweet == "nan":
            print(f"[{i+1}/{total}] Skipped (empty tweet)")
            continue

        try:
            qwen = analyze_tweet(tweet, MODELS["qwen"])
            llama = analyze_tweet(tweet, MODELS["llama"])

            keep = should_keep(qwen, llama)

            # Tentukan fix_label
            if keep and qwen["label"] and llama["label"] and qwen["label"] == llama["label"]:
                fix_label = qwen["label"]
            else:
                fix_label = ""  # beda prediksi atau data dibuang → kosong dulu

            results.append({
                "full_text": tweet,
                "old_label": old_label,
                "qwen_relevance": qwen["relevance"],
                "qwen_emotion_clear": qwen["emotion_clear"],
                "qwen_pred": qwen["label"] if keep else "",
                "llama_relevance": llama["relevance"],
                "llama_emotion_clear": llama["emotion_clear"],
                "llama_pred": llama["label"] if keep else "",
                "keep": keep,
                "fix_label": fix_label
            })

            # Auto-save setiap 50 baris
            if (i + 1) % 50 == 0:
                pd.DataFrame(results).to_excel(output_file, index=False)
                kept = sum(1 for r in results if r["keep"])
                agreed = sum(1 for r in results if r["fix_label"])
                print(f"[{i+1}/{total}] Saved | Kept: {kept} | Auto-labeled: {agreed}")

        except Exception as e:
            print(f"[ERROR] Row {i}: {e}")
            # Tetap simpan baris error agar tidak hilang dari checkpoint
            results.append({
                "full_text": tweet,
                "old_label": old_label,
                "qwen_relevance": "ERROR",
                "qwen_emotion_clear": "ERROR",
                "qwen_pred": "",
                "llama_relevance": "ERROR",
                "llama_emotion_clear": "ERROR",
                "llama_pred": "",
                "keep": False,
                "fix_label": ""
            })

    # Final save
    final_df = pd.DataFrame(results)
    
    # Untuk output akhir: hanya simpan data yang keep=True
    kept_df = final_df[final_df["keep"] == True].copy()
    kept_df.to_excel(output_file, index=False)

    # Statistik akhir
    total_kept = len(kept_df)
    total_removed = total - total_kept
    auto_labeled = (kept_df["fix_label"] != "").sum()
    need_review = (kept_df["fix_label"] == "").sum()

    print(f"\n{'─'*40}")
    print(f"Sheet: {sheet_name}")
    print(f"Total awal     : {total}")
    print(f"Data disimpan  : {total_kept} ({total_kept/total*100:.1f}%)")
    print(f"Data dibuang   : {total_removed} ({total_removed/total*100:.1f}%)")
    print(f"Auto-labeled   : {auto_labeled} (kedua model setuju)")
    print(f"Perlu review   : {need_review} (prediksi berbeda)")
    print(f"Output         : {output_file}")
    print(f"{'─'*40}")

# Jalankan
process_sheet("Pendidikan", "pendidikan_labeled.xlsx")
process_sheet("Kesehatan", "kesehatan_labeled.xlsx")
print("\n✅ SELESAI")