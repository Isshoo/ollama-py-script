import pandas as pd
import json
import re
import os
from ollama import chat

# Konfigurasi
INPUT_CLEAN_FILE = "testing/pendidikan_cleaned.xlsx"
OUTPUT_BALANCED_FILE = "testing/pendidikan_cleaned_balanced.xlsx"
CHECKPOINT_FILE = "testing/pendidikan_cleaned_balanced_checkpoint.xlsx" # <--- TAMBAHAN: Nama file checkpoint
TOPIK = "Pendidikan"  # Ubah jadi "Kesehatan" jika memproses dataset kesehatan
MODEL = "qwen2.5:14b"
TARGET_COUNT = 2000
LABELS = ["senang", "percaya", "terkejut", "netral", "takut", "sedih", "marah"]

PROMPT_AUGMENT = """
Anda adalah pakar pembuat data teks (Data Augmentation) Twitter/X Bahasa Indonesia.
Tugas Anda adalah membuat {jumlah_minta} contoh tweet baru yang unik dan natural mengenai topik "Kebijakan Efisiensi Anggaran di Sektor {topik}" yang mengekspresikan emosi "{emosi}" dengan sangat jelas.

Konteks Isu Efisiensi Anggaran:
Tweet harus berpusat pada dampak penghematan, pemotongan dana bantuan/subsidi, pembatasan anggaran riset/fasilitas, penundaan insentif pekerja sektor {topik}, atau keluhan/pujian masyarakat terhadap efisiensi dana tersebut.

Ketentuan Pembuatan Tweet:
1. Gaya bahasa harus santai, kasual, gaya anak sosmed Twitter/X Indonesia jaman sekarang (boleh pakai singkatan wajar seperti 'yg', 'bgt', 'bikin', 'gimana', 'pake', dll).
2. Jangan gunakan kata emosi itu sendiri secara gamblang (Misal: jika emosinya 'sedih', jangan selalu tulis "saya sedih", tapi ekspresikan lewat situasi miris atau kekecewaan terhadap fasilitas {topik} yang dikurangi akibat efisiensi).
3. Tweet harus bervariasi konteksnya (ada yang bersudut pandang masyarakat umum, praktisi/pekerja di bidang {topik}, pelajar/pasien yang terdampak, atau pengamat kebijakan).
4. Satu tweet cukup 1-3 kalimat pendek saja.

Wajib kembalikan hasil dalam format JSON array objek seperti contoh di bawah ini tanpa penjelasan teks lain:
{{
  "tweets": [
    "contoh tweet hasil augmentasi 1",
    "contoh tweet hasil augmentasi 2"
  ]
}}
"""

def generate_synthetic_tweets(emosi, jumlah_minta):
    prompt = PROMPT_AUGMENT.format(jumlah_minta=jumlah_minta, topik=TOPIK, emosi=emosi)
    try:
        response = chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response["message"]["content"]
        
        match = re.search(r"\{.*?\}", content, re.DOTALL)
        if match:
            result = json.loads(match.group())
            list_tweet = result.get("tweets", [])
            return [str(t).strip() for t in list_tweet if t]
    except Exception as e:
        print(f"   [Error LLM saat membuat data emosi {emosi}]: {e}")
    return []

def balance_dataset():
    print(f"=== MEMULAI PROSES BALANCING DATA TOPIK: {TOPIK} ===")
    if not os.path.exists(INPUT_CLEAN_FILE):
        print(f"File data bersih {INPUT_CLEAN_FILE} tidak ditemukan! Jalankan script bersih-bersih dulu.")
        return

    df = pd.read_excel(INPUT_CLEAN_FILE)
    balanced_list = []

    # --- TAMBAHAN: CEK & LOAD CHECKPOINT JIKA ADA ---
    processed_labels = []
    if os.path.exists(CHECKPOINT_FILE):
        print(f"🔄 Menemukan file checkpoint! Memuat progress sebelumnya...")
        df_checkpoint = pd.read_excel(CHECKPOINT_FILE)
        balanced_list.append(df_checkpoint)
        processed_labels = df_checkpoint["fix_label"].unique().tolist()
    # ------------------------------------------------

    for label in LABELS:
        # --- TAMBAHAN: SKIP LABEL YANG SUDAH SELESAI DI CHECKPOINT ---
        if label in processed_labels:
            print(f"\n► Melewati Label: [{label.upper()}] | Sudah selesai diproses di checkpoint sebelumnya.")
            continue
        # -------------------------------------------------------------

        df_label = df[df["fix_label"] == label]
        current_count = len(df_label)
        print(f"\n► Memproses Label: [{label.upper()}] | Jumlah Data Saat Ini: {current_count}")

        if current_count >= TARGET_COUNT:
            # Kasus 1: Data berlebih -> Downsampling acak
            print(f"   Data melimpah. Mengambil {TARGET_COUNT} sampel acak...")
            df_sampled = df_label.sample(n=TARGET_COUNT, random_state=42)
            balanced_list.append(df_sampled)
            print(f"   Label [{label.upper()}] sekarang pas {TARGET_COUNT} baris.")
        
        else:
            # Kasus 2: Data kurang -> Augmentasi menggunakan Qwen
            balanced_list.append(df_label) # Masukkan dulu data asli yang ada
            needed = TARGET_COUNT - current_count
            print(f"   Data kurang! Membutuhkan {needed} data baru via augmentasi Qwen...")
            
            synthetic_tweets = []
            attempts = 0
            
            # Looping sampai jumlah kekurangan terpenuhi
            while len(synthetic_tweets) < needed and attempts < (needed // 2 + 5):
                attempts += 1
                # Minta kelipatan 5 atau sisa data yang dibutuhkan agar tidak overload
                batch_size = min(5, needed - len(synthetic_tweets))
                
                print(f"   -> [Batch {attempts}] Meminta {batch_size} tweet emosi {label}...")
                new_tweets = generate_synthetic_tweets(label, batch_size)
                
                for nt in new_tweets:
                    if nt not in synthetic_tweets: # Hindari duplikasi
                        synthetic_tweets.append(nt)
                
                print(f"   -> Berhasil mengumpulkan total: {len(synthetic_tweets)} / {needed} data buatan.")

            # Potong kelebihan jika hasil batch melebihi 'needed'
            synthetic_tweets = synthetic_tweets[:needed]

            # Ubah data buatan menjadi DataFrame
            synthetic_data = {
                "full_text": synthetic_tweets,
                "old_label": ["synthetic_augment"] * len(synthetic_tweets),
                "fix_label": [label] * len(synthetic_tweets)
            }
            df_synthetic = pd.DataFrame(synthetic_data)
            balanced_list.append(df_synthetic)
            print(f"   Label [{label.upper()}] sekarang berhasil dipenuhi menjadi {TARGET_COUNT} baris.")

        # --- TAMBAHAN: SIMPAN PROGRESS (CHECKPOINT) PER LABEL ---
        print(f"   💾 Menyimpan checkpoint untuk label [{label.upper()}]...")
        df_current_progress = pd.concat(balanced_list, ignore_index=True)
        df_current_progress.to_excel(CHECKPOINT_FILE, index=False)
        # --------------------------------------------------------

    # Gabungkan semua data yang sudah seimbang
    final_balanced_df = pd.concat(balanced_list, ignore_index=True)
    
    # Acak barisnya secara keseluruhan agar distribusinya merata saat dibaca
    final_balanced_df = final_balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Simpan hasil akhir
    final_balanced_df.to_excel(OUTPUT_BALANCED_FILE, index=False)
    
    # --- TAMBAHAN: HAPUS CHECKPOINT JIKA SEMUA PROSES BERHASIL ---
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"\n🗑️  File checkpoint otomatis dihapus karena proses telah selesai sepenuhnya.")
    # -------------------------------------------------------------
    
    print(f"\n{'='*50}")
    print(f"✅ PROSES SELESAI!")
    print(f"Data seimbang disimpan di: {OUTPUT_BALANCED_FILE}")
    print(f"Total baris akhir: {len(final_balanced_df)} baris.")
    print(f"{'='*50}")

if __name__ == "__main__":
    balance_dataset()