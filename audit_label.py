import pandas as pd
from ollama import chat

FILE_PATH = "dataset.xlsx"

LABELS = [
    "senang",
    "percaya",
    "terkejut",
    "netral",
    "takut",
    "sedih",
    "marah"
]

MODELS = {
    "llama": "llama3.1:8b",
    "qwen": "qwen2.5:7b"
}

PROMPT_TEMPLATE = """
Anda adalah ahli klasifikasi emosi Bahasa Indonesia.

Pilih SATU label berikut:

- senang
- percaya
- terkejut
- netral
- takut
- sedih
- marah


Aturan:
- Jawab hanya satu label.
- Huruf kecil.
- Jangan menjelaskan.

Tweet:
{tweet}

Label:
"""

def predict_label(tweet, model_name):

    prompt = PROMPT_TEMPLATE.format(tweet=tweet)

    response = chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    prediction = (
        response["message"]["content"]
        .strip()
        .lower()
    )

    if prediction not in LABELS:
        prediction = "invalid"

    return prediction


results = []

for sheet in ["Pendidikan", "Kesehatan"]:

    print(f"\nProcessing {sheet}")

    df = pd.read_excel(FILE_PATH, sheet_name=sheet)

    df = df[df["text_label"].notna()]

    sample = df.sample(
        n=min(100, len(df)),
        random_state=42
    )

    for idx, row in sample.iterrows():

        tweet = str(row["full_text"])
        old_label = str(row["text_label"]).strip().lower()

        try:

            llama_pred = predict_label(
                tweet,
                MODELS["llama"]
            )

            qwen_pred = predict_label(
                tweet,
                MODELS["qwen"]
            )

        except Exception as e:

            print(e)
            continue

        results.append({
            "sheet": sheet,
            "index": idx,
            "full_text": tweet,
            "label_lama": old_label,

            "llama_pred": llama_pred,
            "qwen_pred": qwen_pred,

            "old_vs_llama": old_label == llama_pred,
            "old_vs_qwen": old_label == qwen_pred,

            "llama_vs_qwen": llama_pred == qwen_pred
        })

        print(
            f"{sheet} | "
            f"old={old_label} | "
            f"llama={llama_pred} | "
            f"qwen={qwen_pred}"
        )

result_df = pd.DataFrame(results)

result_df.to_excel(
    "audit_3way.xlsx",
    index=False
)

print("\n=== AGREEMENT ===")

old_llama = result_df["old_vs_llama"].mean() * 100
old_qwen = result_df["old_vs_qwen"].mean() * 100
llama_qwen = result_df["llama_vs_qwen"].mean() * 100

print(f"Old vs Llama : {old_llama:.2f}%")
print(f"Old vs Qwen  : {old_qwen:.2f}%")
print(f"Llama vs Qwen: {llama_qwen:.2f}%")

review_df = result_df[
    (result_df["old_vs_llama"] == False)
    &
    (result_df["old_vs_qwen"] == False)
    &
    (result_df["llama_vs_qwen"] == True)
]

review_df.to_excel(
    "high_confidence_suspect.xlsx",
    index=False
)