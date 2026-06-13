import pandas as pd

file_path = "dataset.xlsx"

for sheet in ["Pendidikan", "Kesehatan"]:

    df = pd.read_excel(file_path, sheet_name=sheet)

    print(f"\n===== {sheet} =====")

    for label in [
        "senang",
        "percaya",
        "terkejut",
        "netral",
        "takut",
        "sedih",
        "marah"
    ]:

        sample = (
            df[df["text_label"] == label]
            .sample(3, random_state=42)
        )

        print(f"\n--- {label.upper()} ---")

        for text in sample["full_text"]:
            print(text)
            print("-" * 50)