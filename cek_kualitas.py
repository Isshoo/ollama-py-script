import pandas as pd

file_path = "testing/pendidikan_balanced.xlsx"

xls = pd.ExcelFile(file_path)

for sheet in xls.sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet)

    print(f"\n=== {sheet} ===")
    print("Jumlah data:", len(df))

    print("\nKolom:")
    print(df.columns.tolist())

    print("\nJumlah label kosong:")
    print(df["fix_label"].isna().sum())

    print("\nDistribusi label:")
    print(df["fix_label"].value_counts(dropna=False))
    
    print("\nTweet kosong:")
    print(df["full_text"].isna().sum())

    print("\nTweet duplikat:")
    print(df["full_text"].duplicated().sum())


    
