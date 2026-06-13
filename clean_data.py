import pandas as pd

file_path = "testing/kesehatan_labeled.xlsx"

df = pd.read_excel(file_path)

print("=" * 50)
print("STATISTIK DATASET SEBELUM CLEANING")
print("=" * 50)

print(f"Jumlah data           : {len(df):,}")
print(f"Label kosong          : {df['fix_label'].isna().sum():,}")
print(f"Tweet kosong          : {df['full_text'].isna().sum():,}")
print(f"Tweet duplikat        : {df['full_text'].duplicated().sum():,}")

print("\nDistribusi Label:")
print("-" * 50)
print(df["fix_label"].value_counts(dropna=False).to_string())

# Cleaning
df = df.dropna(subset=["full_text", "fix_label"])
df = df.drop_duplicates(subset=["full_text"])

# Simpan
df.to_excel(file_path, index=False)

print("\n")
print("=" * 50)
print("STATISTIK DATASET SETELAH CLEANING")
print("=" * 50)

print(f"Jumlah data           : {len(df):,}")
print(f"Label kosong          : {df['fix_label'].isna().sum():,}")
print(f"Tweet kosong          : {df['full_text'].isna().sum():,}")
print(f"Tweet duplikat        : {df['full_text'].duplicated().sum():,}")

print("\nDistribusi Label:")
print("-" * 50)
print(df["fix_label"].value_counts(dropna=False).to_string())