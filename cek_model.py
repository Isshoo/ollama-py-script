import time
from ollama import chat

tweet = "Beneeeer tidak ada pemotongan KIP dan LPDP. Tapiiii tetep aja dana pendidikan dipotong!! Harusnya biaya sekolah sampek tingkat kuliah itu gratis ini mahal banget. Pengen pinter aja nggk didukung, gimana mau menuju Indonesia emas 2045. MUSTAHIL BANGET!! Paling bener 20450!!"

prompt = f"""
Anda adalah sistem klasifikasi emosi untuk tweet Bahasa Indonesia.

Label yang tersedia:

- senang    : kebahagiaan, kegembiraan, rasa syukur, puas
- percaya   : keyakinan, kepercayaan, optimisme, harapan
- terkejut  : kaget, heran, tidak menyangka
- netral    : tidak menunjukkan emosi yang jelas
- takut     : cemas, khawatir, takut, panik
- sedih     : kecewa, sedih, putus asa, menyesal
- marah     : marah, kesal, geram, jengkel

Tweet:
{tweet}

Aturan:
1. Pilih tepat satu label.
2. Jawab hanya dengan label.
3. Jangan memberi penjelasan.
4. Jangan membuat label baru.

Output:
"""

start = time.time()

response = chat(
    model="qwen2.5:14b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("Waktu:", time.time() - start)
print("Label:", response["message"]["content"].strip())