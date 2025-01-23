import json
import random
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai


# Daftar contoh nama produk
nama_produk_list = [
    "Kursi Ergonomis", "Meja Kantor", "Lampu Meja", "Rak Buku", "Sofa Minimalis",
    "Kasur Lipat", "Lemari Pakaian", "Meja Belajar", "Karpet Ruang Tamu", "Kursi Goyang",
    "Meja Rapat", "Cermin Dekoratif", "Lampu Gantung", "Gorden Polos", "Bantal Sofa",
    "Selimut Hangat", "Jam Dinding", "Vas Bunga", "Meja TV", "Bean Bag",
    "Kursi Gaming", "Rak Sepatu", "Hiasan Dinding", "Lemari Dapur", "Tempat Tidur",
    "Kursi Bar", "Meja Makan", "Rak Sudut", "Piring Keramik", "Gelas Kaca",
    "Kompor Gas", "Dispenser Air", "Setrika Listrik", "Penghisap Debu", "AC Portable",
    "Kipas Angin", "Penyaring Udara", "Mesin Cuci", "Kulit Asli Sofa", "Panci Anti Lengket",
    "Blender Multifungsi", "Rice Cooker", "Penggorengan", "Microwave Oven", "Vacuum Cleaner",
    "Water Heater", "Lemari Es", "Lampu LED", "Meja Lipat", "Kasur Busa"
]

# Daftar contoh kategori
kategori_list = ["Furniture", "Elektronik", "Dekorasi", "Peralatan Dapur", "Perlengkapan Rumah"]

# Fungsi untuk membuat deskripsi produk
def generate_description(nama):
    return f"{nama} berkualitas tinggi dengan desain modern dan harga terjangkau."

# Membuat array produk
products = []
for i in range(1, 101):  # Loop untuk membuat 100 produk
    nama_produk = random.choice(nama_produk_list)
    kategori = random.choice(kategori_list)
    harga = random.randint(50000, 2000000)  # Harga antara 50 ribu - 2 juta
    produk = {
        "id": i,
        "nama_produk": nama_produk,
        "deskripsi": generate_description(nama_produk),
        "kategori": kategori,
        "harga": harga
    }
    products.append(produk)

# Simpan hasil ke dalam array JSON
products_json = json.dumps(products, indent=4)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Muat model USE
# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

product_names = [product["nama_produk"] for product in products]

# Menghitung embedding untuk setiap teks produk
embeddings = model.encode(product_names, convert_to_tensor=True)
# Menghitung cosine similarity antar embeddings
cosine_sim = cosine_similarity(embeddings)

def get_closest_product(prompt):
  # hitung embeding 
  input_embeding = model.encode(prompt, convert_to_tensor=True)

  # hitung persamaan antara text dengan data
  cosine_sim_input = util.pytorch_cos_sim(input_embeding, embeddings)
  # Ambil data yg memiliki similarity tertinggi dari daftar product
  max_sim_idx = torch.argmax(cosine_sim_input).item()
  max_sim_value = cosine_sim_input[0, max_sim_idx].item()
  relevant_product = products[max_sim_idx]

  return relevant_product

# integrasikan hasil cosine similarity dengan model AI
# disini sayah menggunakan model AI flash dari gemini
# sebelum dikirimkan ke model AI yg bertugas untuk menjelaskan produk
# atau yang membuatkan orderan, input user dicek dlu oleh model AI khusus

KEY_API_GEMINI="YOUR_GEMINI_KEY"

genai.configure(api_key=KEY_API_GEMINI)
model_ai = genai.GenerativeModel("gemini-1.5-flash")

# Tambahkan system prompt
clasification_prompt = (
    ''' 
      Kamu adalah AI assistant yang bekerja untuk toko serba ada.
      tugas kamu adalah:
      1. mengklasifikasi permintaan user, jika user bertanya tentang produk maka alihkan ke "agent_product"
      2. mengklasifikasi permintaan user, jika user meminta dibuatkan orderan maka alihkan ke "agent_taking_order"
      3. Jika user hanya menyapa atau mengucapkan salam, langsung arahkan ke "agent_product".


      jika pertanyaan diluar dari konteks maka jawab dengan "Saya tidak bisa membantu dengan itu. apakah ada orderan yg ingin saya buatkan?".
      cukup jawab dengan "agent_product" atau "agent_taking_order"
    '''
)

def clasification(prompt):
  # Gabungkan system prompt dan user input
  final_prompt = clasification_prompt + "\nUser input: " + prompt
  # Kirim permintaan ke model_ai
  response = model_ai.generate_content(final_prompt)
  return response.text


history = []

def agent_products(prompt):
  # Simpan percakapan sebelumnya
  history.append(f"User: {prompt}")

  if len(history) > 5:
     history.pop(0)

  # Gabungkan history percakapan dengan prompt baru
  history_context = "\n".join(history)  # Gabungkan seluruh history menjadi satu string
    
  # Mendapatkan produk terdekat
  closest_product = get_closest_product(prompt)
    
  # System prompt untuk agent produk
  agent_product_prompt = f"""
        Kamu adalah AI assistant yang bekerja sebagai customer service di toko serba ada.
        Tugas kamu adalah:
        1. Menjelaskan produk yang ditanyakan oleh user berdasarkan data yang ada.
        2. Gunakan data berikut untuk menjawab pertanyaan:
        
        Produk: {{
            "nama_produk": "{closest_product['nama_produk']}",
            "deskripsi": "{closest_product['deskripsi']}",
            "kategori": "{closest_product['kategori']}",
            "harga": {closest_product['harga']}
        }}

        Pastikan jawabanmu jelas, singkat, dan membantu.
    """

  # Gabungkan system prompt, history percakapan, dan user input
  final_prompt = agent_product_prompt + "\nRiwayat percakapan: \n" + history_context + "\nUser input: " + prompt
    
  # Kirim permintaan ke model AI
  response = model_ai.generate_content(final_prompt)
    
  # Simpan respons dalam history
  history.append(f"AI: {response.text}")
    
  return response.text


def agent_taking_order(prompt):
    # Dapatkan produk yang paling relevan berdasarkan similarity
    closest_product = get_closest_product(prompt)

    # Periksa apakah produk ditemukan
    if not closest_product:
        return "Maaf, saya tidak dapat menemukan produk yang relevan dengan permintaan Anda."

    # Buat system prompt untuk agent order
    agent_order_prompt = (
        f"""
        Kamu adalah AI assistant yang bekerja sebagai customer service di toko serba ada.
        Tugas kamu adalah:
        1. Membuatkan orderan berdasarkan produk yang diminta user.
        2. Gunakan data berikut untuk membuat order:
        
        Produk:
        {{
            "id": {closest_product['id']},
            "nama_produk": "{closest_product['nama_produk']}",
            "deskripsi": "{closest_product['deskripsi']}",
            "kategori": "{closest_product['kategori']}",
            "harga": {closest_product['harga']}
        }}

        Jawab dengan format JSON berikut (ganti "qty" dengan jumlah sesuai permintaan user):

        {{
            "order": {{
                "product_id": {closest_product['id']},
                "qty": <jumlah>
            }},
            "message": "<jumlah> {closest_product['nama_produk']} berhasil ditambahkan ke keranjang"
        }}
        """
    )

    # Gabungkan system prompt dan user input
    final_prompt = agent_order_prompt + "\nUser input: " + prompt

    # Kirim permintaan ke model
    response = model_ai.generate_content(final_prompt)
    # Bersihkan backticks dan bagian yang tidak perlu
    cleaned_response = response.text.split("```json")[1].split("```")[0].strip()
    json_response = json.loads(cleaned_response)
    
    return json_response

def chat_customer_service(prompt):
  clasification_purpose = clasification(prompt)
  if clasification_purpose.strip() == 'agent_product':
    return agent_products(prompt)
  elif clasification_purpose.strip() == 'agent_taking_order':
    return agent_taking_order(prompt)['message']

  return clasification_purpose
  
# Fungsi utama untuk berinteraksi dengan terminal
def main():
    print("Selamat datang di Customer Service!")
    print("Ketikkan pertanyaan atau perintah Anda (ketik 'keluar' untuk keluar):")

    while True:
        # Memunculkan input di terminal
        user_input = input("Anda: ")

        # Berhenti jika pengguna mengetik 'keluar'
        if user_input.lower() == 'keluar':
            print("Terima kasih telah menggunakan Customer Service. Sampai jumpa!")
            break

        # Memanggil fungsi chat_customer_service dan mencetak responsnya
        response = chat_customer_service(user_input)
        print("AI:", response)


# Jalankan program utama
if __name__ == "__main__":
    main()