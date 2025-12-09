import streamlit as st
import pickle, os, numpy as np
from sklearn.preprocessing import normalize
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("HijrahFood Recommender (Prototype)")
ART_PATH = '/content/artifacts'

# Load embeddings dan tokenizer
with open(os.path.join(ART_PATH, 'sku_embeddings.pkl'), 'rb') as f:
    data = pickle.load(f)
    sku_list = data['sku_list']
    product_embeddings = data['embeddings']

product_embeddings_norm = normalize(product_embeddings, axis=1)

with open(os.path.join(ART_PATH, 'tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)

q = st.text_input('Masukkan masakan atau produk (contoh: rendang / ayam boneless dada)')
k = st.slider('Jumlah rekomendasi', 1, 10, 5)

if st.button('Cari'):
    if not q:
        st.warning('Masukkan query teks dulu.')
    else:
        seq = tokenizer.texts_to_sequences([q])
        pad = pad_sequences(seq, maxlen=28, padding='post')
        # Placeholder: untuk produksi sebaiknya load model dan hitung cosine similarity.
        # Saat ini tampilkan sku teratas sebagai demo.
        st.write('Hasil (placeholder):')
        for i in range(min(k, len(sku_list))):
            st.write(f"{sku_list[i]}")
