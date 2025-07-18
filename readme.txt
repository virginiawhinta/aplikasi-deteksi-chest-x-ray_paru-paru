- Instalasi Flask: pip install flask
- Buat Struktur Proyek:
     - Buat folder baru untuk proyek Flask Anda.
     - Di dalam folder proyek, buat file dengan nama app.py. File ini akan berisi kode Flask Anda.
     - Buat folder bernama templates di dalam folder proyek. Folder ini akan berisi file-template HTML yang akan digunakan untuk tampilan web.
- Kode Dasar Flask:
     Buka file app.py dengan editor teks dan tambahkan kode berikut ini:

-----------------------------------------------------------
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return "Halaman Utama"

if __name__ == '__main__':
    app.run(debug=True)
-----------------------------------------------------------

- Template HTML:
     - Buat file HTML baru di dalam folder templates. Beri nama file index.html.
     - Buka file index.html dengan editor teks dan tambahkan kode html kita.

- cara running: python app.py
- menerima gambar: static_folder='images'
- menerima request: methods=['POST']


================= upload flask di PythonAnywhere untuk orang awam ===================

- Buka situs web PythonAnywhere (https://www.pythonanywhere.com/) dan buat akun baru.
- Unggah aplikasi Flask ke PythonAnywhere
- Buat virtual environment (lingkungan virtual):
     - mkvirtualenv --python=/usr/bin/python3.8 myenv
     - workon myenv
     - pip install flask

- Simpan konfigurasi dan restart aplikasi web.