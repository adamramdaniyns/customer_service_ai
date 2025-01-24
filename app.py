import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from AI.main import chat_customer_service


# Load .env file
load_dotenv()
app = Flask(__name__)

port = int(os.getenv("FLASK_RUN_PORT", 5000)) 

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route('/chat-cs', methods=['POST'])
def handle_chat_cs():
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Invalid input, 'prompt' is required"}), 400
    
    # Mengambil prompt dari user
    prompt = data["prompt"]

    if prompt == '':
        return jsonify({"error": "Invalid input. Please provide a valid question."}), 400
    
    # Memproses prompt menggunakan fungsi dari modul AI
    response = chat_customer_service(prompt)
    
    # Mengembalikan hasil sebagai JSON
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True, port=port)