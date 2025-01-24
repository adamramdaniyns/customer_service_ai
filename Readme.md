# Customer Service Chatbot with Gemini 1.5 and Cosine Similarity

This project is a customer service chatbot that leverages the Google Gemini 1.5 model (Flash) along with embedding and cosine similarity techniques to match products to user inquiries. The chatbot is designed to help users with product-related queries, such as explaining product details and facilitating order placements.

## Features

- **Product Query Handling**: The chatbot responds to questions about products and provides relevant information based on the products in the database.
- **Order Handling**: Users can request to add products to their cart, and the chatbot generates an order in the specified format.
- **Cosine Similarity Matching**: Embedding is used to convert text input into vector representations, and cosine similarity is applied to match user input with the most relevant products.
- **Google Gemini 1.5**: The chatbot uses the Flash model from Google's Gemini suite for natural language processing and response generation.

## Installation

To set up the project, follow these steps:

1. Clone this repository:

```bash
git clone https://github.com/adamramdaniyns/customer_service_ai.git
cd customer_service_ai
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
**Dependencies**
The project uses the following libraries:

- google-generativeai: For interfacing with Google Gemini's API (Flash 1.5 model).
- scikit-learn: For implementing cosine similarity in text comparison.
- sentence-transformers: For generating embeddings from user input and products.
- torch: For supporting model computations with PyTorch.

**Usage**
1. Starting the Chatbot
To interact with the chatbot, you can call the chat_customer_service function, which will classify user input and route it to the appropriate agent (either for product queries or order handling).
```bash
response = chat_customer_service("Apa harga Meja TV?")
print(response)
```
2. Classifying User Input
The input is classified into two categories:

```bash
Product-related questions: The agent_product will provide product information based on the closest match to the user's query.
Order-related requests: The agent_taking_order will create an order for the product that the user requests.
```

3. Embedding and Cosine Similarity
The system uses sentence embeddings to represent both product data and user queries. The cosine similarity is then used to find the most relevant product based on the similarity score between the query and the products.
```bash
input_embedding = model.encode(user_query, convert_to_tensor=True)
cosine_sim = util.pytorch_cos_sim(input_embedding, product_embeddings)
```

**Customizing Product Data**

The product data used by the chatbot can be easily replaced with your own dataset. The current product data is stored in the product_data.json file. To update the product list, simply modify this file to include the products you want the chatbot to recognize and provide information about. Make sure to maintain the format of the product data for proper functionality.

4. Flask Web Server

This application also uses **Flask** to provide an API for interacting with the chatbot. Flask handles incoming user requests through HTTP endpoints and routes them to the chatbot for further processing. Flask enables users to interact with the chatbot via a simple REST API.

- Route /chat-cs: This endpoint accepts POST requests with JSON data containing the user's prompt. Flask will call the chat_customer_service(prompt) function to process the request and return the chatbot's response in JSON format.

Example API request using **POST**:

```bash
curl -X POST http://localhost:5000/chat-cs -H "Content-Type: application/json" -d '{"prompt": "What is the price of the TV table?"}'
```

- Route /: The root endpoint that simply returns the message "Hello, Flask!" to ensure the server is running correctly.

**Running the Flask Server:**
- To start the Flask server, simply run the following command:

```bash
python app.py
```

Once the server is up, you can access it at http://localhost:5000.

With Flask, users can send requests to get product information or process orders via the API, while the AI model handles natural language processing and response generation.

