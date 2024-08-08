
from flask import Flask, request, jsonify, render_template
import pickle
import random
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///conversations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load the pre-trained model and vectorizer
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Simulated product prices and other info
product_prices = {
    'product1': 100,
    'product2': 150,
}

# Define multiple responses for each intent
responses = {
    'price_query': [
        "my name is Chattu.",
        "The price of product1 is $100.",
        "Product1 costs $100.",
        "You can get product1 for $100."
    ],
    'negotiate': [
        "You can get product1 for $90 with a discount.",
        "We can offer product1 at a 10% discount, making it $90.",
        "Product1 is available for $90 if you negotiate."
    ],
    'order_help': [
        "For order help, please visit our Order Support page or contact our customer service.",
        "Need help with your order? Check our Order Support page or reach out to customer service.",
        "Order issues? Visit our Order Support page or get in touch with customer service."
    ],
    'order_tracking': [
        "You can track your order using the tracking number sent to your email.",
        "To track your order, use the tracking number provided in your email.",
        "Check your email for the tracking number to monitor your order."
    ],
    'return_policy': [
        "Our return policy allows returns within 10 days of purchase. Please visit our Returns page for more details.",
        "You can return items within 10 days of purchase. See our Returns page for more information.",
        "Returns are accepted within 10 days of purchase. Check out our Returns page for details."
    ],
    'return_help': [
        "To return a product, please visit our Returns page and follow the instructions provided.",
        "Visit our Returns page for step-by-step instructions on returning a product.",
        "Need to return a product? Follow the instructions on our Returns page."
    ],
    'payment_issues': [
        "For payment issues, please contact our support team or check our Payment Help page.",
        "Payment problems? Reach out to our support team or visit our Payment Help page.",
        "If you encounter payment issues, contact support or check the Payment Help page."
    ],
    'payment_methods': [
        "We accept various payment methods including credit cards, PayPal, and bank transfers.",
        "You can pay using credit cards, PayPal, or bank transfers.",
        "Available payment methods include credit cards, PayPal, and bank transfers."
    ],
    'customer_support': [
        "Yes, we have a dedicated customer support team available 24/7.",
        "Our customer support team is available around the clock.",
        "We provide 24/7 customer support for your convenience."
    ],
    'contact_support': [
        "You can contact our customer support via email at support@example.com or call us at 123-456-7890.",
        "Reach our customer support at support@example.com or call 123-456-7890.",
        "For support, email us at support@example.com or dial 123-456-7890."
    ]
}

# Database model for storing conversations
class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_message = db.Column(db.String(500), nullable=False)
    bot_response = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Conversation {self.id}>'

# Create database tables
with app.app_context():
    db.create_all()

def get_intent(query):
    X = vectorizer.transform([query])
    intent = model.predict(X)[0]
    return intent

def get_response(intent):
    if intent in responses:
        return random.choice(responses[intent])
    return "I'm sorry, I didn't understand that."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json['query']
    intent = get_intent(user_query)
    response = get_response(intent)

    # Store conversation in the database
    conversation = Conversation(user_message=user_query, bot_response=response)
    db.session.add(conversation)
    db.session.commit()

    return jsonify({'response': response})

@app.route('/conversations', methods=['GET'])
def get_conversations():
    conversations = Conversation.query.all()
    result = []
    for convo in conversations:
        result.append({
            'user_message': convo.user_message,
            'bot_response': convo.bot_response,
            'timestamp': convo.timestamp
        })
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
