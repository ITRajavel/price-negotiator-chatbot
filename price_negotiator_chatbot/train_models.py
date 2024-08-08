import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Expanded and diverse training data
data = {
    'query': [
        # Pricing queries
        'How much is product1?',
        'What is the cost of product1?',
        'How much does product1 cost?',
        'Price of product1?',
        'Can you tell me the price of product1?',
        
        # Negotiation queries
        'Can I get a discount on product1?',
        'Can you lower the price of product1?',
        'Is there a discount available for product1?',
        'Offer me a better price for product1.',
        'I want to negotiate the price of product1.',
        
        # Order help queries
        'I need help with my order.',
        'How can I get support for my order?',
        'I have issues with my order.',
        'Can you assist me with my order?',
        'Help with order problems.',
        
        # Order tracking queries
        'How do I track my order?',
        'Can I check the status of my order?',
        'Where can I find my order tracking number?',
        'I want to track my order.',
        'Tracking my order.',
        
        # Return policy queries
        'What is your return policy?',
        'How do I return a product?',
        'Can I return a purchased item?',
        'Tell me about your return policy.',
        'How to return an item.',
        
        # Return help queries
        'I need help returning a product.',
        'How do I process a return?',
        'Help with returning an item.',
        'Return instructions, please.',
        'Guide on returning a product.',
        
        # Payment issues queries
        'I am having trouble with payment.',
        'Payment issue assistance needed.',
        'How to resolve payment problems?',
        'Help with payment issues.',
        'I encountered a payment problem.',
        
        # Payment methods queries
        'What payment methods do you accept?',
        'How can I pay for my purchase?',
        'Available payment options?',
        'Which payments are accepted?',
        'Payment methods available.',
        
        # Customer support queries
        'Do you have customer support?',
        'Can I reach out for support?',
        'Customer support availability.',
        'Support team contact.',
        'Is customer support available?',
        
        # Contact support queries
        'How can I contact support?',
        'What is the support email?',
        'Support phone number?',
        'Reach customer support at?',
        'Contact information for support.'
    ],
    'intent': [
        # Pricing intents
        'price_query',
        'price_query',
        'price_query',
        'price_query',
        'price_query',
        
        # Negotiation intents
        'negotiate',
        'negotiate',
        'negotiate',
        'negotiate',
        'negotiate',
        
        # Order help intents
        'order_help',
        'order_help',
        'order_help',
        'order_help',
        'order_help',
        
        # Order tracking intents
        'order_tracking',
        'order_tracking',
        'order_tracking',
        'order_tracking',
        'order_tracking',
        
        # Return policy intents
        'return_policy',
        'return_policy',
        'return_policy',
        'return_policy',
        'return_policy',
        
        # Return help intents
        'return_help',
        'return_help',
        'return_help',
        'return_help',
        'return_help',
        
        # Payment issues intents
        'payment_issues',
        'payment_issues',
        'payment_issues',
        'payment_issues',
        'payment_issues',
        
        # Payment methods intents
        'payment_methods',
        'payment_methods',
        'payment_methods',
        'payment_methods',
        'payment_methods',
        
        # Customer support intents
        'customer_support',
        'customer_support',
        'customer_support',
        'customer_support',
        'customer_support',
        
        # Contact support intents
        'contact_support',
        'contact_support',
        'contact_support',
        'contact_support',
        'contact_support'
    ]
}

df = pd.DataFrame(data)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['query'])
y = df['intent']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model and vectorizer
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
