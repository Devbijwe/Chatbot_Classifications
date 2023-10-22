import os
import nltk
import ssl
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV

# Import and configure the necessary NLP libraries
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Define your intents as you did previously
with open('intents.json', 'r') as file:
    intents = json.load(file)
    
intents = [
    {
    "tag": "greeting",
    "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
    "responses": [
        "Hello and welcome! How can I assist you today?",
        "Hi there! I'm here to help. What can I do for you?",
        "Hey! It's great to see you. How can I assist you?",
        "Hello! I hope you're having a fantastic day. How can I be of service?",
        "What's up? I'm here to chat and provide information. How can I assist you?"
    ]
}
,
    {
    "tag": "goodbye",
    "patterns": ["Bye", "See you later", "Goodbye", "Bye for now"],
    "responses": [
        "Goodbye! If you have more questions in the future, don't hesitate to ask.",
        "See you later! Feel free to return whenever you need assistance.",
        "Farewell! If you ever need help again, you know where to find me.",
        "Goodbye for now! Take care, and have a wonderful day.",
        "Bye for now! If you ever want to chat, I'll be here."
    ]
}
,
    {
    "tag": "thanks",
    "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
    "responses": [
        "You're very welcome! If you have more questions, feel free to ask anytime.",
        "No problem at all! If you need assistance in the future, just let me know.",
        "You're welcome! It was my pleasure to help. If you have more inquiries, don't hesitate to reach out.",
        "You're welcome! I'm here to assist whenever you need it.",
        "I'm glad I could assist you! If you ever need help again, don't hesitate to contact me."
    ]
}
,
    {
    "tag": "about",
    "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
    "responses": [
        "I am a sophisticated chatbot designed to assist you with a wide range of tasks. My purpose is to answer questions, provide information, and make your life easier.",
        "I'm your friendly virtual assistant! My main goal is to provide you with helpful information, answer your queries, and assist you with various tasks.",
        "I am an AI-powered chatbot, and I'm here to assist you with your questions and tasks. My mission is to make your life more convenient.",
        "I'm here to help you. I'm a virtual assistant with a mission to provide information and support. Feel free to ask me anything!",
        "I'm your virtual helper! My purpose is to offer you information and assistance whenever you need it. How can I assist you today?"
    ]
}
,
    {
    "tag": "help",
    "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
    "responses": [
        "Of course, I'm here to help. What do you need assistance with?",
        "I'm here to assist you. What's the issue you're facing or the question you have?",
        "How can I assist you today? Feel free to share your concern or question, and I'll do my best to help.",
        "Sure, I'm here to provide support. What's on your mind? How can I assist you today?",
        "I'm here to help. What can I do for you today? Feel free to ask any questions or share your concerns."
    ]
}
,
    {
    "tag": "age",
    "patterns": ["How old are you", "What's your age"],
    "responses": [
        "I don't have an age. I'm a chatbot created in the digital realm, so I'm ageless.",
        "I don't age as I'm a digital creation. I exist to assist you with your questions and tasks.",
        "Age is just a number, but I don't have one. I'm a digital entity here to assist you.",
        "I was 'birthed' in the digital world, so I don't have a physical age like humans do.",
        "I'm ageless since I exist solely in the digital domain, ready to help you at any time."
    ]
}
,
    {
    "tag": "weather",
    "patterns": ["What's the weather like", "How's the weather today"],
    "responses": [
        "I'm sorry, but I cannot provide real-time weather information. To get the current weather, I recommend using a weather app or website specific to your location.",
        "For up-to-date weather information, it's best to use a dedicated weather service or app that provides real-time data based on your location.",
        "I don't have access to real-time weather data. To check the current weather conditions, I suggest using a reliable weather website or app.",
        "I'm unable to provide real-time weather updates. You can easily find current weather information by using a weather app or visiting a weather website.",
        "To get accurate and up-to-date weather information for your location, I recommend using a dedicated weather service or app."
    ]
}
,
    {
    "tag": "budget",
    "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
    "responses": [
        "Creating a budget is a vital step in managing your finances. Begin by calculating your monthly income. Next, list your fixed expenses like rent, utilities, groceries, and bills. Allocate a portion of your income to savings and debt repayment. The remainder can be spent on non-essential expenses like entertainment. One popular budgeting method is the 50/30/20 rule, which divides your income into needs, wants, and savings/debt repayment, respectively.",
        "Budgeting is a wise financial practice. Start by tracking your income and expenses. Categorize your spending into essential and non-essential categories. Allocate most of your income to essentials, some to savings, and a portion to non-essentials. A well-known budgeting approach is the 50/30/20 rule, which suggests dividing your income into these categories.",
        "To create an effective budget, begin by calculating your monthly income. Next, list your fixed costs, such as rent, utilities, and groceries. Assign a portion of your income to savings and debt payments, and the rest can be used for non-essential expenses. Many people find success with the 50/30/20 rule, dividing their income accordingly.",
        "Making a budget is a smart financial move. Start by calculating your monthly income. Then, identify your fixed costs, like rent, utilities, and groceries. Allocate funds for savings and debt payments, and the remainder can be used for non-essential expenses. A popular approach is the 50/30/20 rule, which suggests dividing your income into these categories.",
        "Budgeting is a valuable skill for managing your money. Begin by calculating your monthly income, listing your fixed costs, and allocating funds for savings and discretionary spending. A classic approach is the 50/30/20 rule, which suggests dividing your income into these categories."
    ]
}
,
    {
    "tag": "credit_score",
    "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
    "responses": [
        "A credit score is a numerical representation of your creditworthiness. It's based on your credit history and is used by lenders to assess your credit risk. A higher credit score indicates lower credit risk, making it easier to qualify for loans and better interest rates. You can check your credit score for free on various websites like Credit Karma and Credit Sesame. To improve your credit score, pay your bills on time, reduce credit card balances, and avoid opening too many new accounts.",
        "Your credit score is a three-digit number that reflects your creditworthiness. Lenders use it to evaluate your credit risk when you apply for loans or credit cards. You can check your credit score for free on websites like Credit Karma or Credit Sesame. To improve your credit score, focus on paying your bills on time, maintaining a low credit card balance, and managing your credit responsibly.",
        "A credit score is a number that represents your creditworthiness. It's calculated based on your credit history and helps lenders assess your credit risk. You can check your credit score for free on websites like Credit Karma. To improve your credit score, make sure to pay your bills on time, reduce credit card balances, and manage your credit responsibly.",
        "Your credit score is a crucial financial indicator. It's a number that reflects your creditworthiness and is used by lenders to make decisions about loans and credit. You can check your credit score for free on platforms like Credit Karma. To boost your credit score, focus on paying your bills promptly, keeping credit card balances low, and managing your credit wisely.",
        "A credit score is a numerical representation of your creditworthiness. It's derived from your credit history and is used by lenders to evaluate your credit risk. To check your credit score, you can use free services like Credit Karma. To improve your credit score, be sure to pay your bills on time, maintain low credit card balances, and use credit responsibly."
    ]
}
,
    {
    "tag": "about_brand",
    "patterns": ["Tell me about your brand", "What's your brand all about?"],
    "responses": [
        "Our brand is dedicated to quality and style. We take pride in offering products that not only look great but are also built to last.",
        "We're all about quality and style. Our brand focuses on creating products that are not only fashionable but also comfortable and long-lasting.",
        "Our brand is known for its commitment to quality and style. We believe that great products should be a blend of fashion and comfort.",
        "We're passionate about quality and style. Our brand is committed to offering unique designs and products that make a statement.",
        "Quality and style are at the core of our brand. We're here to provide you with products that not only look good but feel great to wear."
    ]
}
,
       {
    "tag": "product_inquiry",
    "patterns": ["Show me your products", "What do you sell?"],
    "responses": [
        "Explore our extensive range of products on our website. We offer a variety of items, including T-shirts, hoodies, and much more.",
        "You can discover our wide selection of products on our website. Our product range includes T-shirts, hoodies, and many other exciting items.",
        "Visit our website to see our diverse product lineup. We offer a broad range of products, including T-shirts, hoodies, and more.",
        "You'll find a variety of products on our website, including T-shirts, hoodies, and numerous other items. Feel free to browse and discover what interests you.",
        "Our website showcases a diverse selection of products, from stylish T-shirts to comfortable hoodies and beyond. Take a look to see what catches your eye."
    ]
}
,
       {
    "tag": "quality_tshirts",
    "patterns": ["Tell me about your T-shirt quality", "How good are your T-shirts?"],
    "responses": [
        "Our T-shirts are crafted from high-quality materials that are chosen for comfort and durability. We prioritize providing products that our customers can rely on.",
        "When it comes to T-shirt quality, we take it seriously. Our T-shirts are made from premium materials to ensure both comfort and longevity.",
        "We place a strong emphasis on quality when it comes to our T-shirts. You can expect our T-shirts to be both comfortable and durable, offering you the best of both worlds.",
        "We understand the importance of T-shirt quality. That's why our T-shirts are made from top-notch materials, ensuring they're comfortable and built to last.",
        "We take pride in the quality of our T-shirts. They are designed to provide a high level of comfort and are constructed with durability in mind."
    ]
}
,
       {
    "tag": "order_status",
    "patterns": ["Where's my order?", "Check my order status"],
    "responses": [
        "Certainly, I can help you with that. To check your order status, please provide your order number, and I'll retrieve the information for you.",
        "No problem, I'd be happy to assist. To check your order status, I'll need your order number. Once you provide that, I can look it up for you.",
        "I can certainly check your order status. To get started, may I have your order number, please?",
        "Sure, I can help with that. Please share your order number, and I'll promptly check the status of your order.",
        "Of course, I can assist with your order status. To proceed, kindly provide your order number, and I'll look up the information for you."
    ]
}
,
        {
    "tag": "order_change",
    "patterns": ["Can I change my order?", "Modify my order"],
    "responses": [
        "We'll do our best to assist you with order changes. To proceed, please contact our dedicated support team, and they will guide you through the process.",
        "Certainly, order modifications can be made. I recommend reaching out to our support team, and they'll provide you with the necessary steps to adjust your order.",
        "Order changes are possible. To initiate any modifications, please get in touch with our support team, and they will guide you through the process.",
        "You can make order changes. For this, I recommend reaching out to our support team, and they will assist you with the necessary steps for modifying your order.",
        "Yes, order changes can be accommodated. To get started, I suggest contacting our support team, and they will provide you with the information and guidance you need."
    ]
}
,
       {
    "tag": "shipping_info",
    "patterns": ["Tell me about shipping", "How long does shipping take?"],
    "responses": [
        "Our shipping process is efficient and designed to get your order to you as quickly as possible. Shipping times may vary based on your location. For detailed shipping information, please visit our website.",
        "We aim to provide speedy delivery to our customers. However, shipping times can vary depending on your location. For precise details on shipping times, I recommend visiting our website or contacting our support team.",
        "We prioritize efficient shipping to ensure your order reaches you promptly. Shipping times may differ based on your location, so I suggest checking our website for specific details.",
        "We work diligently to ensure efficient shipping. Shipping times may vary by your location, so for precise estimates, I recommend visiting our website or reaching out to our support team.",
        "Efficiency is key in our shipping process. Shipping times are influenced by your location, so I encourage you to check our website for detailed information or reach out to our support team."
    ]
}
,
       {
    "tag": "return_policy",
    "patterns": ["How can I return a product?", "What's your return policy?"],
    "responses": [
        "Returning a product is straightforward. Detailed instructions can be found on our return policy page on our website. It will guide you through the process step by step.",
        "Returning a product is easy. Our return policy page on our website provides comprehensive instructions on the return process. You can follow the steps outlined there.",
        "Our return policy page on the website contains detailed instructions on returning a product. I recommend visiting the page for a step-by-step guide on the return process.",
        "The process for returning a product is outlined in detail on our return policy page on our website. You can find step-by-step instructions there to guide you through the process.",
        "To understand how to return a product, I suggest referring to our return policy page on our website. It provides a clear, step-by-step guide to help you with the process."
    ]
}
,
        {
    "tag": "size_guide",
    "patterns": ["What sizes are available?", "How do I choose the right size?"],
    "responses": [
        "Selecting the perfect size is crucial. To find your ideal fit for our T-shirts, refer to our size guide on our website. It provides comprehensive guidance on choosing the right size based on your measurements.",
        "Choosing the right size is important for comfort. Our website includes a size guide to assist you in selecting the ideal fit for our T-shirts. It's based on your measurements.",
        "To ensure your T-shirt fits perfectly, use our size guide available on our website. It will help you choose the right size based on your measurements.",
        "Finding the right T-shirt size is important for your comfort. On our website, you'll discover a size guide that offers guidance on selecting the best size based on your measurements.",
        "We want you to have the perfect fit. Our website features a size guide to help you choose the right size for our T-shirts. Simply follow the recommendations based on your measurements."
    ]
}
,
       {
    "tag": "promotions",
    "patterns": ["Are there any ongoing promotions?", "Tell me about discounts"],
    "responses": [
        "Yes, we frequently run promotions and offer discounts to our valued customers. To stay updated with the latest deals and offers, I recommend visiting our website's 'Promotions' section.",
        "Absolutely, we regularly have promotions and provide discounts to our customers. To stay informed about the most current deals, please visit the 'Promotions' section on our website.",
        "We love offering promotions and discounts to our customers. To keep track of our current deals, be sure to visit our website's 'Promotions' section.",
        "Of course, we frequently have promotions and discounts. To see the latest deals and stay informed about our promotions, please check the 'Promotions' section on our website.",
        "Yes, we're all about promotions and discounts for our customers. To find out about our current deals, I encourage you to visit our website's 'Promotions' section."
    ]
}
,
        {
    "tag": "feedback",
    "patterns": ["I want to provide feedback", "Leave a review"],
    "responses": [
        "Your feedback is greatly appreciated! You can share your thoughts and leave a review on our website. Your input helps us improve our products and services.",
        "We value your feedback! Feel free to leave a review on our website to let us know about your experience. Your input is essential for us to enhance our offerings.",
        "Your feedback is important to us. You can provide your thoughts and leave a review on our website. We're eager to hear about your experience with our products and services.",
        "We appreciate your feedback. You can share your thoughts and leave a review on our website. Your input assists us in continually improving our products and services.",
        "Your feedback is crucial to us. You can easily provide your thoughts and leave a review on our website. We look forward to hearing about your experience."
    ]
}
,
        {
            "tag": "contact_info",
            "patterns": ["How can I contact your customer support?", "Give me your contact details"],
            "responses": ["Our customer support team is available at [customer-support-email] or [customer-support-phone]. Feel free to reach out anytime.", "For assistance, contact our customer support at [customer-support-email] or [customer-support-phone]."]
        },
        {
            "tag": "social_media",
            "patterns": ["Are you on social media?", "Tell me about your social accounts"],
            "responses": ["Follow us on [social-media-platforms] for the latest updates, promotions, and behind-the-scenes content.", "Stay connected with us on social media for the latest news."]
        },
        {
            "tag": "sustainability",
            "patterns": ["What is your commitment to sustainability?", "Tell me about eco-friendly practices"],
            "responses": ["We are dedicated to sustainability. Learn more about our eco-friendly initiatives on our website.", "Discover our efforts toward sustainability on our website."]
        },
        {
            "tag": "bulk_orders",
            "patterns": ["Can I place a bulk order?", "Tell me about bulk discounts"],
            "responses": ["Yes, we accommodate bulk orders. Contact us for special pricing and details.", "Bulk orders are welcome. Reach out for pricing details."]
        },
         {
            "tag": "availability",
            "patterns": ["Is this T-shirt available in blue?", "What colors are in stock?"],
            "responses": ["Yes, we have the T-shirt in blue.", "Our available colors include blue, red, black, and white."]
        },
        {
            "tag": "discounts",
            "patterns": ["Tell me about current discounts", "Are there any special offers?"],
            "responses": ["Check our website for ongoing discounts and special offers.", "You can find current discounts on our promotions page."]
        },
        {
            "tag": "gifts",
            "patterns": ["Can I purchase a T-shirt as a gift?", "Gift options"],
            "responses": ["Absolutely, you can buy a T-shirt as a gift and choose gift-wrapping during checkout.", "We offer gift options, including gift wrapping and personalized notes."]
        },
        {
            "tag": "order_confirmation",
            "patterns": ["How will I receive my order confirmation?", "Send me a confirmation email"],
            "responses": ["You will receive an email confirmation shortly after placing your order.", "An order confirmation email will be sent to the provided email address."]
        },
        {
            "tag": "order_cancel",
            "patterns": ["I need to cancel my order", "Cancel my recent order"],
            "responses": ["Please contact our support team to initiate the order cancellation process.", "Order cancellations can be requested by contacting our support."]
        },
        {
            "tag": "styling_tips",
            "patterns": ["Give me styling tips for T-shirts", "How to style a T-shirt?"],
            "responses": ["T-shirts can be styled in various ways, from casual to chic. Check our blog for styling tips.", "Explore our blog for T-shirt styling ideas and inspiration."]
        },
        {
            "tag": "customer_reviews",
            "patterns": ["Can I see customer reviews?", "Show me product reviews"],
            "responses": ["You can read customer reviews on our product pages to hear about their experiences.", "Customer reviews are available on individual product pages."]
        },
        {
            "tag": "international_shipping",
            "patterns": ["Do you offer international shipping?", "Can I order from outside the country?"],
            "responses": ["Yes, we offer international shipping. Shipping rates and times may vary.", "International shipping is available. Please check shipping details for your location."]
        },
        {
            "tag": "loyalty_program",
            "patterns": ["Tell me about your loyalty program", "Do you have a rewards program?"],
            "responses": ["We have a loyalty program that rewards our valued customers. Visit our loyalty program page for details.", "Learn about our rewards program on our website."]
        },
        {
            "tag": "size_exchange",
            "patterns": ["What if I order the wrong size?", "Can I exchange for a different size?"],
            "responses": ["No worries! We offer size exchanges. Contact our support to arrange an exchange.", "You can exchange a product for a different size by reaching out to our support team."]
        },
        {
            "tag": "new_arrivals",
            "patterns": ["Show me the latest T-shirt arrivals", "What's new in the collection?"],
            "responses": ["Explore our latest T-shirt arrivals on the 'New In' page.", "Check out our newest T-shirt arrivals in the 'New Collection' section."]
        },
        {
            "tag": "return_policy",
            "patterns": ["What is your return policy?", "How can I return a T-shirt?"],
            "responses": ["You can review our return policy on the 'Returns & Refunds' page.", "Our return policy details are available in the 'Customer Service' section."]
        },
        {
            "tag": "order_tracking",
            "patterns": ["How can I track my order?", "Check order status", "Where's my T-shirt?"],
            "responses": ["To track your order, enter your order number on our 'Order Tracking' page.", "You can check the status of your order by visiting our 'Order Status' page."]
        },
        {
            "tag": "customer_support",
            "patterns": ["I need help from customer support", "Contact support", "Help me with an issue"],
            "responses": ["Our customer support team is here to assist you. You can reach out to them via our 'Contact Us' page.", "For assistance, please contact our customer support team through our 'Help Center'."]
        },
        {
            "tag": "payment_methods",
            "patterns": ["What payment methods do you accept?", "Can I pay with PayPal?", "Credit card payment"],
            "responses": ["We accept various payment methods, including credit cards and PayPal. Details are available during checkout.", "You can pay for your order using major credit cards and PayPal as secure payment options."]
        },
        {
            "tag": "sizing_guide",
            "patterns": ["Guide me on T-shirt sizing", "How do I find the right size?", "Sizing chart"],
            "responses": ["Use our sizing guide and chart to select the perfect size for your T-shirt.", "Find the right size for your T-shirt with our sizing chart and guide available on the product page."]
        },
        {
            "tag": "order_delivery",
            "patterns": ["When can I expect delivery?", "Tell me about order delivery", "Delivery time"],
            "responses": ["Delivery times may vary, but you can find estimated delivery details on the product page.", "Expected delivery times can be found on each product's details page."]
        },
        {
            "tag": "tshirt_care",
            "patterns": ["How do I care for my T-shirt?", "T-shirt maintenance tips", "Washing instructions"],
            "responses": ["Check the care label on your T-shirt for washing instructions. General tips: machine wash cold and tumble dry low.", "To maintain your T-shirt, follow the care label's instructions. Typically, machine wash in cold water and tumble dry on low heat."]
        },
        {
            "tag": "customization",
            "patterns": ["Can I customize a T-shirt?", "Personalized T-shirts", "Custom design options"],
            "responses": ["Yes, we offer customization options for T-shirts. You can add your personal touch during the ordering process.", "Create a unique T-shirt with our customization feature. Personalize the design and colors."]
        },
        {
            "tag": "delivery_costs",
            "patterns": ["How much is shipping?", "Tell me about delivery costs", "Is shipping free?"],
            "responses": ["Shipping costs depend on the location and shipping method. Details can be found on the checkout page.", "Shipping fees vary by location and delivery method. You can view the costs during checkout."]
        },
        {
            "tag": "best_sellers",
            "patterns": ["Show me the best-selling T-shirts", "What's popular?", "Top-rated T-shirts"],
            "responses": ["Explore our best-sellers in the 'Top Picks' section.", "Discover our most popular T-shirts in the 'Best Sellers' category."]
        },
        {
            "tag": "bulk_orders",
            "patterns": ["Can I order T-shirts in bulk?", "Bulk purchasing options", "Large orders"],
            "responses": ["Yes, we offer bulk ordering options. Contact our sales team for special pricing.", "For large orders, please get in touch with our sales department to discuss bulk purchase arrangements."]
        },
        {
            "tag": "color_options",
            "patterns": ["What colors are available?", "Tell me about color choices", "Available T-shirt colors"],
            "responses": ["Our T-shirts come in a variety of colors. Browse the product pages to view available color options.", "You'll find a wide range of colors for our T-shirts. Check the product details for color choices."]
        },
        {
            "tag": "gift_options",
            "patterns": ["Do you offer gift packaging?", "Gift-wrapping options", "Gift-ready T-shirts"],
            "responses": ["We provide gift packaging options for special occasions. Select this during checkout.", "Make your T-shirt a perfect gift with our gift-wrapping service available at checkout."]
        },
        {
            "tag": "materials_used",
            "patterns": ["What materials are T-shirts made of?", "Fabric used for T-shirts", "T-shirt material"],
            "responses": ["Our T-shirts are made from high-quality and comfortable materials. Details are available on the product pages.", "We use premium materials to craft our T-shirts. You can find material information on each product's page."]
        },
        {
            "tag": "out_of_stock",
            "patterns": ["Is this T-shirt out of stock?", "Availability of T-shirt", "When will it be restocked?"],
            "responses": ["If a T-shirt is out of stock, you can request a restock notification on the product page.", "For out-of-stock items, you can sign up for restock notifications on the product details page."]
        },
        {
            "tag": "discount_offers",
            "patterns": ["Are there any ongoing discounts?", "Current promotions", "Tell me about sales"],
            "responses": ["We frequently offer discounts and promotions. Check our 'Offers' page for current deals.", "Stay updated with our latest discounts and sales on the 'Promotions' page."]
        },
        {
            "tag": "recommendations",
            "patterns": ["Can you recommend T-shirts for me?", "Suggest some T-shirts", "Recommendations based on my style"],
            "responses": ["Sure, I can help. What's your preferred style or color?", "I'd be happy to suggest T-shirts based on your style. Could you share your preferences?"]
        },
        {
            "tag": "product_availability",
            "patterns": ["Is this T-shirt available in my size?", "Check size availability", "Do you have it in size L?"],
            "responses": ["You can check the availability of sizes on the product page. Select your size to see if it's in stock.", "To see if your size is available, go to the product page and choose your size."]
        },
        {
            "tag": "delivery_status",
            "patterns": ["Where is my T-shirt? When will it arrive?", "Track my order", "Status of my delivery"],
            "responses": ["To track your order and view delivery status, enter your order number in the 'Order Tracking' section.", "You can track your order's delivery status by entering your order number on the 'Delivery Status' page."]
        }
]

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(max_df=0.85, max_features=1000, stop_words='english')  # You can adjust these parameters
clf = LogisticRegression(random_state=0, max_iter=1000, C=1.0, solver='lbfgs')  # You can adjust these parameters
calibrated_clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
calibrated_clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    intent_probs = calibrated_clf.predict_proba(input_text).max()
    print("prob:", intent_probs)
    # Define a confidence threshold
    confidence_threshold = 0.07  # Adjust as needed

    if intent_probs >= confidence_threshold:
        tag = calibrated_clf.predict(input_text)[0]
        for intent in intents:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                return response
    else:
        return "I'm sorry, I couldn't understand that. Please try rephrasing your question or provide more context."

def main():
    while True:
        user_input = input("You (Type 'no' to stop): ").lower()
        if user_input == "no":
            print("Chatbot: Goodbye!")
            break
        response = chatbot(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
