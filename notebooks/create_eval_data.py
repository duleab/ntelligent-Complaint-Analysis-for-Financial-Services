import json
import os

# Create the data directory if it doesn't exist
os.makedirs('../data', exist_ok=True)

# Sample evaluation questions
eval_data = {
    "questions": [
        {
            "question": "What are common issues with credit cards?",
            "ground_truth": "Common credit card issues include unauthorized charges, billing errors, fraud, and difficulty with payment processing.",
            "expected_products": ["Credit card"],
            "expected_issues": ["Unauthorized transactions", "Billing disputes"]
        },
        {
            "question": "How can I dispute a bank transaction?",
            "ground_truth": "To dispute a bank transaction, contact your bank's customer service, provide details of the transaction, and submit any supporting documentation.",
            "expected_products": ["Checking account", "Savings account"],
            "expected_issues": ["Transaction dispute"]
        },
        {
            "question": "What should I do if I suspect fraud on my account?",
            "ground_truth": "If you suspect fraud, immediately contact your financial institution, freeze your accounts if possible, and monitor your credit reports.",
            "expected_products": ["Credit card", "Bank account"],
            "expected_issues": ["Fraud", "Identity theft"]
        }
    ]
}

# Write to file
with open('../data/evaluation_questions.json', 'w') as f:
    json.dump(eval_data, f, indent=2)

print("Evaluation questions file created successfully!")
