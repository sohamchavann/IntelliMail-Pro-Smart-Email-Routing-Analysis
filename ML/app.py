from flask import Flask, request, jsonify
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
import pickle
import requests
import logging
from bs4 import BeautifulSoup

# Initialize the Flask web application
app = Flask(__name__)

# --- Load Machine Learning Models and Tokenizers ---

# Load the pre-trained model for email category classification
category_model_path = "distilbert_model_final.pkl"
with open(category_model_path, "rb") as f:
    category_model = pickle.load(f)

# Load the corresponding tokenizer for the category model
category_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Define the dictionary to map model output labels to human-readable categories
category_label_dict = {0: 'Finance', 1: 'Energy', 2: 'Technology', 3: 'Pharmaceutical', 4: 'Travel'}

# Load the pre-trained model for email sentiment analysis
sentiment_model_path = "sentiment_analysis.pkl"
with open(sentiment_model_path, "rb") as f:
    sentiment_model = pickle.load(f)

# Load the corresponding tokenizer for the sentiment model
sentiment_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Define the dictionary to map model output labels to human-readable sentiments
sentiment_label_dict = {0: 'enquiry', 1: 'satisfied', 2: 'default',3:'dissatisfied',4:'emergency'}

# --- Helper Functions for Email Processing ---

def preprocess_text(text):
    """Cleans up text by converting to lowercase and stripping whitespace."""
    return text.strip().lower()

def predict_category(text):
    """Predicts the category of an email body using the pre-trained model."""
    cleaned_text = preprocess_text(text)
    # Tokenize the text for the model
    inputs = category_tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True)
    # Get model outputs
    outputs = category_model(**inputs)
    # Get the predicted label with the highest probability
    predicted_label = torch.argmax(outputs.logits).item()
    # Map the numerical label to its category string
    predicted_category = category_label_dict[predicted_label]
    return predicted_category

def predict_sentiment(text):
    """Predicts the sentiment of an email body using the pre-trained model."""
    cleaned_text = preprocess_text(text)
    # Tokenize the text for the model
    inputs = sentiment_tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True)
    # Get model outputs
    outputs = sentiment_model(**inputs)
    # Get the predicted label with the highest probability
    predicted_label = torch.argmax(outputs.logits).item()
    # Map the numerical label to its sentiment string
    predicted_sentiment = sentiment_label_dict[predicted_label]
    return predicted_sentiment

def html_to_text(html_content):
    """Converts HTML content to plain text using BeautifulSoup."""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text()
        return text
    except Exception as e:
        # Log any errors that occur during conversion
        logging.error(f"Error converting HTML to text: {e}")
        return ""

def process_emails(emails_data):
    """
    Processes a list of raw email data dictionaries.

    Extracts ID, subject, and body, and converts the HTML body to plain text.
    """
    try:
        processed_emails = []
        for email in emails_data:
            email_id = email.get('id', '')
            email_subject = email.get('subject', '')
            email_content = email.get('body', {}).get('content', '')
            plain_text_content = html_to_text(email_content)
            email_body = f"{plain_text_content}"
            processed_email = {
                "id": email_id,
                "body": email_body,
                "subject": email_subject,
                "category": "",
                "sentiment": "",
            }
            processed_emails.append(processed_email)
        logging.info(f"Processed {len(processed_emails)} emails successfully")
        return processed_emails
    except Exception as e:
        logging.error(f"Error processing emails: {e}")
        return []

def fetch_emails():
    """
    Fetches emails from the Microsoft Graph API.
    """
    # access token for API authentication 
    access_token = "eyJ0eXAiOiJKV1QiLCJub25jZSI6ImR5cS1rd1pNZXc4VmdsTWdydktSUExqRHZuZV9xUTg4VW5ETGtoaDA0VTQiLCJhbGciOiJSUzI1NiIsIng1dCI6InEtMjNmYWxldlpoaEQzaG05Q1Fia1A1TVF5VSIsImtpZCI6InEtMjNmYWxldlpoaEQzaG05Q1Fia1A1TVF5VSJ9.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTAwMDAtYzAwMC0wMDAwMDAwMDAwMDAiLCJpc3MiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC8zMTVlMzJhNC1jMTAwLTQ1YTktODUxNS01YWU1M2Y4YjE2MTkvIiwiaWF0IjoxNzEyOTQ3MDQ1LCJuYmYiOjE3MTI5NDcwNDUsImV4cCI6MTcxMzAzMzc0NSwiYWNjdCI6MCwiYWNyIjoiMSIsImFpbyI6IkFUUUF5LzhXQUFBQWdpLzRxODhOME9FRTJYUXVEajFnSkFGcnRpU1krMjJHZTlBeSsrbU55WjQvOGNtdjN3NWFSQ2YzRldBVHBNVXAiLCJhbXIiOlsicHdkIl0sImFwcF9kaXNwbGF5bmFtZSI6IkdyYXBoIEV4cGxvcmVyIiwiYXBwaWQiOiJkZThiYzhiNS1kOWY5LTQ4YjEtYThhZC1iNzQ4ZGE3MjUwNjQiLCJhcHBpZGFjciI6IjAiLCJmYW1pbHlfbmFtZSI6ImxvbGFnZSIsImdpdmVuX25hbWUiOiJvbWthciIsImlkdHlwIjoidXNlciIsImlwYWRkciI6IjExMy4xOTMuNTguMjIwIiwibmFtZSI6Im9ta2FyIGxvbGFnZSIsIm9pZCI6ImMwMmQyYTg5LWE3NmYtNDBlNS05NDU5LTBmZjkxNGMyOWUyZiIsInBsYXRmIjoiMyIsInB1aWQiOiIxMDAzMjAwMzZBRTI3NkY1IiwicmgiOiIwLkFTc0FwREplTVFEQnFVV0ZGVnJsUDRzV0dRTUFBQUFBQUFBQXdBQUFBQUFBQUFEQ0FMcy4iLCJzY3AiOiJBUElDb25uZWN0b3JzLlJlYWQuQWxsIEFQSUNvbm5lY3RvcnMuUmVhZFdyaXRlLkFsbCBNYWlsLlJlYWQgTWFpbC5SZWFkQmFzaWMgTWFpbC5SZWFkV3JpdGUgTWFpbC5TZW5kIE1haWxib3hTZXR0aW5ncy5SZWFkIG9wZW5pZCBwcm9maWxlIFVzZXIuUmVhZCBlbWFpbCIsInN1YiI6Im9GOFlTa2NUZlc4WGRmY192UDYxLVFzeGJnWHZQNmtXRU51dGY3VmZpejgiLCJ0ZW5hbnRfcmVnaW9uX3Njb3BlIjoiQVMiLCJ0aWQiOiIzMTVlMzJhNC1jMTAwLTQ1YTktODUxNS01YWU1M2Y4YjE2MTkiLCJ1bmlxdWVfbmFtZSI6Im9ta2FyTG9sYWdlQGhhY2tvaGlyZS5vbm1pY3Jvc29mdC5jb20iLCJ1cG4iOiJvbWthckxvbGFnZUBoYWNrb2hpcmUub25taWNyb3NvZnQuY29tIiwidXRpIjoiM2NJYnBXcU9oRWUzaEcxYTFHTGxBQSIsInZlciI6IjEuMCIsIndpZHMiOlsiNjJlOTAzOTQtNjlmNS00MjM3LTkxOTAtMDEyMTc3MTQ1ZTEwIiwiYjc5ZmJmNGQtM2VmOS00Njg5LTgxNDMtNzZiMTk0ZTg1NTA5Il0sInhtc19jYyI6WyJDUDEiXSwieG1zX3NzbSI6IjEiLCJ4bXNfc3QiOnsic3ViIjoicmtlRFN4eUZZamRLcG8taHJRampQd1VpNmJzVzJJUVRWMjBmSDA5WWo4ZyJ9LCJ4bXNfdGNkdCI6MTcxMTcwMjUxN30.i4CLdeGMHDBgm9y2MAOReULE5Yirge_3ye6DhZF9IqBEHFRzhuiMBOYzOqjT6ARUSAHAvOjT61yJ6_bOaZKxzhEw0SVmzsQWC2JzS4oiQ_XnCdZ1OHDvRAKCc_HhxqV30l1kvD9V7Np800esdPCy_xHAjSHC-AYjJetyBaDSu5K8QzbhS-t_UM3m85nhrL0Bze2385QsRwIsUPsbEcvjHVaq-ora3s5zZ5EGGSOFGhmNxtemeZ2SWWSDqCGuOdx2Y_vqCDS5zPFO3TqclIXDwX5_ol_lfxU7a1NW6oHE4Z3UiATbvi3fpCHbf-LlKZ5XikIoqRyal0_JJDaH3ZfPrA"
    headers = {
        'Authorization': 'Bearer ' + access_token,
        'Content-Type': 'application/json'
    }
    # Query to select email ID, subject, and body
    select_query = '$select=id,subject,body'
    # API endpoint to get messages for a specific user
    endpoint = f'https://graph.microsoft.com/v1.0/users/c02d2a89-a76f-40e5-9459-0ff914c29e2f/messages?{select_query}'
    try:
        response = requests.get(endpoint, headers=headers)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f'HTTP error occurred: {http_err}')
        return []
    except Exception as err:
        logging.error(f'An error occurred: {err}')
        return []
    if response.status_code == 200:
        emails = response.json().get('value')
        # Process the fetched emails
        return process_emails(emails)
    else:
        logging.error(f'Failed to fetch emails. Status code: {response.status_code}')
        return []

# --- Flask Route for Prediction and Action ---

@app.route('/predict', methods=['GET', 'POST','PATCH'])
def predict():
    """
    Main API endpoint to fetch, classify, update, and forward emails.
    """
    emails = fetch_emails()
    
    access_token = "eyJ0eXAiOiJKV1QiLCJub25jZSI6ImR5cS1rd1pNZXc4VmdsTWdydktSUExqRHZuZV9xUTg4VW5ETGtoaDA0VTQiLCJhbGciOiJSUzI1NiIsIng1dCI6InEtMjNmYWxldlpoaEQzaG05Q1Fia1A1TVF5VSIsImtpZCI6InEtMjNmYWxldlpoaEQzaG05Q1Fia1A1TVF5VSJ9.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTAwMDAtYzAwMC0wMDAwMDAwMDAwMDAiLCJpc3MiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC8zMTVlMzJhNC1jMTAwLTQ1YTktODUxNS01YWU1M2Y4YjE2MTkvIiwiaWF0IjoxNzEyOTQ3MDQ1LCJuYmYiOjE3MTI5NDcwNDUsImV4cCI6MTcxMzAzMzc0NSwiYWNjdCI6MCwiYWNyIjoiMSIsImFpbyI6IkFUUUF5LzhXQUFBQWdpLzRxODhOME9FRTJYUXVEajFnSkFGcnRpU1krMjJHZTlBeSsrbU55WjQvOGNtdjN3NWFSQ2YzRldBVHBNVXAiLCJhbXIiOlsicHdkIl0sImFwcF9kaXNwbGF5bmFtZSI6IkdyYXBoIEV4cGxvcmVyIiwiYXBwaWQiOiJkZThiYzhiNS1kOWY5LTQ4YjEtYThhZC1iNzQ4ZGE3MjUwNjQiLCJhcHBpZGFjciI6IjAiLCJmYW1pbHlfbmFtZSI6ImxvbGFnZSIsImdpdmVuX25hbWUiOiJvbWthciIsImlkdHlwIjoidXNlciIsImlwYWRkciI6IjExMy4xOTMuNTguMjIwIiwibmFtZSI6Im9ta2FyIGxvbGFnZSIsIm9pZCI6ImMwMmQyYTg5LWE3NmYtNDBlNS05NDU5LTBmZjkxNGMyOWUyZiIsInBsYXRmIjoiMyIsInB1aWQiOiIxMDAzMjAwMzZBRTI3NkY1IiwicmgiOiIwLkFTc0FwREplTVFEQnFVV0ZGVnJsUDRzV0dRTUFBQUFBQUFBQXdBQUFBQUFBQUFEQ0FMcy4iLCJzY3AiOiJBUElDb25uZWN0b3JzLlJlYWQuQWxsIEFQSUNvbm5lY3RvcnMuUmVhZFdyaXRlLkFsbCBNYWlsLlJlYWQgTWFpbC5SZWFkQmFzaWMgTWFpbC5SZWFkV3JpdGUgTWFpbC5TZW5kIE1haWxib3hTZXR0aW5ncy5SZWFkIG9wZW5pZCBwcm9maWxlIFVzZXIuUmVhZCBlbWFpbCIsInN1YiI6Im9GOFlTa2NUZlc4WGRmY192UDYxLVFzeGJnWHZQNmtXRU51dGY3VmZpejgiLCJ0ZW5hbnRfcmVnaW9uX3Njb3BlIjoiQVMiLCJ0aWQiOiIzMTVlMzJhNC1jMTAwLTQ1YTktODUxNS01YWU1M2Y4YjE2MTkiLCJ1bmlxdWVfbmFtZSI6Im9ta2FyTG9sYWdlQGhhY2tvaGlyZS5vbm1pY3Jvc29mdC5jb20iLCJ1cG4iOiJvbWthckxvbGFnZUBoYWNrb2hpcmUub25taWNyb3NvZnQuY29tIiwidXRpIjoiM2NJYnBXcU9oRWUzaEcxYTFHTGxBQSIsInZlciI6IjEuMCIsIndpZHMiOlsiNjJlOTAzOTQtNjlmNS00MjM3LTkxOTAtMDEyMTc3MTQ1ZTEwIiwiYjc5ZmJmNGQtM2VmOS00Njg5LTgxNDMtNzZiMTk0ZTg1NTA5Il0sInhtc19jYyI6WyJDUDEiXSwieG1zX3NzbSI6IjEiLCJ4bXNfc3QiOnsic3ViIjoicmtlRFN4eUZZamRLcG8taHJRampQd1VpNmJzVzJJUVRWMjBmSDA5WWo4ZyJ9LCJ4bXNfdGNkdCI6MTcxMTcwMjUxN30.i4CLdeGMHDBgm9y2MAOReULE5Yirge_3ye6DhZF9IqBEHFRzhuiMBOYzOqjT6ARUSAHAvOjT61yJ6_bOaZKxzhEw0SVmzsQWC2JzS4oiQ_XnCdZ1OHDvRAKCc_HhxqV30l1kvD9V7Np800esdPCy_xHAjSHC-AYjJetyBaDSu5K8QzbhS-t_UM3m85nhrL0Bze2385QsRwIsUPsbEcvjHVaq-ora3s5zZ5EGGSOFGhmNxtemeZ2SWWSDqCGuOdx2Y_vqCDS5zPFO3TqclIXDwX5_ol_lfxU7a1NW6oHE4Z3UiATbvi3fpCHbf-LlKZ5XikIoqRyal0_JJDaH3ZfPrA"
    headers = {
        'Authorization': 'Bearer ' + access_token,
        'Content-Type': 'application/json'
    }

    for email in emails:
        body_text = email['body']
        email_subject = email['subject']
        # Predict the email's category and sentiment
        predicted_category = predict_category(body_text)
        email['category'] = predicted_category
        predicted_sentiment = predict_sentiment(body_text)
        email['sentiment'] = predicted_sentiment
        email_id = email['id']
        # Construct the API endpoint to update the email
        url = f'https://graph.microsoft.com/v1.0/users/c02d2a89-a76f-40e5-9459-0ff914c29e2f/messages/{email_id}'
        # Data payload for the PATCH request to add a category
        data = {
            'subject': email_subject,
            'categories': [predicted_category],
        }

        try:
            # Send a PATCH request to update the email
            response = requests.patch(url, headers=headers, json=data)
            logging.info(f"Response status code: {response.status_code}")
            logging.info(f"Response headers: {response.headers}")
            logging.info(f"Response body: {response.text}")

            # Conditional logic to forward the email based on its predicted category
            if predicted_category == "Finance":
                forward_email(email_id, "aadityapatil@hackohire.onmicrosoft.com", "Aaditya Patil")
            elif predicted_category == "Travel":
                forward_email(email_id, "omkarLolage@hackohire.onmicrosoft.com", "Omkar Lolage")
            elif predicted_category == "Energy":
                forward_email(email_id, "omkarLolage@hackohire.onmicrosoft.com", "Omkar Lolage")
            # This is a logical error, "Pharmaceutical" is misspelled
            elif predicted_category == "Pharmaceuticals":
                forward_email(email_id, "aadityapatil@hackohire.onmicrosoft.com", "Aaditya Patil")
            # This is a redundant check for "Energy"
            elif predicted_category == "Energy":
                forward_email(email_id, "aadityapatil@hackohire.onmicrosoft.com", "Aaditya Patil")

        except requests.exceptions.RequestException as e:
            logging.error(f"An error occurred: {e}")

    return jsonify(emails)

def forward_email(email_id, recipient_email, recipient_name):
    """
    Forwards an email to a specified recipient via Microsoft Graph API.

    Note: The headers have a blank access token, causing this request to fail.
    """
    forward_endpoint = f"https://graph.microsoft.com/v1.0/users/c02d2a89-a76f-40e5-9459-0ff914c29e2f/messages/{email_id}/microsoft.graph.forward"
    print('Forward endpoint:', forward_endpoint)
    request_body = {
        "comment": "FYI",
        "toRecipients": [
            {
                "emailAddress": {
                    "address": recipient_email,
                    "name": recipient_name
                }
            }
        ]
    }
    request_body_json = json.dumps(request_body)
    print('Request body:', request_body_json)

    # Blank Authorization header (will cause 401 Unauthorized)
    headers = {
        'Authorization': 'Bearer ',
        'Content-Type': 'application/json'
    }

    try:
        forward_response = requests.post(forward_endpoint, headers=headers, data=request_body_json)
        # Check for HTTP errors
        forward_response.raise_for_status()
        print("Email forwarded successfully!")

    except requests.exceptions.HTTPError as e:
        try:
            error_response = forward_response.json()
            print('Error response:', error_response)

        except json.JSONDecodeError:
            print("Error response could not be parsed as JSON.")

# Configure basic logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Application Entry Point ---

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True, host='0.0.0.0')
