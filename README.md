# Chatbot Conversations Classification App

This Streamlit application allows users to upload an Excel file containing chatbot queries, classify them using a pre-trained BERT model, and download the predictions in an Excel file.

## Features

- Upload an Excel file with queries.
- Classify queries using a pre-trained BERT model.
- Display the classified queries.
- Download the results as an Excel file.

## Installation

1. Clone the repository:

```sh
git clone <repository_url>
cd label_conversations
```

2. Create Virtual Environment:

```sh
conda create -p venv python==3.10 -y
conda activate venv/
```

3. Activate the environment:

```sh
pip install -r requirements.txt
```

## Running Application

1. Navigate to project directory

```sh
cd src
```

2. Run the streamlit application

```sh
streamlit run bertmodel.py
```

3. Open the provided URL in your web browser to interact with the application.

## Application Usage

1. Upload File: Use the file uploader to upload an Excel file containing queries. The file must contain a column named Query.

2. Classify Queries: The application will classify the queries using the pre-trained BERT model and display the results.

3. Download Predictions: Click the "Download Predictions as Excel" button to download the classified queries in an Excel file.

## Model and Data

1. **BERT Model:** The pre-trained BERT model and tokenizer are saved in the `src/models/bert_model` and `src/models/bert_tokenizer` directories, respectively.
2. **Training Data:** The `chatbot_conversation_history.xlsx` file contains the original training data with labels.

## License
This project is licensed under the MIT License.
