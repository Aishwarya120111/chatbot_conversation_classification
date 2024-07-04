import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.preprocessing import LabelEncoder
import os
import io

# Load the saved model and tokenizer using absolute paths
model_path = os.path.abspath('src/models/bert_model')
tokenizer_path = os.path.abspath('src/models/bert_tokenizer')

model = BertForSequenceClassification.from_pretrained(model_path, from_tf=False, use_safetensors=True)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Recreate the label encoder (assuming you have the original labels in a dataset)
df = pd.read_excel('data/chatbot_conversation_history.xlsx')
label_encoder = LabelEncoder()
df['Label_encoded'] = label_encoder.fit_transform(df['Label'])

# Define a custom dataset class with optional labels
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, labels=None):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx]
        }
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item

# Function to classify queries
def classify_queries(queries):
    # Tokenize the queries
    encoded_inputs = tokenizer(
        queries, 
        padding=True, 
        truncation=True, 
        return_tensors='pt', 
        max_length=128
    )

    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)

    # Create the dataset
    dataset = CustomDataset(input_ids, attention_mask)

    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

    # Predict labels for the queries
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(batch_predictions)

    # Decode the predicted labels
    predicted_labels = label_encoder.inverse_transform(predictions)
    return predicted_labels

# Streamlit App
st.title("Chatbot Conversations Classification App")
st.write("Upload an Excel file with queries to classify them using a pre-trained BERT model.")

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    # Read the uploaded file
    input_df = pd.read_excel(uploaded_file)
    
    if 'Query' not in input_df.columns:
        st.error("The input file must contain a column named 'Query'.")
    else:
        queries = input_df['Query'].tolist()
        
        # Classify the queries
        predicted_labels = classify_queries(queries)
        
        # Add predicted labels to the dataframe
        input_df['Predicted_Label'] = predicted_labels
        
        # Display the dataframe with predictions
        st.write("### Classified Queries")
        st.dataframe(input_df)
        
        # Convert DataFrame to Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            input_df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.close()
            processed_data = output.getvalue()

        st.download_button(
            label="Download Predictions as Excel",
            data=processed_data,
            file_name="classified_queries.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
