# main.py: Train a BERT-based model on book data

import pandas as pd
import numpy as np  # Import numpy for numerical operations
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. Load and Prepare Data
# =========================

# Load data from the existing BooksDataSet.csv
data = pd.read_csv("BooksDataSet.csv")

# Display the first few rows to verify
print("First few rows of the dataset:")
print(data.head())

# Rename columns to match the expected names in the script
data.rename(columns={
    'book_name': 'title',
    'summary': 'description',
    'genre': 'category'
}, inplace=True)

# Verify the renaming
print("\nColumns after renaming:")
print(data.columns)

# Remove rows with missing or invalid descriptions
data.dropna(subset=['description'], inplace=True)
data = data[data['description'].apply(lambda x: isinstance(x, str))]

# Display the number of books per category before balancing
print("\nNumber of samples per category before balancing:")
print(data['category'].value_counts())

# =========================
# 2. Balancing the Dataset
# =========================

# Define the number of books per category
desired_num_books = 500

# Get the current count of books per category
category_counts = data['category'].value_counts()

# Identify categories with fewer than desired_num_books
categories_to_adjust = category_counts[category_counts < desired_num_books].index.tolist()

# For categories with fewer than desired_num_books, decide on a strategy:
# - Option 1: Combine similar categories
# - Option 2: Accept fewer books
# - Option 3: Fetch additional data using data.py (if applicable)

# For simplicity, we'll proceed with Option 2 and accept fewer books for underrepresented categories

# Sample up to desired_num_books per category
balanced_data = data.groupby('category').apply(lambda x: x.sample(n=min(len(x), desired_num_books), random_state=42)).reset_index(drop=True)

# Display the number of books per category after balancing
print("\nNumber of samples per category after balancing:")
print(balanced_data['category'].value_counts())

# Encode category labels
balanced_data['category_encoded'] = balanced_data['category'].astype('category').cat.codes
num_labels = balanced_data['category_encoded'].nunique()

# =========================
# 3. Train-Validation-Test Split
# =========================

# Split the data into training, validation, and testing sets
train_data, temp_data = train_test_split(
    balanced_data,
    test_size=0.4,
    stratify=balanced_data['category_encoded'],
    random_state=42
)
val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    stratify=temp_data['category_encoded'],
    random_state=42
)

print(f"\nTraining set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Testing set size: {len(test_data)}")

# =========================
# 4. Tokenizer and Model Initialization
# =========================

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize the BERT model for sequence classification
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# =========================
# 5. Data Conversion to TensorFlow Datasets
# =========================

def convert_data_to_tf_dataset(df, tokenizer, max_length=128):
    """
    Converts a pandas DataFrame to a TensorFlow Dataset.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - tokenizer (BertTokenizer): The BERT tokenizer.
    - max_length (int): Maximum length for tokenization.

    Returns:
    - tf.data.Dataset: The converted TensorFlow Dataset.
    """
    input_ids, attention_masks, labels = [], [], []
    for _, row in df.iterrows():
        encoded = tokenizer.encode_plus(
            row['description'],
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        labels.append(row['category_encoded'])

    # Concatenate all inputs
    input_ids = tf.concat(input_ids, axis=0)
    attention_masks = tf.concat(attention_masks, axis=0)
    labels = tf.convert_to_tensor(labels, dtype=tf.int64)
    
    # Create TensorFlow Dataset
    return tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_masks}, labels))

# Convert the datasets
train_dataset = convert_data_to_tf_dataset(train_data, tokenizer).shuffle(1000).batch(8)
val_dataset = convert_data_to_tf_dataset(val_data, tokenizer).batch(8)
test_dataset = convert_data_to_tf_dataset(test_data, tokenizer).batch(8)

# =========================
# 6. Model Compilation
# =========================

# Define optimizer, loss, and metrics
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# =========================
# 7. Training the Model
# =========================

# Define a callback to save the model after each epoch
class CustomSaveModel(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        path = f"book_category_classifier_epoch_{epoch+1}"
        self.model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        print(f"\nModel saved at: {path}")

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    callbacks=[CustomSaveModel()]
)

# =========================
# 8. Save Final Model
# =========================

# Save the final model and tokenizer
model.save_pretrained("book_category_classifier")
tokenizer.save_pretrained("book_category_classifier")

# =========================
# 9. Evaluate Model
# =========================

# Make predictions on the test set
predictions = model.predict(test_dataset)
predicted_labels = tf.argmax(predictions.logits, axis=1).numpy()

# Gather true labels
true_labels = []
for _, labels in test_dataset.unbatch():
    true_labels.append(labels.numpy())
true_labels = np.array(true_labels)

# Print classification report
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=balanced_data['category'].unique()))

# Plot confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=balanced_data['category'].unique(), yticklabels=balanced_data['category'].unique(), cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
