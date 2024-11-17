# Import necessary libraries
import pandas as pd
import requests
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 1. Verify and Set Compatible Library Versions
# =========================

print("Transformers Version:", tf.__version__)
print("TensorFlow Version:", tf.__version__)

# =========================
# 2. Data Acquisition
# =========================

def fetch_open_library_data(subject, limit=500):
    """
    Fetches book data from Open Library API based on a subject/category.

    Parameters:
    - subject (str): The subject/category to search for.
    - limit (int): Number of books to fetch.

    Returns:
    - DataFrame: A pandas DataFrame containing titles, descriptions, and categories.
    """
    books = []
    page = 1
    fetched = 0
    while fetched < limit:
        url = f'https://openlibrary.org/subjects/{subject.lower()}.json?limit=100&offset={(page-1)*100}'
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch data for subject '{subject}' on page {page}. Status Code: {response.status_code}")
            break
        data = response.json()
        works = data.get('works', [])
        if not works:
            break
        for work in works:
            title = work.get('title', '')
            description = work.get('description', {})
            if isinstance(description, dict):
                desc = description.get('value', '')
            elif isinstance(description, str):
                desc = description
            else:
                desc = ''
            books.append({'title': title, 'description': desc, 'category': subject})
            fetched += 1
            if fetched >= limit:
                break
        page += 1
        time.sleep(1)
    return pd.DataFrame(books)

# List of desired categories
desired_categories = ['Science Fiction', 'Fantasy', 'Mystery', 'Romance',
                      'Horror', 'Biography', 'History', 'Children',
                      'Self-Help', 'Philosophy', 'Travel', 'Art']

# Fetch and merge data
all_books = pd.DataFrame(columns=['title', 'description', 'category'])
for category in desired_categories:
    print(f"Fetching data for category: {category}")
    category_df = fetch_open_library_data(category, limit=500)
    all_books = pd.concat([all_books, category_df], ignore_index=True)
    print(f"Fetched {len(category_df)} books for category '{category}'.")

# =========================
# 3. Data Cleaning and Preparation
# =========================

# Remove duplicates and missing descriptions
all_books.drop_duplicates(subset=['title', 'description'], inplace=True)
all_books.dropna(subset=['description'], inplace=True)
all_books.reset_index(drop=True, inplace=True)

# Display samples per category before balancing
print("\nNumber of samples per category before balancing:")
print(all_books['category'].value_counts())

# Balance dataset
balanced_books = pd.DataFrame(columns=['title', 'description', 'category'])
for category in desired_categories:
    category_df = all_books[all_books['category'] == category]
    if len(category_df) >= 500:
        sampled_df = category_df.sample(n=500, random_state=42)
    else:
        sampled_df = category_df
    balanced_books = pd.concat([balanced_books, sampled_df], ignore_index=True)

# Display samples per category after balancing
print("\nNumber of samples per category after balancing:")
print(balanced_books['category'].value_counts())

# Encode category labels
balanced_books['category_encoded'] = balanced_books['category'].astype('category').cat.codes
num_labels = balanced_books['category_encoded'].nunique()

# =========================
# 4. Train-Validation-Test Split
# =========================

train_data, temp_data = train_test_split(
    balanced_books,
    test_size=0.4,
    stratify=balanced_books['category_encoded'],
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
# 5. Tokenizer and Model Initialization
# =========================

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# =========================
# 6. Data Conversion to TensorFlow Datasets
# =========================

def convert_data_to_tf_dataset(df, tokenizer, max_length=128):
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

    input_ids = tf.concat(input_ids, axis=0)
    attention_masks = tf.concat(attention_masks, axis=0)
    labels = tf.convert_to_tensor(labels, dtype=tf.int64)
    return tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_masks}, labels))

train_dataset = convert_data_to_tf_dataset(train_data, tokenizer).shuffle(1000).batch(8)
val_dataset = convert_data_to_tf_dataset(val_data, tokenizer).batch(8)
test_dataset = convert_data_to_tf_dataset(test_data, tokenizer).batch(8)

# =========================
# 7. Model Compilation
# =========================

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# =========================
# 8. Training the Model
# =========================

class CustomSaveModel(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        path = f"book_category_classifier_epoch_{epoch+1}"
        self.model.save_pretrained(path)
        print(f"\nModel saved at: {path}")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    callbacks=[CustomSaveModel()]
)

# =========================
# 9. Saving the Model
# =========================

model.save_pretrained("book_category_classifier")
tokenizer.save_pretrained("book_category_classifier")

# =========================
# 10. Detailed Evaluation Metrics
# =========================

predictions = model.predict(test_dataset)
predicted_labels = tf.argmax(predictions.logits, axis=1).numpy()

true_labels = []
for _, labels in test_dataset.unbatch():
    true_labels.append(labels.numpy())
true_labels = np.array(true_labels)

print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=desired_categories))

cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=desired_categories, yticklabels=desired_categories, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

try:
    roc_auc = roc_auc_score(
        tf.keras.utils.to_categorical(true_labels, num_classes=num_labels),
        tf.nn.softmax(predictions.logits),
        multi_class='ovr'
    )
    print(f"ROC AUC Score: {roc_auc:.4f}")
except Exception as e:
    print(f"ROC AUC Score could not be computed: {e}")

print("\nModel training complete, evaluated, and saved successfully.")
