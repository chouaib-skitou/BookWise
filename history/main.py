# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Load the dataset
data = pd.read_csv("book_descriptions_corrected.csv")

# Encode labels as integers
data['category'] = data['category'].astype('category').cat.codes
num_labels = data['category'].nunique()

# Split the dataset into 60% training and 40% temporary (which will be split further)
train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)

# Split the temporary set into 20% validation and 20% testing
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Display the sizes of each set
print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Testing set size: {len(test_data)}")

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Helper function to convert data to TensorFlow dataset
def convert_data_to_tf_dataset(df, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []
    labels = []

    for _, row in df.iterrows():
        encoded_dict = tokenizer.encode_plus(
            row['description'],
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(row['category'])

    # Convert lists to tensors
    input_ids = tf.concat(input_ids, axis=0)
    attention_masks = tf.concat(attention_masks, axis=0)
    labels = tf.convert_to_tensor(labels, dtype=tf.int64)

    return tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_masks}, labels))

# Convert the data to TensorFlow datasets
train_dataset = convert_data_to_tf_dataset(train_data, tokenizer).shuffle(100).batch(8)
val_dataset = convert_data_to_tf_dataset(val_data, tokenizer).batch(8)
test_dataset = convert_data_to_tf_dataset(test_data, tokenizer).batch(8)

# Compile the model with a specific loss function and optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model with validation data
history = model.fit(train_dataset, validation_data=val_dataset, epochs=2)

# Save the trained model
model.save_pretrained("book_category_classifier")
tokenizer.save_pretrained("book_category_classifier")

# Final evaluation on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("Model training complete and saved.")
