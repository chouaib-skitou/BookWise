import requests
import pandas as pd

# Step 1: Fetch Book Data from Open Library API
def fetch_books_from_openlibrary(query, limit=10):
    url = f"http://openlibrary.org/search.json?q={query}&limit={limit}"
    response = requests.get(url)
    books = []
    
    if response.status_code == 200:
        data = response.json()
        for book in data['docs']:
            if 'title' in book and 'subject' in book:
                # Try to get description from first_sentence or description fields
                description = None
                if isinstance(book.get('first_sentence'), dict):
                    description = book['first_sentence'].get('value')
                elif isinstance(book.get('description'), str):
                    description = book['description']
                
                # Only add if title and category exist
                books.append({
                    'title': book['title'],
                    'description': description or 'No description available',
                    'category': book['subject'][0] if 'subject' in book and book['subject'] else 'Unknown'
                })
    else:
        print(f"Failed to fetch data for {query}")

    return books

# Fetch data for multiple categories
categories = ["fiction", "science", "history", "romance", "mystery"]
book_data = []

for category in categories:
    print(f"Fetching books for category: {category}")
    books = fetch_books_from_openlibrary(category, limit=100)  # Increased limit for more data
    book_data.extend(books)

# Convert to DataFrame
open_library_data = pd.DataFrame(book_data)
open_library_data.to_csv("openlibrary_books_filtered.csv", index=False)
print("Open Library data saved to openlibrary_books_filtered.csv")

# Step 2: Merge with a Secondary Dataset (Assuming 'kaggle_books.csv' exists)
# Load a secondary dataset from Kaggle or another source with descriptions
kaggle_data = pd.read_csv("kaggle_books.csv")  # Replace with your actual Kaggle dataset file
kaggle_data = kaggle_data[['title', 'description']]  # Keep only relevant columns

# Merge Open Library data with Kaggle data on title
merged_data = pd.merge(open_library_data, kaggle_data, on='title', how='left', suffixes=('_ol', '_kaggle'))

# Fill missing Open Library descriptions with Kaggle descriptions
merged_data['description'] = merged_data['description_ol'].combine_first(merged_data['description_kaggle'])
merged_data.drop(columns=['description_ol', 'description_kaggle'], inplace=True)

# Step 3: Filter Out Entries with No Valid Descriptions
filtered_data = merged_data[merged_data['description'] != 'No description available']

# Save the final filtered dataset
filtered_data.to_csv("final_books_with_descriptions.csv", index=False)
print("Filtered dataset saved to final_books_with_descriptions.csv")
