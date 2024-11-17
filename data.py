import pandas as pd
import requests
import time
from tqdm import tqdm

# =========================
# 1. Configuration
# =========================

API_KEY = 'AIzaSyD4efmVx7Ll46x_cBhDyoz3LwWS5Jo1MTI'  # Replace with your actual API key

desired_categories = {
    'Science Fiction': ['Science Fiction', 'Sci-Fi', 'SF', 'Space Opera', 'Cyberpunk'],
    'Fantasy': ['Fantasy', 'Fairy Tale', 'Epic Fantasy', 'High Fantasy'],
    'Mystery': ['Mystery', 'Detective', 'Crime', 'Noir', 'Thriller'],
    'Romance': ['Romance', 'Love Story', 'Heartwarming', 'Contemporary Romance'],
    'Horror': ['Horror', 'Thriller', 'Supernatural'],
    'Biography': ['Biography', 'Biographies', 'Life Story', 'Memoir'],
    'History': ['History', 'Historical', 'Chronicles'],
    'Children': ['Children', 'Juvenile Fiction', 'Young Adult', 'Kids'],
    'Self-Help': ['Self-Help', 'Self Improvement', 'Personal Development', 'Motivational'],
    'Philosophy': ['Philosophy', 'Philosophical', 'Metaphysics'],
    'Travel': ['Travel', 'Travelogue', 'Journey', 'Adventure'],
    'Art': ['Art', 'Fine Arts', 'Visual Arts', 'Painting', 'Sculpture'],
}

books_per_category_google = 500
books_per_category_open_library = 500
output_csv = 'books_data_enhanced.csv'
delay_between_requests = 1  # seconds

# =========================
# 2. Function Definitions
# =========================

def fetch_books_from_google(category, synonyms, max_books=500, api_key=None):
    books = []
    start_index = 0
    total_fetched = 0
    pbar = tqdm(total=max_books, desc=f"Google Books: '{category}'", unit='book')

    query = ' OR '.join([f'intitle:"{term}"' for term in synonyms])

    while total_fetched < max_books:
        params = {
            'q': query,
            'startIndex': start_index,
            'maxResults': 40,
            'printType': 'books',
            'langRestrict': 'en',
            'key': api_key,
        }
        try:
            response = requests.get('https://www.googleapis.com/books/v1/volumes', params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"\nError fetching from Google Books: {e}")
            break

        items = data.get('items', [])
        if not items:
            break

        for item in items:
            volume_info = item.get('volumeInfo', {})
            title = volume_info.get('title', '').strip()
            description = volume_info.get('description', '').strip()
            if not description:
                continue
            books.append({'title': title, 'description': description, 'category': category})
            total_fetched += 1
            pbar.update(1)
            if total_fetched >= max_books:
                break

        start_index += 40
        time.sleep(delay_between_requests)

    pbar.close()
    return books

def fetch_books_from_open_library(category, synonyms, max_books=500):
    books = []
    page = 1
    total_fetched = 0
    pbar = tqdm(total=max_books, desc=f"Open Library: '{category}'", unit='book')

    query = ' OR '.join(synonyms)

    while total_fetched < max_books:
        url = 'https://openlibrary.org/search.json'
        params = {'q': query, 'page': page, 'limit': 100}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"\nError fetching from Open Library: {e}")
            break

        docs = data.get('docs', [])
        if not docs:
            break

        for doc in docs:
            title = doc.get('title', '').strip()
            first_sentence = doc.get('first_sentence')
            subtitle = doc.get('subtitle', '').strip()
            description = ''

            # Handle description extraction logic
            if isinstance(first_sentence, dict):
                description = first_sentence.get('value', '').strip()
            elif isinstance(first_sentence, str):
                description = first_sentence.strip()

            if not description:
                description = subtitle

            if not description:
                continue

            books.append({'title': title, 'description': description, 'category': category})
            total_fetched += 1
            pbar.update(1)
            if total_fetched >= max_books:
                break

        page += 1
        time.sleep(delay_between_requests)

    pbar.close()
    return books

# =========================
# 3. Main Execution
# =========================

def main():
    if API_KEY == 'YOUR_GOOGLE_BOOKS_API_KEY' or not API_KEY.strip():
        print("Error: Invalid Google Books API Key.")
        return

    all_books_data = []

    for category, synonyms in desired_categories.items():
        google_books = fetch_books_from_google(category, synonyms, books_per_category_google, api_key=API_KEY)
        all_books_data.extend(google_books)

        open_library_books = fetch_books_from_open_library(category, synonyms, books_per_category_open_library)
        all_books_data.extend(open_library_books)

    if not all_books_data:
        print("No books fetched.")
        return

    df = pd.DataFrame(all_books_data, columns=['title', 'description', 'category'])
    df.drop_duplicates(subset=['title', 'description'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nData saved to '{output_csv}'.")

if __name__ == "__main__":
    main()
