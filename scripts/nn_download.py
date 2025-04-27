import requests
from bs4 import BeautifulSoup
import time
import os
import re
import json
from urllib.parse import urljoin

# --- Configuration ---
BASE_LIST_URL = 'https://www.nature.com/neuro/research-articles'
SAVE_DIR = 'nn_pdfs_oa'
DB_FILE = 'nn_downloaded_pdfs.json'
LAST_PAGE_FILE = 'nn_last_page.txt'
MAX_PAGES = 5  # Set None for no limit
DOWNLOAD_DELAY_SECONDS = 2  # polite delay between articles
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
# --- End Configuration ---

# Create save directory if needed
os.makedirs(SAVE_DIR, exist_ok=True)

# Load existing database
if os.path.exists(DB_FILE):
    with open(DB_FILE, 'r', encoding='utf-8') as f:
        downloaded_db = json.load(f)
else:
    downloaded_db = {}

# Load last page info
if os.path.exists(LAST_PAGE_FILE):
    with open(LAST_PAGE_FILE, 'r') as f:
        start_page = int(f.read().strip())
    print(f"Resuming from page {start_page}.")
else:
    start_page = 1

# Filename sanitizer
def sanitize_filename(filename):
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    sanitized = sanitized.replace(' ', '_')
    return sanitized[:200]

# Get article details
def get_article_details(article_url):
    try:
        print(f"  Checking article: {article_url}")
        response = requests.get(article_url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        open_access_tag = soup.find('a', attrs={'data-test': 'open-access'})
        if not (open_access_tag and 'open access' in open_access_tag.get_text(strip=True).lower()):
            print("    Article is not Open Access.")
            return None, None

        print("    Article detected as Open Access.")

        pdf_link_tag = soup.find('a', attrs={'data-test': 'download-pdf'})
        if pdf_link_tag and pdf_link_tag.get('href'):
            pdf_url = urljoin(article_url, pdf_link_tag['href'])

            title_tag = soup.find('h1', class_='c-article-title')
            article_title = title_tag.get_text(strip=True) if title_tag else "untitled_article"
            safe_filename_base = sanitize_filename(article_title)

            pdf_name_match = re.search(r'/([^/]+\.pdf)$', pdf_url, re.IGNORECASE)
            unique_pdf_part = pdf_name_match.group(1) if pdf_name_match else pdf_url.split('/')[-1]

            filename = f"{safe_filename_base}_{unique_pdf_part}"
            if not filename.lower().endswith('.pdf'):
                filename += '.pdf'

            return pdf_url, filename
        else:
            print("    OA article found, but PDF link not located.")
            return None, None

    except Exception as e:
        print(f"    Error processing article: {e}")
        return None, None

# --- Main Scraping Loop ---
page_url = f"{BASE_LIST_URL}?page={start_page}"
page_count = 0

while True:
    if MAX_PAGES is not None and page_count >= MAX_PAGES:
        print(f"\nReached maximum page limit ({MAX_PAGES}). Stopping.")
        break

    print(f"\n--- Scraping: {page_url} ---")
    try:
        response = requests.get(page_url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find articles
        article_tags = soup.find_all('article', class_='u-full-height')
        page_article_links = []
        for article in article_tags:
            title_tag = article.find('h3', class_='c-card__title')
            if title_tag:
                link_tag = title_tag.find('a', href=True)
                if link_tag:
                    article_href = link_tag['href']
                    full_link = urljoin(BASE_LIST_URL, article_href)
                    page_article_links.append(full_link)

        if not page_article_links:
            print("No articles found. Assuming end of results.")
            break

        print(f"Found {len(page_article_links)} articles.")

        # Process each article
        for link in page_article_links:
            if link in downloaded_db:
                print(f"    Already downloaded: {link}. Skipping.")
                continue

            pdf_url, filename = get_article_details(link)

            if pdf_url and filename:
                filepath = os.path.join(SAVE_DIR, filename)
                try:
                    print(f"  Downloading: {filename}")
                    pdf_response = requests.get(pdf_url, headers=HEADERS, timeout=60, stream=True)
                    pdf_response.raise_for_status()

                    content_type = pdf_response.headers.get('Content-Type', '').lower()
                    if 'application/pdf' not in content_type:
                        print(f"    Warning: Expected PDF, got {content_type}. Skipping.")
                        continue

                    with open(filepath, 'wb') as f:
                        for chunk in pdf_response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    print(f"    Saved: {filename}")

                    # Update DB
                    downloaded_db[link] = {
                        'filename': filename,
                        'pdf_url': pdf_url,
                        'downloaded_at': time.strftime('%Y-%m-%d %H:%M:%S')
                    }

                    # Save DB immediately
                    with open(DB_FILE, 'w', encoding='utf-8') as f:
                        json.dump(downloaded_db, f, indent=2)

                except Exception as e:
                    print(f"    Error saving PDF: {e}")

            time.sleep(DOWNLOAD_DELAY_SECONDS)

        # Update Last Page
        current_page_match = re.search(r'page=(\d+)', page_url)
        if current_page_match:
            current_page = int(current_page_match.group(1)) + 1
        else:
            current_page = start_page + page_count + 1

        with open(LAST_PAGE_FILE, 'w') as f:
            f.write(str(current_page))

        page_count += 1

        # Find Next Page link
        next_page_li = soup.find('li', attrs={'data-test': 'page-next'})
        if next_page_li:
            next_page_a = next_page_li.find('a', href=True)
            if next_page_a:
                page_url = urljoin(BASE_LIST_URL, next_page_a['href'])
                print(f"--- Delaying {DOWNLOAD_DELAY_SECONDS * 2}s before next page ---")
                time.sleep(DOWNLOAD_DELAY_SECONDS * 2)
                continue
            else:
                print("\nNo 'Next' link found. Reached the end.")
                break
        else:
            print("\nNo 'Next' link found. Reached the end.")
            break

    except Exception as e:
        print(f"Error fetching page: {e}")
        break

print("\n--- Scraping finished ---")
