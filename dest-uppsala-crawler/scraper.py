import requests
from bs4 import BeautifulSoup
import time
import json
import xml.etree.ElementTree as ET
import textwrap

SITEMAP_URL = "https://destinationuppsala.se/sitemap_index.xml"
visited = set()
chunks = []


def get_sitemap_urls(sitemap_url):
    """Fetch all URLs from a sitemap or sitemap index."""
    print(f"Fetching sitemap: {sitemap_url}")
    response = requests.get(sitemap_url, timeout=10)
    response.raise_for_status()

    urls = []
    root = ET.fromstring(response.content)

    # Namespaces (Yoast sitemaps use these)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # Case 1: sitemap index (contains <sitemap> tags)
    for sitemap in root.findall("sm:sitemap", ns):
        loc = sitemap.find("sm:loc", ns).text
        urls.extend(get_sitemap_urls(loc))

    # Case 2: regular sitemap (contains <url> tags)
    for url in root.findall("sm:url", ns):
        loc = url.find("sm:loc", ns).text
        urls.append(loc)

    return urls


def split_text(text, max_length=500):
    """Split text into smaller chunks for LLM ingestion."""
    return textwrap.wrap(text, width=max_length, break_long_words=False, replace_whitespace=False)


def scrape_page(url):
    """Scrape text content from a single page and split into chunks."""
    print(f"Scraping: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed: {url} ({e})")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.title.string.strip() if soup.title else ""

    page_chunks = []
    heading = None
    chunk_id = 0

    # Walk through headings and paragraphs in order
    for element in soup.find_all(["h1", "h2", "h3", "p"]):
        if element.name in ["h1", "h2", "h3"]:
            heading = element.get_text(strip=True)
        elif element.name == "p":
            paragraph = element.get_text(strip=True)
            if paragraph:
                # Split into smaller chunks
                for piece in split_text(paragraph, max_length=500):
                    page_chunks.append({
                        "url": url,
                        "title": title,
                        "heading": heading,
                        "chunk_id": f"{url}#chunk-{chunk_id}",
                        "text": piece
                    })
                    chunk_id += 1

    return page_chunks


if __name__ == "__main__":
    # Step 1: Collect all URLs from sitemap(s)
    all_urls = get_sitemap_urls(SITEMAP_URL)
    print(f"Found {len(all_urls)} URLs in sitemap(s).")

    # Step 2: Scrape pages into chunks
    for url in all_urls:
        if url not in visited:
            visited.add(url)
            page_chunks = scrape_page(url)
            chunks.extend(page_chunks)
            time.sleep(1)  # polite crawling

    # Step 3: Save results
    with open("uppsala_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Scraped {len(chunks)} text chunks. Data saved to uppsala_chunks.json")
