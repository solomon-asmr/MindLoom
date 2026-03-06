import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def scan_links(url):
    """Visit a URL and find all links on the page."""
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")

        base_domain = urlparse(url).netloc

        links = []
        seen = set()

        for a_tag in soup.find_all("a", href=True):
            full_url = urljoin(url, a_tag["href"])
            link_text = a_tag.get_text(strip=True)

            if not full_url.startswith("http"):
                continue

            if full_url in seen:
                continue

            if full_url == url:
                continue

            seen.add(full_url)

            link_domain = urlparse(full_url).netloc
            is_internal = (link_domain == base_domain)

            links.append({
                "url": full_url,
                "title": link_text or full_url,
                "is_internal": is_internal
            })

        return links

    except Exception as e:
        print(f"Error scanning {url}: {e}")
        return []


def scrape_page(url):
    """Download a page and extract clean text content."""
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        return clean_text

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""


def scrape_multiple_pages(urls):
    """Scrape a list of URLs and return their content."""
    results = []

    for url in urls:
        print(f"Scraping: {url}")
        text = scrape_page(url)

        if text:
            results.append({
                "url": url,
                "content": text,
                "char_count": len(text)
            })
            print(f"  ✅ Got {len(text)} characters")
        else:
            results.append({
                "url": url,
                "content": "",
                "char_count": 0
            })
            print(f"  ❌ No content found")

    return results