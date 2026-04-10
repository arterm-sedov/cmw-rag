"""
Web scraping script for GPU and LLM pricing research in Russia
Uses Playwright to scrape Russian retailer and cloud provider websites
Run: python scrape_prices.py
Output: JSON with scraped prices
"""

import json
import re
from playwright.sync_api import sync_playwright
from datetime import datetime

OUTPUT_FILE = "scraped_prices_2026.json"


def scrape_site(url, name, selectors=None):
    """Generic site scraping function"""
    results = {"source": name, "url": url, "prices": [], "timestamp": datetime.now().isoformat()}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto(url, timeout=30000, wait_until="networkidle")
            page.wait_for_timeout(3000)
            content = page.content()

            # Extract all prices in rubles
            prices = re.findall(r"(\d[\d\s]*\d)\s*₽", content)
            results["prices"] = [p.strip() for p in prices[:50]]

            # Extract GPU mentions
            gpus = re.findall(
                r'(RTX\s*\d+|A100|A200|H100|H200|B100|B200|V100)[^<"\n]{0,30}', content
            )
            results["gpu_mentions"] = gpus[:30]

            print(
                f"✓ {name}: {len(results['prices'])} prices, {len(results['gpu_mentions'])} GPU mentions"
            )

        except Exception as e:
            results["error"] = str(e)
            print(f"✗ {name}: ERROR - {e}")

        browser.close()

    return results


def scrape_dns():
    """DNS-shop.ru GPU search"""
    return scrape_site("https://dns-shop.ru/search/?q=видеокарта+RTX", "DNS-shop")


def scrape_1dedic():
    """1dedic.ru GPU servers"""
    return scrape_site("https://1dedic.ru/gpu-servers", "1dedic")


def scrape_selectel():
    """Selectel GPU cloud"""
    return scrape_site("https://selectel.ru/services/cloud/servers/gpu/", "Selectel")


def scrape_cloudru():
    """Cloud.ru pricing"""
    return scrape_site("https://cloud.ru/ru/prices", "Cloud.ru")


def scrape_yandex():
    """Yandex Cloud pricing"""
    return scrape_site("https://cloud.yandex.ru/services/compute/pricing", "Yandex Cloud")


def scrape_regard():
    """Regard.ru GPU search"""
    return scrape_site("https://www.regard.ru/search?query=RTX+4090", "Regard")


def scrape_aitunnel():
    """AITUNNEL API pricing"""
    return scrape_site("https://aitunnel.ru/pricing", "AITUNNEL")


def main():
    print("=" * 60)
    print("Web Scraping: Russian GPU & LLM Prices (April 2026)")
    print("=" * 60)

    all_results = []

    # Scrape all sites
    sites = [
        scrape_1dedic,
        scrape_selectel,
        scrape_cloudru,
        scrape_yandex,
        scrape_regard,
    ]

    for scraper in sites:
        try:
            result = scraper()
            all_results.append(result)
        except Exception as e:
            print(f"Error in {scraper.__name__}: {e}")

    # Save results
    output = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "total_sources": len(all_results),
        },
        "results": all_results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved to {OUTPUT_FILE}")
    return output


if __name__ == "__main__":
    main()
