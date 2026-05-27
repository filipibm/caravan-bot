"""Entry point: crawl seed URLs, follow same-domain links, chunk, embed, store."""
import asyncio
import sys
from urllib.parse import urljoin, urlparse

from playwright.async_api import Browser, async_playwright

from .chunker import chunk
from .embedder import embed_batch
from .store import upsert

# Seed URLs — the crawler starts here and follows every same-domain link it finds
DEFAULT_URLS = [
    "https://www.teknikkaravan.com.tr/menudetay-3-karavanlar.html",
]

BATCH_SIZE = 32


SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".pdf", ".zip", ".rar", ".mp4", ".mp3", ".avi", ".mov",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
}


def same_domain(base: str, href: str) -> bool:
    return urlparse(href).netloc == urlparse(base).netloc


def normalize(base: str, href: str) -> str | None:
    """Resolve relative href, strip fragments/query strings, skip non-HTML URLs."""
    url = urljoin(base, href).split("#")[0].split("?")[0]
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None
    ext = "." + parsed.path.rsplit(".", 1)[-1].lower() if "." in parsed.path.rsplit("/", 1)[-1] else ""
    if ext in SKIP_EXTENSIONS:
        return None
    return url


IMAGE_URL_PATTERNS = ("/upload/Big/", "/upload/Medium/")


def is_product_image(src: str) -> bool:
    return any(p in src for p in IMAGE_URL_PATTERNS)


async def get_page_data(browser: Browser, url: str) -> tuple[str, list[str], list[str]]:
    """Return (body text, hrefs, absolute image URLs) for a single page."""
    page = await browser.new_page()
    try:
        await page.goto(url, wait_until="networkidle", timeout=30_000)
        text = await page.inner_text("body")
        hrefs = await page.eval_on_selector_all(
            "a[href]", "els => els.map(e => e.getAttribute('href'))"
        )
        srcs = await page.eval_on_selector_all(
            "img[src]", "els => els.map(e => e.src)"
        )
    finally:
        await page.close()

    images = list(dict.fromkeys(s for s in srcs if is_product_image(s)))
    return text, hrefs, images


async def crawl(seed_urls: list[str]) -> None:
    visited: set[str] = set()
    queue: list[str] = list(seed_urls)

    async with async_playwright() as p:
        browser = await p.chromium.launch()

        while queue:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            print(f"[{len(visited)}] Scraping {url}")
            try:
                text, hrefs, images = await get_page_data(browser, url)
            except Exception as exc:
                print(f"  SKIP — {exc}")
                continue

            # Enqueue discovered same-domain links
            for href in hrefs:
                absolute = normalize(url, href)
                if absolute and same_domain(url, absolute) and absolute not in visited:
                    queue.append(absolute)

            # Chunk → embed → store
            chunks = chunk(text)
            if not chunks:
                continue

            if images:
                print(f"  {len(images)} images found")
            print(f"  {len(chunks)} chunks — embedding…")
            vectors: list[list[float]] = []
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i : i + BATCH_SIZE]
                vectors.extend(embed_batch(batch))

            upsert(chunks, vectors, source_url=url, images=images)
            print(f"  stored {len(chunks)} points")

        await browser.close()

    print(f"\nDone. {len(visited)} pages crawled.")


async def main() -> None:
    seeds = sys.argv[1:] or DEFAULT_URLS
    await crawl(seeds)


if __name__ == "__main__":
    asyncio.run(main())
