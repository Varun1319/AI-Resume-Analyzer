# jobs/linkedin.py
"""
Simple public LinkedIn Jobs scraper for demo purposes.

- Uses only the public jobs search (no login).
- Be gentle with job_count and usage to avoid hitting limits.
"""
from __future__ import annotations

import time
from typing import List
from urllib.parse import quote_plus

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, WebDriverException


class DriverStartError(RuntimeError):
    pass


def webdriver_setup() -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1440,900")
    # Slightly reduce logging noise
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    try:
        return webdriver.Chrome(options=options)
    except WebDriverException as e:
        raise DriverStartError(
            "Failed to start Chrome WebDriver. Make sure ChromeDriver is installed and on PATH."
        ) from e


def build_search_url(job_title: str, job_location: str, country_domain: str = "in") -> str:
    base = f"https://{country_domain}.linkedin.com/jobs/search"
    params = f"?keywords={quote_plus(job_title)}&location={quote_plus(job_location)}"
    return base + params


def _scroll_until_enough_jobs(driver, max_scrolls: int, min_jobs: int) -> None:
    """Scroll the page down multiple times until we have at least `min_jobs`
    job cards or we hit `max_scrolls`.
    """
    body = driver.find_element(By.TAG_NAME, "body")
    last_len = 0

    for _ in range(max_scrolls):
        body.send_keys(Keys.END)
        time.sleep(1.5)  # gentle wait for new cards to load

        cards = driver.find_elements(By.CSS_SELECTOR, "div.base-card")
        curr_len = len(cards)
        if curr_len >= min_jobs:
            break
        # If nothing new is loading, stop early
        if curr_len == last_len:
            break
        last_len = curr_len


def get_jobs(job_title: str, job_location: str, job_count: int, country_domain: str = "in") -> pd.DataFrame:
    """Company, title, location and URL of up to `job_count` jobs, de-duplicated by URL."""
    driver = webdriver_setup()
    url = build_search_url(job_title, job_location, country_domain)

    try:
        driver.get(url)

        # Wait until at least one job card is visible or timeout
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.base-card"))
        )

        # Scroll until we have enough jobs or hit scroll limit
        _scroll_until_enough_jobs(
            driver,
            max_scrolls=10,          # be gentle; can tweak
            min_jobs=job_count * 2,   # overshoot a bit, we dedupe later
        )

        # Extract job info
        cards = driver.find_elements(By.CSS_SELECTOR, "div.base-card")
        companies: List[str] = []
        titles: List[str] = []
        locations: List[str] = []
        urls: List[str] = []

        for card in cards:
            try:
                company_el = card.find_element(By.CSS_SELECTOR, "h4.base-search-card__subtitle")
                title_el = card.find_element(By.CSS_SELECTOR, "h3.base-search-card__title")
                loc_el = card.find_element(By.CSS_SELECTOR, "span.job-search-card__location")
                link_el = card.find_element(By.CSS_SELECTOR, "a.base-card__full-link")
            except NoSuchElementException:
                continue  # skip malformed cards

            companies.append(company_el.text.strip())
            titles.append(title_el.text.strip())
            locations.append(loc_el.text.strip())
            urls.append(link_el.get_attribute("href"))

        # Deduplicate by URL while preserving order
        seen = set()
        unique_rows = []
        for c, t, l, u in zip(companies, titles, locations, urls):
            if u in seen:
                continue
            seen.add(u)
            unique_rows.append((c, t, l, u))

        df = pd.DataFrame(unique_rows, columns=["Company", "Job Title", "Location", "URL"])
        return df.head(job_count)

    finally:
        driver.quit()
