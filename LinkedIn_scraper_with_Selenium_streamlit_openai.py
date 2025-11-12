import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# ---------------------- Setup Chrome WebDriver ---------------------- #
def webdriver_setup():
    """Setup Chrome WebDriver using webdriver-manager (auto downloads)."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # comment this if you want to see the browser
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.maximize_window()
    return driver


# ---------------------- Build LinkedIn Search URL ---------------------- #
def build_url(job_titles, job_location):
    """Build LinkedIn job search URL."""
    job_titles_encoded = ["%20".join(title.strip().split()) for title in job_titles]
    title_query = "%2C%20".join(job_titles_encoded)
    return f"https://in.linkedin.com/jobs/search?keywords={title_query}&location={job_location}&geoId=102713980"


# ---------------------- Open and Wait for Page ---------------------- #
def open_link(driver, link):
    """Open LinkedIn job search page and wait for load."""
    while True:
        try:
            driver.get(link)
            driver.implicitly_wait(5)
            time.sleep(3)
            driver.find_element(By.TAG_NAME, "body")
            return
        except NoSuchElementException:
            continue


# ---------------------- Scroll and Collect Job Cards ---------------------- #
def scroll_and_collect(driver, scroll_times=5):
    """Scroll page down to load job listings."""
    for _ in range(scroll_times):
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.PAGE_DOWN)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)


# ---------------------- Scrape Company Data ---------------------- #
def scrap_company_data(driver):
    """Scrape company, title, location, and job URLs."""
    companies = [i.text.strip() for i in driver.find_elements(By.CSS_SELECTOR, "h4.base-search-card__subtitle")]
    titles = [i.text.strip() for i in driver.find_elements(By.CSS_SELECTOR, "h3.base-search-card__title")]
    locations = [i.text.strip() for i in driver.find_elements(By.CSS_SELECTOR, "span.job-search-card__location")]
    urls = [i.get_attribute("href") for i in driver.find_elements(By.XPATH, '//a[contains(@href, "/jobs/")]')]

    min_len = min(len(companies), len(titles), len(locations), len(urls))
    if min_len == 0:
        print("⚠️ No job cards found on page. Try different keywords or location.")
        return pd.DataFrame(columns=["Company", "Job Title", "Location", "URL"])

    df = pd.DataFrame({
        "Company": companies[:min_len],
        "Job Title": titles[:min_len],
        "Location": locations[:min_len],
        "URL": urls[:min_len]
    })
    return df


# ---------------------- Scrape Job Descriptions ---------------------- #
def scrap_job_description(driver, df, job_count):
    """Scrape job descriptions safely, keeping lengths aligned."""
    df = df.head(job_count).copy()
    descriptions = []

    for idx, link in enumerate(df["URL"].tolist(), start=1):
        print(f"🔍 Scraping job {idx}/{len(df)}...")
        try:
            driver.get(link)
            driver.implicitly_wait(5)
            time.sleep(2)

            # Try clicking 'See more' if present
            try:
                btn = driver.find_elements(
                    By.CSS_SELECTOR, 'button[data-tracking-control-name="public_jobs_show-more-html-btn"]'
                )
                if btn:
                    btn[0].click()
                    time.sleep(1)
            except Exception:
                pass

            desc_elem = driver.find_elements(By.CSS_SELECTOR, "div.show-more-less-html__markup")
            descriptions.append(desc_elem[0].text if desc_elem else "Not Available")

        except Exception as e:
            print(f"⚠️ Error scraping job {idx}: {e}")
            descriptions.append("Not Available")

    df["Job Description"] = descriptions
    return df


# ---------------------- MAIN FUNCTION ---------------------- #
def main():
    print("\n💼 LINKEDIN JOB SCRAPER 💼\n")

    job_titles_input = input("Enter job titles (comma-separated): ").split(",")
    job_titles = [t.strip() for t in job_titles_input if t.strip()]
    job_location = input("Enter job location (e.g., India): ").strip() or "India"
    job_count = int(input("Enter number of jobs to scrape: "))

    if not job_titles:
        print("❌ Please enter at least one job title.")
        return

    driver = webdriver_setup()

    try:
        print("\n🚀 Opening LinkedIn...")
        url = build_url(job_titles, job_location)
        open_link(driver, url)
        scroll_and_collect(driver, scroll_times=6)

        print("\n📊 Collecting company and job data...")
        df = scrap_company_data(driver)

        print("\n📝 Collecting job descriptions...")
        df_final = scrap_job_description(driver, df, job_count)

        print("\n✅ Scraping complete!")
        print(df_final.head())

        # Export results
        save_path = "linkedin_jobs.csv"
        df_final.to_csv(save_path, index=False)
        print(f"\n💾 Data saved to: {save_path}")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
    finally:
        driver.quit()
        print("\n👋 Browser closed.")


# ---------------------- RUN SCRIPT ---------------------- #
if __name__ == "__main__":
    main()
