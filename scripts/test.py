from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

def debug_motley_fool():
    """
    Debug script to see what's actually on the page
    """
    print("Setting up Chrome driver...")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    try:
        url = "https://www.fool.com/earnings-call-transcripts/"
        print(f"\nOpening: {url}")
        driver.get(url)
        
        print("Waiting for page to load...")
        time.sleep(5)
        
        # Save page source to inspect
        with open('page_source.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        print("✓ Saved page source to page_source.html")
        
        # Find ALL links
        all_links = driver.find_elements(By.TAG_NAME, 'a')
        print(f"\nTotal links found: {len(all_links)}")
        
        # Print first 30 links with their text
        print("\nFirst 30 links:")
        for i, link in enumerate(all_links[:30], 1):
            href = link.get_attribute('href')
            text = link.text.strip()[:60]
            print(f"{i}. [{text}] -> {href}")
        
        # Look for specific patterns
        print("\n" + "="*60)
        print("Looking for transcript-like links...")
        print("="*60)
        
        transcript_patterns = [
            'transcript',
            'earnings',
            'q1-', 'q2-', 'q3-', 'q4-',
            '2024', '2025'
        ]
        
        potential_transcripts = []
        for link in all_links:
            href = link.get_attribute('href')
            text = link.text.strip().lower()
            
            if href:
                for pattern in transcript_patterns:
                    if pattern in href.lower() or pattern in text:
                        if href not in potential_transcripts and '#' not in href:
                            potential_transcripts.append((text[:60], href))
                        break
        
        print(f"\nFound {len(potential_transcripts)} potential transcript links:")
        for i, (text, href) in enumerate(potential_transcripts[:20], 1):
            print(f"{i}. [{text}] -> {href}")
        
    finally:
        print("\nClosing browser...")
        driver.quit()

if __name__ == "__main__":
    debug_motley_fool()