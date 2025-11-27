from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

def scrape_motley_fool_transcripts(num_transcripts=15):
    """
    Scrape earnings call transcripts from Motley Fool using Selenium
    """
    os.makedirs('transcripts', exist_ok=True)
    
    print("Setting up Chrome driver...")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    try:
        # Go to the transcripts page
        url = "https://www.fool.com/earnings-call-transcripts/"
        print(f"\nOpening: {url}")
        driver.get(url)
        
        print("Waiting for page to load...")
        time.sleep(5)
        
        # Find all links
        all_links = driver.find_elements(By.TAG_NAME, 'a')
        
        # Filter for actual transcript URLs
        transcript_urls = []
        for link in all_links:
            href = link.get_attribute('href')
            if href and '/earnings/call-transcripts/' in href and href not in transcript_urls:
                # Skip the base URL and anchor links
                if href != url and '#' not in href:
                    transcript_urls.append(  )
        
        print(f"\nFound {len(transcript_urls)} transcript links")
        print(f"Will fetch first {min(num_transcripts, len(transcript_urls))} transcripts\n")
        
        # Fetch each transcript
        count = 0
        for transcript_url in transcript_urls[:num_transcripts]:
            print(f"[{count+1}/{min(num_transcripts, len(transcript_urls))}] Fetching: {transcript_url}")
            
            try:
                driver.get(transcript_url)
                time.sleep(3)  # Wait for page to load
                
                # Get title
                try:
                    title_element = driver.find_element(By.TAG_NAME, 'h1')
                    title = title_element.text.strip()
                except:
                    title = f"Transcript {count+1}"
                
                print(f"  Title: {title}")
                
                # Get article content - try multiple selectors
                content = None
                selectors = [
                    (By.TAG_NAME, 'article'),
                    (By.CLASS_NAME, 'article-body'),
                    (By.CLASS_NAME, 'tailwind-article-body'),
                ]
                
                for by, selector in selectors:
                    try:
                        article_element = driver.find_element(by, selector)
                        content = article_element.text
                        if len(content) > 500:
                            break
                    except:
                        continue
                
                # If still no content, try getting all paragraphs
                if not content or len(content) < 500:
                    try:
                        paragraphs = driver.find_elements(By.TAG_NAME, 'p')
                        content = '\n\n'.join([p.text for p in paragraphs if len(p.text) > 50])
                    except:
                        pass
                
                if content and len(content) > 500:
                    # Save to file
                    filename = f"transcripts/{title.replace(' ', '_')}_{count+1:02d}.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"Title: {title}\n")
                        f.write(f"Source: {transcript_url}\n")
                        f.write(f"Date: {time.strftime('%Y-%m-%d')}\n\n")
                        f.write(content)
                    
                    count += 1
                    print(f"  ✓ Saved ({len(content):,} characters)\n")
                else:
                    print(f"  ✗ Content too short or not found\n")
                
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
            
            time.sleep(2)  # Be polite to the server
        
        print(f"{'='*60}")
        print(f"✓ Successfully saved {count} transcripts to ./transcripts/")
        print(f"{'='*60}")
        
    finally:
        print("\nClosing browser...")
        driver.quit()

if __name__ == "__main__":
    scrape_motley_fool_transcripts(num_transcripts=15)