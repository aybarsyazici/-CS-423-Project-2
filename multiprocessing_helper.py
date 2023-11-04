import requests
from bs4 import BeautifulSoup
import re
import time
from tqdm import tqdm
# Get the plot summary from IMDB
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def get_plot_summary(link):
    # Get the themoviedb.org/movie/ link
    tmdb_link = 'https://www.themoviedb.org/movie/' + link
    #print(tmdb_link)
    while True:
        try:
            # Get the html
            html = requests.get(tmdb_link, headers=headers)
            # Parse the html
            soup = BeautifulSoup(html.text, 'html.parser')
            #print(soup)
            # Find the plot summary
            plot_summary = soup.find('div', class_='overview')
            # Remove the extra spaces and newlines
            plot_summary = plot_summary.text.strip()
            # Remove the extra spaces
            plot_summary = re.sub(' +', ' ', plot_summary)
            return plot_summary
        except:
            # We might have been blocked
            # Sleep a bit and try again
            time.sleep(5)

def get_plot_summary_chunk(chunk):
    results = []
    progress = 0
    print(f'Getting plot summaries for {len(chunk)} movies')
    for link in chunk:
        results.append(get_plot_summary(link))
        progress += 1
        # print at every 10% progress
        if progress % (len(chunk) // 10) == 0:
            print(f'{progress / len(chunk) * 100:.0f}% done')
    return results