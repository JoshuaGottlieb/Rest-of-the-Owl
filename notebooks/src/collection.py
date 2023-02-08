import requests
from bs4 import BeautifulSoup
import os
from PIL import Image
import time

def check_directory_existence(directory):
    '''
    Helper function to check if a directory exists and to create the directory if it does not exist.
    
    directory: string, file path of directory to check the existence of.
               All parent directories of target directory must exist.
    '''
    
    print(f'Checking if {directory} exists.')
    
    if not os.path.isdir(directory):
        print(f'Creating {directory}.')
        
        os.mkdir(directory)
        
        print(f'{directory} created.')
    else:
        print(f'{directory} exists.')
        
    return

def scrape_vectorstock(directory, start_page, end_page):
    '''
    Function to scrape vectorstock.com owl vector sketches.
    
    directory: string, file path of directory to save images to.
    start_page: int, starting page to scrape from vectorstock website.
    end_page: int, ending page to scrape from vectorstock website.
    '''
    
    # Set base url
    url = 'https://www.vectorstock.com/royalty-free-vectors/owl-sketch-vectors'
    
    # Check if the target directory exists, create it if it does not
    check_directory_existence(directory)
    
    # Create subdirectory if it does not exist
    directory_tail = f'Page_{start_page:02d}-{end_page:02d}'
    subdirectory = directory + '/' + directory_tail
    
    check_directory_existence(subdirectory)
    
    page_range = range(start_page, end_page + 1)
    
    # For each page, scrape the page using BeautifulSoup
    for page in page_range:
        time.sleep(0.1)
        
        # Set url tail to scrape specific page
        url_tail = f'-page_{page}'
        print(f'Scraping page {page}')
        
        response = requests.get(url + url_tail)
        soup = BeautifulSoup(response.text)
        
        # All images are wrapped in img tags
        image_tags = soup.find_all('img')
        
        # Write each image to disk
        for index, image in enumerate(image_tags):
            time.sleep(0.1)
            
            print(f'Saving image {index + 1} of {len(image_tags)} to {subdirectory}.')
            
            img = Image.open(requests.get(image['src'], stream = True).raw)
            img_name = f'VectorStock_Page_{page:02d}_Image_{index + 1:03d}.{img.format}'
            img.save(subdirectory + '/' + img_name)
        
    return

def scrape_adobe(directory, start_page, end_page, subcategory_url_string, subcategory_title):
    '''
    Function to scrape stock.adobe.com owl sketches.
    
    directory: string, file path of directory to save images to.
    start_page: int, starting page to scrape from vectorstock website.
    end_page: int, ending page to scrape from vectorstock website.
    subcategory_url_string: string, denoting which subcategory of stock.adobe to scrape (found using filters on website).
    subcategory_title: string, used to name the saved image files, preferably title-cased with no spaces.
    '''
    
    # Set base url
    url = f'https://stock.adobe.com/search/images?&k=owl+{subcategory_url_string}'

    # Check if the target directory exists, create it if it does not
    check_directory_existence(directory)
    
    # Create subdirectory if it does not exist
    directory_tail = f'Page_{start_page:03d}-{end_page:03d}'
    subdirectory = directory + '/' + directory_tail
    
    check_directory_existence(subdirectory)
    
    page_range = range(start_page, end_page + 1)
    
    # For each page, scrape the page using BeautifulSoup
    for page in page_range:
        time.sleep(0.1)
        
        # Set url tail to scrape specific page
        url_tail = f'&search_page={page}'
        print(f'Scraping page {page}')
        
        response = requests.get(url + url_tail)
        soup = BeautifulSoup(response.text)
        
        # All images are wrapped in img tags
        image_tags = soup.find_all('img')
        
        # Some images are site logos - extract only those that are useful images
        image_tags_cleaned = [x for x in image_tags if 'data-lazy' in\
                              x.attrs.keys() or x['src'].endswith('.jpg')]
        
        # Write each image to disk
        for index, image in enumerate(image_tags_cleaned):
            time.sleep(0.1)
            
            print(f'Saving image {index + 1} of {len(image_tags_cleaned)} to {subdirectory}.')
            
            if 'data-lazy' in image.attrs.keys():
                img = Image.open(requests.get(image['data-lazy'], stream = True).raw)
            elif image['src'].endswith('.jpg'):
                img = Image.open(requests.get(image['src'], stream = True).raw)
                
            img_name = f'AdobeStock{subcategory_title}_Page_{page:03d}_Image_{index + 1:03d}.{img.format}'
            img.save(subdirectory + '/' + img_name)
        
    return

def scrape_fineartamerica(directory, start_page, end_page, subcategory_url_string, subcategory_title):
    '''
    Function to scrape fineartarmerica.com owl sketches.
    
    directory: string, file path of directory to save images to.
    start_page: int, starting page to scrape from vectorstock website.
    end_page: int, ending page to scrape from vectorstock website.
    subcategory_url_string: string, denoting which subcategory of fineartamerica.com to scrape.
    subcategory_title: string, used to name the saved image files, preferably title-cased with no spaces.
    '''
    
    # Set base url
    url = f'https://fineartamerica.com/art/{subcategory_url_string}/owl'
    
    # Check if the target directory exists, create it if it does not
    check_directory_existence(directory)
    
    # Create subdirectory if it does not exist
    directory_tail = f'Page_{start_page:03d}-{end_page:03d}'
    subdirectory = directory + '/' + directory_tail
    
    check_directory_existence(subdirectory)
    
    page_range = range(start_page, end_page + 1)
    
    # For each page, scrape the page using BeautifulSoup
    for page in page_range:
        time.sleep(0.1)
        
        # Set url tail to scrape specific page
        url_tail = f'?page={page}'
        print(f'Scraping page {page}')
        
        response = requests.get(url + url_tail)
        soup = BeautifulSoup(response.text)
        
        # All images are wrapped in img tags
        image_tags = soup.find_all('img')
        
        # Some images are site logos - extract only those that are useful images
        image_tags_cleaned = [x for x in image_tags if 'data-src' in\
                              x.attrs.keys() and 'artworkimages' in x['data-src']]
        
        # Write each image to disk
        for index, image in enumerate(image_tags_cleaned):
            time.sleep(0.1)
            
            print(f'Saving image {index + 1} of {len(image_tage_cleaned)} to {subdirectory}.')
            
            img = Image.open(requests.get(image['data-src'], stream = True).raw)
            img_name = f'FineArtAmerica{subcategory_title}_Page_{page:03d}_Image_{index + 1:03d}.{img.format}'
            img.save(subdirectory + '/' + img_name)
        
    return