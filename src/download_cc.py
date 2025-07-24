import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

TIMEOUT = 1
WORKERS = 100

def read_file(path:str): 
    data_list = []
    
    with open(path) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            text = row[0]
            url = row[1]
            
            data_list.append((text, url))
    
    return data_list

def download_image(url: str, verbose=False): 
    import requests
    from pathlib import Path
    try: 
        response = requests.get(url, timeout=TIMEOUT)
        if response.status_code == 200:
            filename = url.split("/")[-1]
            
            #TODO fix this. i think it rejects some online images
            if not filename.endswith(('.jpg', '.jpeg', '.png')):
                if verbose:
                    print(f"Skipping {url}: Not a valid image file.")
                return None
            filepath = Path("res/data/conceptual-captions/images") / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, "wb") as f:
                f.write(response.content)
            return filepath
        else:
            if verbose:
                print(f"Failed to download {url}")
            return None
    except Exception as e:
        if verbose:
            print(f"Error downloading {url}: {e}")
        return None

def download_single(dp):
    text, url = dp
    filepath = download_image(url)
    return (text, url, filepath)

def save_file(downloaded_data):
    with open("res/data/conceptual-captions/downloaded_data.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "filepath"])  # Header
        writer.writerows(downloaded_data)

# data_list = read_file("res/data/conceptual-captions/Validation_GCC-1.1.0-Validation.tsv")
data_list = read_file("res/data/conceptual-captions/Train_GCC-training.tsv")
data_list = data_list[:200000]


error_counter = 0
downloaded_data = []

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(download_single, dp): dp for dp in data_list}
    
    for future in tqdm(as_completed(futures), total=len(data_list), desc="downloading images"):
        text, url, filepath = future.result()
        if filepath is None:
            error_counter += 1
        else:
            downloaded_data.append((text, filepath))


save_file(downloaded_data)

print(f"Failed to download {error_counter} images out of {len(data_list)} total images.")
print(f"Saved {len(downloaded_data)} downloaded images to CSV")