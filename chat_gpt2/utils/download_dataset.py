import os
import urllib.request
import json
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def download_and_load_dataset(file):
    download_file_path = os.getcwd()
    if not os.path.exists(file):
        if file.startswith("https"):
            with urllib.request.urlopen(file) as response:
                text_data = response.read().decode("utf-8")

            filename = os.path.basename(file) or "downloaded_file.txt"
            full_path = os.path.join(download_file_path, filename)

            with open(full_path, "w", encoding="utf-8") as data_file:
                data_file.write(text_data)

            # Load as plain text
            with open(full_path, "r", encoding="utf-8") as data_file:
                data = data_file.read()
            return data
        else:
            raise FileNotFoundError("File does not exist. Please provide a valid data_file_path or URL.")
    else:
        
        if file.endswith(".json"):
            import json
            with open(file, "r", encoding="utf-8") as data_file:
                data = json.load(data_file)
        else:
            with open(file, "r", encoding="utf-8") as data_file:
                data = data_file.read()
        return data



def partition_data(data):
    n = len(data)
    train_portion = int(n * 0.85)
    test_portion = int(n * 0.10)
    val_portion = n - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training data length:", len(train_data))
    print("Test data length:", len(test_data))
    print("Validation data length:", len(val_data))

    return train_data, test_data, val_data



