import requests
import os

def download_public_file(url, company_name):
    """Downloads a publicly accessible file from a URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        destination_path = os.path.join(os.getcwd(), "data", company_name)
        os.makedirs(destination_path, exist_ok=True)
        destination_path = os.path.join(destination_path, os.path.basename(url)).replace("%20", " ")

        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File downloaded to {destination_path}.")
        return destination_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


# Example usage
# Replace with the public URL and desired local path
# download_public_file(
#     url="https://storage.googleapis.com/your-bucket-name/path/to/your/public_file.csv",
#     destination_path="local_data.csv"
# )
