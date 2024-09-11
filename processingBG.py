import requests
import os

def download_images(base_url, start_index=1, output_dir="images"):
    """
    Downloads images from the specified IIIF base URL, processes and saves them in the given directory.
    
    Parameters:
        base_url (str): The base URL for the IIIF image service.
        start_index (int): The starting index for the image sequence.
        output_dir (str): The directory where images will be saved.
    """
    # Create a directory to store the images if it doesn't already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    i = start_index
    while True:
        # Pad the image number with leading zeros (e.g., 001, 002, 003)
        image_number = str(i)
        
        # Define the image metadata URL
        metadata_url = f"{base_url}{image_number}.jpg/info.json"

        try:
            # Fetch the metadata
            response = requests.get(metadata_url)
            
            # Check if the response status code is 200 (OK), meaning the image exists
            if response.status_code == 200:
                # Parse the JSON metadata
                image_api = response.json()

                # Access image dimensions from the JSON
                width = image_api.get('width')
                height = image_api.get('height')

                # Print the dimensions of the image
                print(f"Image {image_number}: Width: {width}, Height: {height}")

                # Define the base image URL for downloading
                image_url = f"{base_url}{image_number}.jpg/full/full/0/default.jpg"
                
                if width and width > 2880:
                    cropped_w = int(width / 2)

                    new_url1 = f"{base_url}{image_number}.jpg/{cropped_w},0,{cropped_w},{height}/full/0/default.jpg"
                    new_url2 = f"{base_url}{image_number}.jpg/0,0,{cropped_w},{height}/full/0/default.jpg"
                    
                    print("Cropped URLs:")
                    print(new_url1)
                    print(new_url2)

                    # Download and save the cropped images
                    for url, filename in [(new_url1, "cropped1.jpg"), (new_url2, "cropped2.jpg")]:
                        img_response = requests.get(url)
                        if img_response.status_code == 200:
                            with open(os.path.join(output_dir, f"{image_number}_{filename}"), "wb") as f:
                                f.write(img_response.content)
                            print(f"Downloaded cropped image: {filename}")
                        else:
                            print(f"Failed to download cropped image: {filename}")

                else:
                    print(f"Image {image_number} does not need resizing")
                    # Download and save the original image
                    img_response = requests.get(image_url)
                    if img_response.status_code == 200:
                        with open(os.path.join(output_dir, f"{image_number}.jpg"), "wb") as f:
                            f.write(img_response.content)
                        print(f"Downloaded image: {image_number}")
                    else:
                        print(f"Failed to download image: {image_number}")
            else:
                # If status code is not 200, it means no more images are found
                print(f"No more images found after image {image_number}")
                break  # Exit the loop

        except requests.exceptions.RequestException as e:
            # Handle potential network issues
            print(f"Error fetching image {image_number}: {e}")
            break  # Exit the loop on error

        # Increment the image number for the next iteration
        i += 1

# Example usage
base_url = "https://iiif.itatti.harvard.edu/iiif/2/bellegreene-full!32044150446383_00"
download_images(base_url)
