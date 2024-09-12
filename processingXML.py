import os
from bs4 import BeautifulSoup

def extract_text_from_xml(xml_file_path, output_file_path):
    """
    Extract text from an XML file and save it to a text file.

    :param xml_file_path: Path to the XML file to read.
    :param output_file_path: Path to the text file to save the extracted text.
    """
    # Open and read the XML file
    with open(xml_file_path, 'r') as f:
        file_content = f.read()

    # Parse the XML file with BeautifulSoup
    soup = BeautifulSoup(file_content, 'xml')

    # Find the 'machine-printed-part' tag
    tag1 = soup.find("machine-printed-part")

    if tag1 is None:
        print(f"No 'machine-printed-part' tag found in {xml_file_path}.")
        return

    # Find all 'machine-printed-line' tags within 'tag1'
    tag2 = tag1.find_all("machine-print-line")

    with open(output_file_path, 'w') as outfile:
        # Write the text from each 'machine-print-line' tag to the file
        for line in tag2:
            text = line.get("text")
            if text:  # Ensure there's text to write
                outfile.write(text + '\n')

def process_all_xml_files_in_directory(input_directory, output_directory):
    """
    Process all XML files in the specified directory and save the extracted text
    to the output directory.

    :param input_directory: Path to the directory containing XML files.
    :param output_directory: Path to the directory where text files will be saved.
    """
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.xml'):
            xml_file_path = os.path.join(input_directory, filename)
            output_file_path = os.path.join(output_directory, filename.replace('.xml', '.txt'))
            
            # Extract text and save to file
            extract_text_from_xml(xml_file_path, output_file_path)
            print(f"Processed {filename}")

# Example usage
input_directory = 'C:/Users/crosi/Documents/GitHub/InternshipITatti/IAM/xml'
output_directory = 'C:/Users/crosi/Documents/GitHub/InternshipITatti/IAM/txt'
process_all_xml_files_in_directory(input_directory, output_directory)

