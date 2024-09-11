from bs4 import BeautifulSoup

def extract_text_to_file(xml_file_path, output_file_path):
    """
    Extract text from XML file and save it to a text file.

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

# Example usage
extract_text_to_file(
    ''
)
