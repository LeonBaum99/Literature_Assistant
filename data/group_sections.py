import json
import os
import glob
from collections import defaultdict

def extract_and_group_sections(input_directory, output_file=None):
    """
    Reads all JSON files in the input_directory, extracts content from the 'sections'
    key, and groups them by section title.
    
    Args:
        input_directory (str): Path to the folder containing JSON files.
        output_file (str, optional): Path to save the grouped JSON output. 
                                     If None, prints to console.
    """
    # Dictionary to hold grouped content: { "Section Name": [ { "paper": "Title", "content": "..." }, ... ] }
    grouped_sections = defaultdict(list)
    
    # Find all .json files in the directory
    json_files = glob.glob(os.path.join(input_directory, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_directory}")
        return

    print(f"Processing {len(json_files)} files...")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract metadata for context
                paper_title = data.get('metadata', {}).get('title', 'Unknown Title')
                filename = data.get('filename', os.path.basename(file_path))
                
                # Extract sections
                sections = data.get('sections', {})
                
                for section_title, section_content in sections.items():
                    # Create a record for this section entry
                    entry = {
                        "paper_title": paper_title,
                        "source_filename": filename,
                        "content": section_content
                    }
                    
                    # Group by the section title
                    grouped_sections[section_title].append(entry)
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Output the results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as out:
            json.dump(grouped_sections, out, indent=4, ensure_ascii=False)
        print(f"Successfully saved grouped sections to {output_file}")
    else:
        # If no output file is provided, print a summary to the console
        for section, entries in grouped_sections.items():
            print(f"\n--- Section: {section} ({len(entries)} entries) ---")
            for entry in entries:
                preview = entry['content'][:100].replace('\n', ' ') + "..."
                print(f"  [{entry['paper_title']}]: {preview}")

# --- Usage Configuration ---
if __name__ == "__main__":
    # 1. Set the directory where your JSON files are stored
    # Use '.' for the current directory or provide a full path like '/path/to/json/files'
    INPUT_DIR = 'C:\\Users\\kronask\\OneDrive - TU Wien\\TU Wien\\3. Semester\\GenAI\\GenAI\\data\\json_papers' 
    
    # 2. Set the output filename
    OUTPUT_FILENAME = 'grouped_paper_sections.json'
    
    extract_and_group_sections(INPUT_DIR, OUTPUT_FILENAME)