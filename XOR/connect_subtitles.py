import re
from datetime import datetime, timedelta

def connect_srt_timestamps(input_text):
    """Remove gaps between subtitles by making each end time match the next start time"""
    
    # Parse the SRT content into blocks
    blocks = input_text.strip().split('\n\n')
    connected_blocks = []
    
    for i in range(len(blocks)):
        lines = blocks[i].strip().split('\n')
        
        # Get current timestamp line
        current_time = re.search(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', lines[1])
        
        if current_time:
            # If there's a next block, get its start time
            if i < len(blocks) - 1:
                next_block = blocks[i + 1]
                next_time = re.search(r'(\d{2}:\d{2}:\d{2},\d{3})', next_block)
                if next_time:
                    # Replace current end time with next start time
                    lines[1] = f"{current_time.group(1)} --> {next_time.group(1)}"
            
            connected_blocks.append('\n'.join(lines))
    
    return '\n\n'.join(connected_blocks)

# Example usage
if __name__ == "__main__":
    with open('output_shifted.srt', 'r', encoding='utf-8') as file:
        srt_content = file.read()
        
    connected_content = connect_srt_timestamps(srt_content)
    
    with open('output_connected.srt', 'w', encoding='utf-8') as file:
        file.write(connected_content)