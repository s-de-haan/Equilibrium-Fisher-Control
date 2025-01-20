import re
from datetime import datetime, timedelta

def shift_srt_time(timestamp, seconds_to_add):
    """Convert SRT timestamp to new time with added seconds"""
    # Parse hours, minutes, seconds, milliseconds
    parts = timestamp.split(',')
    time_parts = parts[0].split(':')
    hours, minutes, secs = map(int, time_parts)
    millisecs = int(parts[1])
    
    # Create timedelta and add seconds
    time_obj = timedelta(hours=hours, minutes=minutes, seconds=secs, milliseconds=millisecs)
    new_time = time_obj + timedelta(seconds=seconds_to_add)
    
    # Format new timestamp
    total_seconds = new_time.total_seconds()
    new_hours = int(total_seconds // 3600)
    new_minutes = int((total_seconds % 3600) // 60)
    new_seconds = int(total_seconds % 60)
    new_millisecs = int((total_seconds * 1000) % 1000)
    
    return f"{new_hours:02d}:{new_minutes:02d}:{new_seconds:02d},{new_millisecs:03d}"

def shift_srt_file(input_text, seconds_to_add):
    """Shift all timestamps in an SRT file by specified seconds"""
    # Regular expression to match SRT timestamp lines
    timestamp_pattern = r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})'
    
    def replace_timestamps(match):
        start_time = match.group(1)
        end_time = match.group(2)
        new_start = shift_srt_time(start_time, seconds_to_add)
        new_end = shift_srt_time(end_time, seconds_to_add)
        return f"{new_start} --> {new_end}"
    
    return re.sub(timestamp_pattern, replace_timestamps, input_text)

# Example usage
if __name__ == "__main__":
    with open('subtitles.srt', 'r', encoding='utf-8') as file:
        srt_content = file.read()
        
    shifted_content = shift_srt_file(srt_content, 22)
    
    with open('output_shifted.srt', 'w', encoding='utf-8') as file:
        file.write(shifted_content)