from typing import Any


def convert_seconds_to_mm_ss(seconds: int):
    """
    Method to convert seconds to minutes & seconds.

    Args:
        seconds (int): Number of seconds. Required.

    Returns:
        Time in mm:ss format.
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02}:{seconds:02}"

def process_file(fp: str, mode: str = "r", content: str | bytes = None):
    """
    Method to handle reading from or writing to a specific file.

    Args:
        fp (str): File path to read. Required.
        mode (str): Mode. Default: 'r'.
        content (str | bytes): Content to write to the file (optional, required for writing).

    Returns:
        read file if reading, None if writing
    """
    try:
        if "r" in mode:
            with open(fp, mode) as file:
                return file.read()
        elif "w" in mode or "a" in mode:
            if content is None:
                raise ValueError("Content must be provided for writing or appending.")
            with open(fp, mode) as file:
                file.write(content)
                return None
        else:
            raise ValueError(f"Unsupported file mode: {mode}")
    except Exception as e:
        print(e)
        return e