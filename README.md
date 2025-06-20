# Adobe Stock Auto Metadata Generator
Scripts for automatic metadata image generation for Adobe Stock

## Requirements

You need the following installed to run `generator.py`:

### Python packages (install with pip):
- openai
- pydantic
- tqdm
- iptcinfo3
- opencv-python

Install all Python dependencies with:
```
pip install -r requirements.txt
```

### System dependencies:
- ffmpeg (for video processing)
- exiftool (for XMP metadata in videos, required for Adobe Stock compatibility)

#### On Ubuntu/Debian:
```
sudo apt-get update
sudo apt-get install ffmpeg libimage-exiftool-perl
```

## Usage

Before running the script, you must set your OpenAI API key as an environment variable:
```
export OPENAI_API_KEY=your_openai_api_key_here
```

Run the script with:
```
python generator.py <path-to-image-or-video-or-directory> [--location "Location Name"]
```

- The script will process images and videos, generate titles and keywords using OpenAI Vision, and embed metadata compatible with Adobe Stock.
- For videos, keywords are embedded using exiftool for XMP compatibility.
