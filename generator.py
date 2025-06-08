import os
import base64
import argparse
from pydantic import BaseModel
from openai import OpenAI
from iptcinfo3 import IPTCInfo
from tqdm import tqdm
import cv2  # Added for video frame extraction
import tempfile
import subprocess

client = OpenAI()


class ImageDescription(BaseModel):
    name: str
    keywords: list[str]


def extract_first_frame(video_path: str) -> str:
    """
    Extracts the first frame from a video and saves it as a temporary JPEG file.
    Returns the path to the temporary image file.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame from video: {video_path}")
    temp_fd, temp_img_path = tempfile.mkstemp(suffix=".jpg")
    os.close(temp_fd)
    cv2.imwrite(temp_img_path, frame)
    return temp_img_path


def embed_metadata_in_video(video_path: str, title: str, keywords: list[str]):
    """
    Embed metadata (title and keywords) into an MP4 video using ffmpeg and exiftool for XMP compatibility with Adobe Stock.
    """
    keywords_str = ";".join(keywords)
    metadata_args = [
        '-metadata', f'title={title}',
        '-metadata', f'keywords={keywords_str}',
        '-metadata', f'comment={keywords_str}',
        '-metadata', f'description={keywords_str}'
    ]
    temp_output = video_path + ".temp.mp4"
    # Step 1: Use ffmpeg to set standard metadata
    cmd = [
        'ffmpeg', '-y', '-i', video_path, *metadata_args, '-codec', 'copy', temp_output
    ]
    subprocess.run(cmd, check=True)
    os.replace(temp_output, video_path)
    # Step 2: Use exiftool to set XMP:Subject (keywords) for Adobe Stock compatibility
    exiftool_keywords = []
    for kw in keywords:
        exiftool_keywords.extend(['-XMP:Subject=' + kw])
    subprocess.run(['exiftool', '-overwrite_original'] + exiftool_keywords + [video_path], check=True)


def process_images_and_embed_metadata(path: str, location: str = None) -> dict:
    """
    Process a single image, video, or all images/videos in a directory, generate a name and 49 keywords for each using OpenAI Vision,
    and embed the results into the image's IPTC metadata.

    Args:
        path (str): Path to an image/video file or a directory containing images/videos.
        location (str, optional): Location where images/videos were taken. Adds this info to the prompt for name/keywords.
    Returns:
        dict: Results for each processed file or a single file.
    """
    results = {}
    if os.path.isdir(path):
        # Gather all image and video files in the directory
        media_files = [
            fname for fname in os.listdir(path)
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".mp4"))
        ]
        for fname in tqdm(media_files, desc="Processing media"):
            fpath = os.path.join(path, fname)
            results[fname] = process_images_and_embed_metadata(fpath, location)
        return results
    elif os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        is_video = False
        if ext in [".jpg", ".jpeg", ".png"]:
            image_path = path
            filename = os.path.basename(path)
        elif ext == ".mp4":
            # Extract first frame from video
            image_path = extract_first_frame(path)
            filename = os.path.basename(path)
            is_video = True
        else:
            return {"error": f"Unsupported file type: {path}"}

        # Read image as bytes and encode to base64
        with open(image_path, "rb") as f:
            raw_bytes = f.read()
        b64_str = base64.b64encode(raw_bytes).decode("utf-8")
        # Compose location info for prompt
        location_text = f" The image/video was taken in or near: {location}." if location else ""
        # Call OpenAI Vision API to generate name and keywords
        vision_resp = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a vision agent that generates a descriptive title and 49 unique, relevant keywords for stock images, "
                        "following Adobe Stock standards. Output should be a JSON object with 'name' and 'keywords' fields."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Please analyze the following image (image name: '{filename}'). Generate:\n"
                                "- A short, descriptive English title for Adobe Stock.\n"
                                "- 49 unique, relevant English keywords as a list of strings, covering subject, concept, location, and mood. "
                                "Avoid duplicates, generic terms, and brand names.\n"
                                f"{location_text}\n"
                                'Return the result as a JSON object: {"name": ..., "keywords": [...]}. '
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_str}",
                            },
                        },
                    ],
                },
            ],
            response_format=ImageDescription
        )
        # Parse response and embed metadata
        img_desc = vision_resp.choices[0].message.parsed
        info = IPTCInfo(image_path, force=True)
        info['object name'] = img_desc.name
        info['keywords'] = img_desc.keywords
        info.save_as(image_path)
        # Remove backup file if it exists
        backup_path = image_path + "~"
        if os.path.exists(backup_path):
            os.remove(backup_path)
        tqdm.write(f"New image name: {img_desc.name}")
        recognized_text = vision_resp.choices[0].message.content

        # Clean up temp image if video and embed metadata in video
        if is_video:
            embed_metadata_in_video(path, img_desc.name, img_desc.keywords)
            if os.path.exists(image_path):
                os.remove(image_path)

        return {"labels": recognized_text}
    else:
        return {"error": f"Path '{path}' is not a valid file or directory."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recognize image(s) and embed metadata using OpenAI vision agent."
    )
    parser.add_argument(
        "path", type=str, help="Path to an image file or directory containing images."
    )
    parser.add_argument(
        "--location", type=str, default=None, help="Location where images/videos were taken (optional)."
    )
    args = parser.parse_args()
    result = process_images_and_embed_metadata(args.path, args.location)
    # Print error if returned
    if isinstance(result, dict) and "error" in result:
        print(result["error"])
