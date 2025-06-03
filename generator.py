import os
import base64
import argparse
from pydantic import BaseModel
from openai import OpenAI
from iptcinfo3 import IPTCInfo
from tqdm import tqdm

client = OpenAI()


class ImageDescription(BaseModel):
    name: str
    keywords: list[str]


def process_images_and_embed_metadata(path: str) -> dict:
    """
    Process a single image or all images in a directory, generate a name and 49 keywords for each image using OpenAI Vision,
    and embed the results into the image's IPTC metadata.

    Args:
        path (str): Path to an image file or a directory containing images.
    Returns:
        dict: Results for each processed image or a single image.
    """
    results = {}
    if os.path.isdir(path):
        # Gather all image files in the directory
        image_files = [
            fname for fname in os.listdir(path)
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        # Show progress bar while processing images
        for fname in tqdm(image_files, desc="Processing images"):
            fpath = os.path.join(path, fname)
            results[fname] = process_images_and_embed_metadata(fpath)
        return results
    elif os.path.isfile(path):
        image_path = path
        filename = os.path.basename(path)
        # Read image as bytes and encode to base64
        with open(image_path, "rb") as f:
            raw_bytes = f.read()
        b64_str = base64.b64encode(raw_bytes).decode("utf-8")
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
                                'Return the result as a JSON object: {"name": ..., "keywords": [...]}.'
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
    args = parser.parse_args()
    result = process_images_and_embed_metadata(args.path)
    # Print error if returned
    if isinstance(result, dict) and "error" in result:
        print(result["error"])
