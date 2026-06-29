# Detect Words

Command-line utility for detecting text in images using Amazon Rekognition.

## Overview

Troubleshooting script that uses Amazon Rekognition's `detect_text` API to extract and display words from image files. Useful for verifying OCR quality when working with screenshot-based data sources.

## Usage

```bash
python detectwords.py --profile <aws_profile> <image_path>
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `image` | Yes | — | Path to local image file |
| `--profile` | No | None | AWS profile name (uses default if not specified) |

### Output

```
Word: AAPL  |  Confidence: 99.2%
Word: $180  |  Confidence: 97.5%
Word: 2024  |  Confidence: 98.8%
```

Each line shows the detected word and its confidence percentage. Only `WORD`-type detections are shown (not full-text block detections).

## Requirements

- `boto3` — AWS SDK for Python
- AWS credentials configured (via `~/.aws/credentials` or environment variables)
- Image stored locally (file path, not S3 URL)

## Implementation

```python
import boto3

session = boto3.Session(profile_name=args.profile)
client = session.client("rekognition", region_name="us-east-1")

with open(args.image, "rb") as f:
    image_bytes = f.read()

response = client.detect_text(Image={"Bytes": image_bytes})

for detection in response["TextDetections"]:
    if detection["Type"] == "WORD":
        print(f"Word: {detection['DetectedText']}  |  Confidence: {detection['Confidence']:.1f}%")
```

Uses `us-east-1` region. Passes image bytes directly via `Image={"Bytes": ...}` rather than S3 reference.
