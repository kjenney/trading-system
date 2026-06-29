import argparse
import boto3

def main():
    parser = argparse.ArgumentParser(description="Detect text in images using Amazon Rekognition")
    parser.add_argument("--profile", default=None, help="AWS profile name")
    parser.add_argument("image", help="Path to local image file")
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile)
    client = session.client("rekognition", region_name="us-east-1")

    with open(args.image, "rb") as f:
        image_bytes = f.read()

    response = client.detect_text(Image={"Bytes": image_bytes})

    for detection in response["TextDetections"]:
        if detection["Type"] == "WORD":
            print(f"Word: {detection['DetectedText']}  |  Confidence: {detection['Confidence']:.1f}%")

if __name__ == "__main__":
    main()
