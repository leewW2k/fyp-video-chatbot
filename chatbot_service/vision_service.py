from typing import List, Optional

import dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis.models import VisualFeatures

import cv2
import os
from skimage.metrics import structural_similarity as ssim

dotenv.load_dotenv()

class VisionService:
    """
    VisionService is a wrapper class of ImageAnalysisClient used for analysing image and text.
    It provides the ability to analyse images/frames using AI Cognitive Vision.

    Args:
        vision_endpoint (str): Azure Vision Endpoint. Example: https://<YOUR_RESOURCE_NAME>.cognitiveservices.azure.com/. Required.
        vision_key (str): Azure OpenAI API Key. Required.
        vision_features (List[VisualFeatures]): Azure Visual Features.
    """
    def __init__(
            self,
            vision_endpoint: str,
            vision_key: str,
            vision_features: Optional[List[VisualFeatures]] = None
    ):
        self.client = ImageAnalysisClient(
            endpoint=vision_endpoint,
            credential=AzureKeyCredential(vision_key)
        )
        if vision_features is None:
            self.vision_features = [
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.READ,
            ]
        else:
            self.vision_features = vision_features

    def analyze_image(self, image_data: bytes, language: str = "en"):
        """
        Perform analysis on the image

        Args:
            image_data (bytes): Image in bytes. Required.
            language (str): Language to analyze image in. Default: "en".

        Returns:
            Dictionary output with relevant information
        """
        result = self.client.analyze(
            image_data=image_data,
            visual_features=self.vision_features,
            language=language
        )

        output = {}

        if hasattr(result, 'caption') and result.caption:
            output["caption"] = {
                "text": getattr(result.caption, 'text', None),
                "confidence": getattr(result.caption, 'confidence', None)
            }

        if hasattr(result, 'dense_captions') and result.dense_captions:
            output["dense_captions"] = [
                {
                    "text": getattr(caption, 'text', None),
                    "confidence": getattr(caption, 'confidence', None)
                }
                for caption in getattr(result.dense_captions, 'list', [])
            ]

        if hasattr(result, 'read') and result.read and hasattr(result.read, 'blocks'):
            output["read"] = [
                {
                    "line": getattr(line, 'text', None),
                    "words": [
                        {
                            "text": getattr(word, 'text', None),
                            "confidence": getattr(word, 'confidence', None)
                        }
                        for word in getattr(line, 'words', [])
                    ]
                }
                for line in result.read.blocks[0].get("lines", [])
            ]

        if hasattr(result, 'tags') and result.tags:
            output["tags"] = [
                {
                    "name": getattr(tag, 'name', None),
                    "confidence": getattr(tag, 'confidence', None)
                }
                for tag in getattr(result.tags, 'list', [])
            ]

        if hasattr(result, 'objects') and result.objects:
            output["objects"] = [
                {
                    "name": getattr(obj.tags[0], 'name', None) if obj.tags else None,
                    "confidence": getattr(obj.tags[0], 'confidence', None) if obj.tags else None
                }
                for obj in getattr(result.objects, 'list', [])
            ]

        return output

    def extract_relevant_frames(self, video_path, output_dir, threshold=0.8, frame_interval=30):
        """
        Extract relevant frames from a video.

        Args:
            video_path (str): Path to the video file. Required.
            output_dir (str): Directory to save the extracted frames. Required.
            threshold (float): SSIM threshold below which frames are considered different. Default: 0.8.
            frame_interval (int): Number of frames to skip between comparisons. Default: 30.

        Returns:
            List of extracted frame paths.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Open the video file
        video_capture = cv2.VideoCapture(video_path)
        frame_count = 0
        extracted_frames = []
        prev_frame = None

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_count += 1
            # Process only every Nth frame
            if frame_count % frame_interval != 0:
                continue

            # Convert the frame to grayscale for comparison
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                # Compare the current frame with the previous frame
                score, _ = ssim(prev_frame, gray_frame, full=True)
                if score < threshold:
                    # Save the frame as it's significantly different
                    timestamp_ms = video_capture.get(cv2.CAP_PROP_POS_MSEC)  # Timestamp in milliseconds
                    timestamp_sec = timestamp_ms / 1000  # Convert to seconds
                    frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    extracted_frames.append({
                        "frame_path": frame_path,
                        "timestamp": timestamp_sec
                    })

            # Update the previous frame
            prev_frame = gray_frame

        video_capture.release()
        return extracted_frames