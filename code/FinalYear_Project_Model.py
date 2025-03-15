import os
import yt_dlp
from moviepy.editor import *
import whisper
import requests
import cv2
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from ultralytics import YOLO
from torchvision import models, transforms
from torch.nn.functional import softmax
from difflib import SequenceMatcher
from moviepy.editor import VideoFileClip


# Step 1: Download YouTube Video
def download_youtube_video(url, output_dir='/content/downloaded_videos'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the full path to where the video will be saved
    video_path = os.path.join(output_dir, 'downloaded_video.mp4')

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',  # Download best quality available
        'outtmpl': video_path,                 # Use specified directory and filename
        'merge_output_format': 'mp4',          # Force merge in .mp4 format
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            # Check if the video was downloaded successfully
            if os.path.exists(video_path):
                print(f"Downloaded video to: {video_path}")
                return video_path
            else:
                print("Download failed: File not found.")
                return None
    except Exception as e:
        print("Download failed:", e)
        return None


# Step 2: Extract Audio from Video
def extract_audio_from_video(video_path):
    # Check if video file exists before proceeding
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return None

    # Define audio path
    audio_path = video_path.replace(".mp4", ".wav")

    try:
        # Extract audio
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        print(f"Extracted audio to: {audio_path}")
        return audio_path
    except Exception as e:
        print("Audio extraction failed:", e)
        return None


# Step 3: Transcribe Audio to Text Using Whisper (Large model for multilingual support)
def transcribe_audio(audio_path):
    model = whisper.load_model("large")  # Large model for better multilingual support
    result = model.transcribe(audio_path)
    return result["text"], result["language"]


from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np

# Function to extract medical terms using BioBERT
def extract_medical_terms_with_biobert(text):
    model_name = "dmis-lab/biobert-base-cased-v1.1"

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    # Tokenize the input text with padding and truncation
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512, is_split_into_words=False)

    # Get model output (logits)
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(**tokens)

    # Get predictions
    predictions = torch.argmax(output.logits, dim=2).numpy()

    medical_terms = []
    current_term = ""

    # Extract medical terms based on predictions
    for i, token_id in enumerate(tokens["input_ids"][0]):
        token = tokenizer.decode([token_id])
        label = predictions[0][i]

        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]", "[UNK]", ",", ".", "##"]:
            continue

        # Check if label indicates a medical term (BIO tagging expected)
        if label > 0:  # Assuming medical terms have labels > 0
            # Handle subword tokens
            if token.startswith("##"):
                current_term += token[2:]  # Merge subword tokens
            else:
                if current_term:  # Add completed term to the list
                    medical_terms.append(current_term.strip())
                current_term = token  # Start new term

    # Add the last term if it exists
    if current_term:
        medical_terms.append(current_term.strip())

    # Remove duplicates
    medical_terms = list(set(medical_terms))

    # Filter out irrelevant or generic terms (optional step)
    stop_words = set(["and", "of", "in", "the", "to", "with", "a", "is"])
    medical_terms = [term for term in medical_terms if term.lower() not in stop_words]

    return medical_terms

# Step 5: Multilingual Fact-Checking Function
def fact_check_medical_content_all(medical_terms, language):
    flagged_terms = []

    # Updated database endpoints to include more multilingual or regional databases
    databases = {
        "Primary Care": ["ClinicalTrials.gov", "WHO", "Europe PMC"],
        "Surgical Specialties": ["ClinicalTrials.gov", "MedlinePlus", "Medscape"],
        "Medical Specialties": ["PubMed", "ClinicalTrials.gov", "Europe PMC"],
        "Diagnostic Specialties": ["PubMed", "ClinicalTrials.gov"],
        "Mental Health": ["Psychiatric Database", "PubMed", "Europe PMC"],
        "Sports Medicine": ["Sports Medicine Research Journals", "PubMed"],
        "Medical Genetics": ["GeneOntology", "NCBI Genetic Databases", "Europe PMC"],
        "Rehabilitation and Pain Management": ["PubMed", "ClinicalTrials.gov"],
        "Other Specialties": ["PubMed", "ClinicalTrials.gov"]
    }

    for term in medical_terms:
        validated = False

        for db_type, db_list in databases.items():
            for db in db_list:
                if db == "ClinicalTrials.gov":
                    response = requests.get(f"https://clinicaltrials.gov/api/query/full_studies?expr={term}&min_rnk=1&max_rnk=1&fmt=json")
                    if response.status_code == 200 and len(response.json()['FullStudiesResponse']['FullStudies']) > 0:
                        validated = True

                elif db == "WHO":
                    response = requests.get(f"https://www.who.int/api/v1/search?q={term}&lang={language}")
                    if response.status_code == 200 and len(response.json().get('results', [])) > 0:
                        validated = True

                elif db == "Europe PMC":
                    response = requests.get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={term}&format=json")
                    if response.status_code == 200 and len(response.json().get('resultList', {}).get('result', [])) > 0:
                        validated = True

                elif db == "PubMed":
                    response = requests.get(f"https://pubmed.ncbi.nlm.nih.gov/?term={term}")
                    if response.status_code == 200:
                        validated = True

        if not validated:
            flagged_terms.append(term)

    return flagged_terms


# Step 6: Analyze Video Frames Using YOLOv8 and CNN for Enhanced Medical Object Detection
def analyze_video_frames_yolo_cnn(video_path):
    # Load YOLO and CNN models
    yolo_model = YOLO('yolov8m.pt')  # YOLO model fine-tuned for medical objects
    cnn_model = models.resnet50(pretrained=True)
    cnn_model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(video_path)
    medical_objects = ["stethoscope", "scalpel", "ECG", "syringe", "dental-tool", "herb", "surgical-mask"]
    detected_items = []
    frame_skip = 5  # Process every 5th frame for efficiency

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            # YOLO object detection
            results = yolo_model(frame)
            for result in results:
                detected_labels = [yolo_model.names[int(cls)] for cls in result.boxes.cls]
                detected_medical_objects = [label for label in detected_labels if label in medical_objects]

                # If YOLO detects a relevant medical object, verify with CNN
                for detected_object in detected_medical_objects:
                    cnn_input = transform(frame).unsqueeze(0)  # Preprocess the frame for CNN input
                    with torch.no_grad():
                        output = cnn_model(cnn_input)
                        prob = softmax(output, dim=1)
                        confidence = prob.max().item()
                        if confidence > 0.6:  # Confidence threshold for CNN
                            detected_items.append(detected_object)

        frame_count += 1

    cap.release()
    return list(set(detected_items))


# Step 7: Flag Video for Misinformation with Enhanced Scoring
def flag_video(medical_terms, flagged_terms, visual_detections):
    print("Extracted Medical Terms:", medical_terms)
    print("Flagged Terms:", flagged_terms)
    print("Detected Visuals:", visual_detections)

    misinformation_score = len(flagged_terms) + (0 if visual_detections else 1)

    if misinformation_score > 0:
        print("Video contains potential medical misinformation.")
    else:
        print("No misinformation detected.")
    return flagged_terms


# Main Function
def main():
    url = input("Enter the YouTube video URL: ")
    video_path = download_youtube_video(url)

    if video_path:
        audio_path = extract_audio_from_video(video_path)
        transcription, language = transcribe_audio(audio_path)

        # Multilingual Medical Term Extraction
        medical_terms = extract_medical_terms_with_biobert(transcription, language)

        # Multilingual Fact-Checking Across All Medical Fields
        flagged_terms = fact_check_medical_content_all(medical_terms, language)

        # YOLO and CNN Hybrid Detection for Medical Imagery
        visual_detections = analyze_video_frames_yolo_cnn(video_path)

        # Flag Video
        flag_video(medical_terms, flagged_terms, visual_detections)

if __name__ == "__main__":
    main()
