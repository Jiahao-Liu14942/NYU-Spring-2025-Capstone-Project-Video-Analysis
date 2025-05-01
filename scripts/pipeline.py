import torch
from torchvision import models, transforms
from PIL import Image
import os
from transformers import pipeline, TrOCRProcessor, VisionEncoderDecoderModel
import matplotlib.pyplot as plt
import pandas as pd
import langdetect
import csv
import whisper
import cv2
import tempfile


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# 1. Extract audio and transcribe it
def transcribe_audio(video_path, output_csv):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    segments = result["segments"]
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'text'])
        for idx, segment in enumerate(segments):
            frame_id = f"frame{idx:03d}.jpg"
            writer.writerow([frame_id, segment['text']])

# 2. Extract frames from video
def extract_frames(video_path, output_folder, interval=1):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(count % (frame_rate * interval)) == 0:
            filename = os.path.join(output_folder, f"frame{saved:03d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
        count += 1
    cap.release()

# 3. Load image classification model
def load_image_model(model_path, class_names):
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 4. Predict image label
def predict_image(image_path, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]
    return label

# 5. Predict text label using zero-shot classifier
def predict_text(text, classifier, labels):
    if not text.strip():
        return 'neutral', 0.0, 'und'
    lang = langdetect.detect(text)
    result = classifier(text, candidate_labels=labels, hypothesis_template="This text is {}.")
    return result['labels'][0], result['scores'][0], lang

# 6. Extract visible text from image using TrOCR
def extract_text_from_image(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()

# 7. Load transcript mapping from CSV
def load_text_mapping(csv_path):
    mapping = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                mapping[row[0]] = row[1]
    return mapping

# 8. Run analysis for a set of image-text pairs
def run_multimodal_analysis(image_folder, text_data, model_path, class_names):
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1")
    image_model = load_image_model(model_path, class_names)

    results = []
    for img_file, audio_text in text_data.items():
        img_path = os.path.join(image_folder, img_file)
        if not os.path.exists(img_path):
            continue

        image_label = predict_image(img_path, image_model, class_names)
        visible_text = extract_text_from_image(img_path)

        audio_label, audio_score, audio_lang = predict_text(audio_text, classifier, class_names)
        ocr_label, ocr_score, ocr_lang = predict_text(visible_text, classifier, class_names)

        results.append({
            'image': img_file,
            'image_label': image_label,
            'audio_text': audio_text,
            'audio_label': audio_label,
            'audio_conf': round(audio_score, 3),
            'audio_lang': audio_lang,
            'visible_text': visible_text,
            'ocr_label': ocr_label,
            'ocr_conf': round(ocr_score, 3),
            'ocr_lang': ocr_lang
        })
    return pd.DataFrame(results)

# 9. Visualize results with bar chart
def visualize_results(df):
    label_counts = pd.Series(df['image_label'].tolist() + df['audio_label'].tolist() + df['ocr_label'].tolist()).value_counts()
    label_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Multimodal Label Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 10. Main pipeline
if __name__ == '__main__':
    class_labels = ['harmful', 'xenophobic', 'misinformation', 'neutral']
    video_path = 'D:/Video_analysis/sample_video.mp4'
    model_weights = 'D:/Video_analysis/models/harmful_content_classifier_resnet50.pth'
    output_csv = 'D:/Video_analysis/multimodal_results.csv'

    frame_folder = tempfile.mkdtemp(prefix="frames_")
    transcript_csv = os.path.join(tempfile.gettempdir(), "transcripts.csv")

    os.makedirs(frame_folder, exist_ok=True)

    print("Extracting frames...")
    extract_frames(video_path, frame_folder)

    print("Transcribing audio...")
    transcribe_audio(video_path, transcript_csv)

    print("Running multimodal analysis...")
    text_mapping = load_text_mapping(transcript_csv)
    df = run_multimodal_analysis(frame_folder, text_mapping, model_weights, class_labels)

    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    visualize_results(df)

