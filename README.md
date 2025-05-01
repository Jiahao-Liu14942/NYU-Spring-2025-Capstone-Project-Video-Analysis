# Multimodal Harmful Content Analysis Tool

This tool performs multimodal analysis of video content, combining visual classification, OCR-based text analysis, and audio transcription with NLP to detect harmful, xenophobic, misinformation, and neutral content.

## 📁 Project Structure

```
Video_analysis/
├── models/               # Contains trained classification model (.pth)
├── scripts/              # All core scripts including pipeline.py
│   ├── pipeline.py       # Main script to run the full analysis
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
```

## ✅ Features

- Extracts video frames at fixed intervals
- Transcribes audio using Whisper
- Extracts visible text using TrOCR (multilingual OCR)
- Classifies:
  - Image content using ResNet50
  - Transcribed audio and OCR text using zero-shot NLP
- Supports multiple languages (audio + text)
- Saves results to a CSV and visualizes label distribution

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

Ensure Python 3.9+ is installed.

## ▶️ Usage

Place your video in any location and run the pipeline as follows:

```bash
python scripts/pipeline.py --video path/to/your_video.mp4
```

Optional arguments:

- `--model`: Path to your .pth image classification model  
  Default: models/harmful_content_classifier_resnet50.pth
- `--output`: Output path for the result CSV  
  Default: multimodal_results.csv

## 💡 Output

After execution, it generates:

- A CSV file (default multimodal_results.csv) with:
  - Image filename
  - Image classification label
  - Transcribed audio + label + language + confidence
  - OCR-detected text + label + language + confidence

- A bar chart displaying the distribution of harmful content types.

## 🧠 Models Used

- Image: ResNet50 (custom fine-tuned)
- Audio transcription: OpenAI Whisper
- OCR: Microsoft TrOCR (microsoft/trocr-base-handwritten)
- Text classification: DeBERTa-v3 Zero-shot (MoritzLaurer/deberta-v3-large-zeroshot-v1)

## 📝 Notes

- This script supports multiple languages in both audio and text.
- All intermediate data (frames, transcripts) is stored temporarily.
- Tested on CPU-only environment.

## 📄 License

This project is provided for academic and research use only.
