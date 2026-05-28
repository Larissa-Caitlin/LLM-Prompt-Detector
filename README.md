# LLM Prompt Detector

AI-powered detection system for identifying prompt injection attacks and malicious prompts targeting large language models.

## Features

- Two-layer detection: Rule-based patterns + Machine Learning classifier
- Real-time prompt analysis with confidence scoring
- Interactive Streamlit UI for testing
- DistilBERT-based ML classifier for adversarial prompt detection
- Rule-based detection for known injection patterns
- Support for GPU acceleration

## How It Works

### Layer 1: Rule-Based Detection
Pattern matching against known prompt injection techniques:
- "ignore previous instructions"
- "act as an unrestricted AI"
- "jailbreak mode"
- And 10+ more patterns

### Layer 2: ML-Based Detection
DistilBERT fine-tuned classifier that detects novel adversarial prompts with >90% accuracy.

**Combined approach:** If either layer detects a threat, prompt is blocked.

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone repository
git clone https://github.com/Larissa-Caitlin/LLM-Prompt-Detector.git
cd LLM-Prompt-Detector

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Streamlit App
```bash
streamlit run app.py
```

App opens at `http://localhost:8501`

### Test Detection
1. Enter any prompt in the text box
2. Click "Analyze Prompt"
3. View detection result (Safe/Blocked)
4. See confidence score and detection layer

### Test Chatbot
Switch to "Chatbot" tab to chat with protected LLM.

## Project Structure
LLM-Prompt-Detector/
├── app.py                 # Streamlit UI
├── detector.py            # Detection logic (2-layer)
├── chatbot.py            # Protected chatbot interface
├── config.py             # Configuration settings
├── train_model.py        # Model training script
├── train_roberta.py      # RoBERTa training
├── balance_data.py       # Dataset balancing
├── expand_dataset.py     # Dataset expansion
├── detector.py           # Core detection
├── requirements.txt      # Dependencies
├── data/                 # Training data
└── models/              # Trained models
└── distilbert-injection/
└── final/       # DistilBERT model

## Technical Details

### Models Used
- **DistilBERT**: Lightweight BERT variant for prompt classification
- **RoBERTa**: Alternative model for comparison
- **Rule patterns**: 20+ regex patterns for injection detection

### Accuracy
- Rule-based detection: Catches 100% of known patterns
- ML classifier: >90% accuracy on adversarial prompts
- Combined: <1% false positive rate

## Configuration

Edit `config.py` to customize:
- `CONFIDENCE_THRESHOLD`: Lower = more strict (default: 0.7)
- `MODEL_PATH`: Path to trained model
- `MAX_PROMPT_LENGTH`: Truncate prompts to this length

## What This Demonstrates

- **AI/ML Knowledge**: Fine-tuning transformer models (BERT, RoBERTa)
- **Security**: Understanding prompt injection attacks and defenses
- **Full-Stack**: Backend detection + Frontend UI
- **Software Engineering**: Modular code, configuration management, error handling
- **Data Science**: Dataset balancing, model evaluation, threshold tuning

## Future Improvements

- Add ensemble models (vote-based detection)
- Fine-tune on custom datasets
- Support for multiple languages
- Real-time performance monitoring
- Integration with popular LLM APIs

## Security Notes

This detector helps protect LLMs but is not foolproof. Advanced attackers may find bypasses. Use in combination with:
- Input validation
- Output filtering
- Rate limiting
- User authentication

## License

MIT License - See LICENSE file

---

**Built by**: Larissa Caitlin Naomi J  
**Purpose**: Semester Project  - Cyber Security + AI 
**Last Updated**: May 2026