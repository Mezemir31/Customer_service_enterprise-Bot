# Customer Service Bot

An AI-powered customer service bot designed for Customer Service Bot system. This bot leverages Google's Gemini API for natural language understanding and provides intelligent responses to user grievances and queries.

## Features

- **AI-Powered Analysis**: Uses Google Gemini to analyze user queries, detect intent, sentiment, and urgency.
- **Intelligent Response Generation**: Generates context-aware and empathetic responses.
- **Multi-Language Support**: Capable of understanding and responding in multiple languages (English, Amharic, Oromo, etc.).
- **Entity Extraction**: Automatically extracts phone numbers, FIN/UIN, FAN/FCN and if needed others from user input.
- **Spelling Correction**: AI-driven spelling correction for better understanding of user queries.
- **Conversation History**: Maintains context across the conversation session.
- **Simulated Backend**: Includes a simulated API to mimic checking customer records and status.

## Project Structure

```
├── main.py                 # Entry point of the application
├── bot.py                  # Core bot logic (Analyzer, Generator, Manager)
├── config.py               # Configuration settings
├── database.py             # Database initialization and schema management
├── api_simulator.py        # Simulated backend API for data retrieval
├── text_processing/        # Text processing modules
│   ├── extraction.py       # Entity extraction logic
│   └── spelling.py         # AI spelling corrector
├── customer_service_advanced.db # SQLite database (created on run)
└── requirements.txt        # Python dependencies
```

## Prerequisites

- Python 3.8+
- Google Gemini API Key

## Installation

1.  **Clone the repository** (or download the files).
2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up Environment Variables**:
    Copy the example environment file to create your own configuration:
    ```bash
    cp .env.example .env
    ```
    Then, open `.env` and replace the placeholder with your actual Gemini API key.

## Usage

Run the main application:

```bash
python main.py
```


## How it Works

1.  **Input Processing**: The bot accepts user input via the command line.
2.  **Entity Extraction**: `text_processing/extraction.py` identifies key information like phone numbers and IDs.
3.  **AI Analysis**: `EnhancedGrievanceAnalyzer` (in `bot.py`) uses Gemini to understand the user's intent and category.
4.  **Response Generation**: `IntelligentResponseGenerator` (in `bot.py`) constructs a helpful response using the analysis and retrieved data.
5.  **Conversation Management**: `ConversationManager` tracks the session context to ensure coherent dialogue.

## Disclaimer

This project is a simulation and does not access real enterprise data and API's. It is for demonstration and development purposes only.

## Demonstration Video

<a href="https://vimeo.com/manage/videos/1150610786" target="_blank" rel="noopener noreferrer">
  Watch the demo video here: https://vimeo.com/manage/videos/1150610786
</a>