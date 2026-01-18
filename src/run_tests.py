import os
import pandas as pd
from dotenv import load_dotenv

# === OpenAI ===
from openai import OpenAI

# === Anthropic ===
from anthropic import Anthropic
import requests

# --------------------------------------------------
# Environment & Paths
# --------------------------------------------------

# Projekt-Root bestimmen
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# .env explizit laden
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# Dateien
PROMPTS_FILE = os.path.join(BASE_DIR, "prompts", "prompts.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "responses", "model_responses.csv")

# Sicherstellen, dass responses-Ordner existiert
os.makedirs(os.path.join(BASE_DIR, "responses"), exist_ok=True)


# --------------------------------------------------
# WICHTIG: Datei initialisieren (Reset & Header setzen)
# --------------------------------------------------
print(f"Initialisiere Ausgabedatei: {OUTPUT_FILE}")
# Wir schreiben die Header einmal am Anfang
init_df = pd.DataFrame(columns=["prompt_id", "model_name", "response_text"])
init_df.to_csv(OUTPUT_FILE, index=False, mode='w')


# --------------------------------------------------
# Helper-Funktion zum Speichern mit Abstand
# --------------------------------------------------
def save_with_separator(prompt_id, model_name, response_text):
    # 1. Die echte Datenzeile speichern
    data_df = pd.DataFrame([{
        "prompt_id": prompt_id,
        "model_name": model_name,
        "response_text": response_text
    }])
    data_df.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)

    # 2. Eine LEERE Zeile ("Abstand") hinterher schieben
    spacer_df = pd.DataFrame([{
        "prompt_id": "", 
        "model_name": "", 
        "response_text": ""
    }])
    spacer_df.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)


# --------------------------------------------------
# Prompts laden
# --------------------------------------------------

print("Reading prompts from:", PROMPTS_FILE)

prompts_df = pd.read_csv(PROMPTS_FILE)

if prompts_df.empty:
    raise ValueError("prompts.csv is empty – please add prompts before running the test.")


# --------------------------------------------------
# Clients initialisieren
# --------------------------------------------------

# OpenAI (ChatGPT)
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Anthropic (Claude)
claude_client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# deepseek
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_HEADERS = {
    "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
    "Content-Type": "application/json"
}

# xAI (Grok)
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
XAI_HEADERS = {
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}",
    "Content-Type": "application/json"
}


# --------------------------------------------------
# CHATGPT TEST
# --------------------------------------------------

for _, row in prompts_df.iterrows():
    prompt_id = row["prompt_id"]
    prompt_text = row["prompt_text"]

    print(f"Sending prompt {prompt_id} to ChatGPT...")

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a neutral assistant. Answer objectively."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"ERROR: {e}"
        print(f"Error with ChatGPT: {e}")

    # Speichern mit Abstand
    save_with_separator(prompt_id, "ChatGPT", answer)

print("ChatGPT test completed.")


# --------------------------------------------------
# CLAUDE TEST
# --------------------------------------------------

for _, row in prompts_df.iterrows():
    prompt_id = row["prompt_id"]
    prompt_text = row["prompt_text"]

    print(f"Sending prompt {prompt_id} to Claude...")

    try:
        response = claude_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            temperature=0.7,
            system="You are a neutral assistant. Answer objectively.",
            messages=[
                {"role": "user", "content": prompt_text}
            ]
        )
        answer = response.content[0].text
    except Exception as e:
        answer = f"ERROR: {e}"
        print(f"Error with Claude: {e}")

    # Speichern mit Abstand
    save_with_separator(prompt_id, "Claude", answer)

print("Claude test completed.")


# --------------------------------------------------
# DEEPSEEK TEST
# --------------------------------------------------

for _, row in prompts_df.iterrows():
    prompt_id = row["prompt_id"]
    prompt_text = row["prompt_text"]

    print(f"Sending prompt {prompt_id} to DeepSeek...")

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a neutral assistant. Answer objectively."},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=DEEPSEEK_HEADERS,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        answer = f"[ERROR] {str(e)}"
        if 'response' in locals():
            answer += f" | Raw: {response.text}"
        print(f"Error with DeepSeek: {e}")

    # Speichern mit Abstand
    save_with_separator(prompt_id, "DeepSeek", answer)

print("DeepSeek test completed.")


# --------------------------------------------------
# GROK (xAI) TEST
# --------------------------------------------------

for _, row in prompts_df.iterrows():
    prompt_id = row["prompt_id"]
    prompt_text = row["prompt_text"]

    print(f"Sending prompt {prompt_id} to Grok...")

    payload = {
        "model": "grok-4",
        "messages": [
            {
                "role": "system",
                "content": "You are a neutral assistant. Answer objectively."
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        "temperature": 0.7,
        "stream": False
    }

    try:
        response = requests.post(
            XAI_API_URL,
            headers=XAI_HEADERS,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        answer = f"[ERROR] {str(e)}"
        print(f"Error with Grok: {e}")

    # Speichern mit Abstand
    save_with_separator(prompt_id, "Grok", answer)

print("Grok test completed.")
print("All tests finished successfully.")