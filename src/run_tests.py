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
#deepseek
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_HEADERS = {
    "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
    "Content-Type": "application/json"
}
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
XAI_HEADERS = {
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}",
    "Content-Type": "application/json"
}


# # --------------------------------------------------
# # CHATGPT TEST
# # # --------------------------------------------------

# chatgpt_results = []

# for _, row in prompts_df.iterrows():
#     prompt_id = row["prompt_id"]
#     prompt_text = row["prompt_text"]

#     print(f"Sending prompt {prompt_id} to ChatGPT...")

#     response = openai_client.chat.completions.create(
#         model="gpt-4.1-mini",
#         messages=[
#             {"role": "system", "content": "You are a neutral assistant. Answer objectively."},
#             {"role": "user", "content": prompt_text}
#         ],
#         temperature=0.7
#     )

#     answer = response.choices[0].message.content

#     chatgpt_results.append({
#         "prompt_id": prompt_id,
#         "model_name": "ChatGPT",
#         "response_text": answer
#     })

# # Ergebnisse speichern (append)
# chatgpt_df = pd.DataFrame(chatgpt_results)
# chatgpt_df.to_csv(
#     OUTPUT_FILE,
#     index=False,
#     mode="a",
#     header=not os.path.exists(OUTPUT_FILE)
# )

# print("ChatGPT test completed.")


# # # --------------------------------------------------
# # # CLAUDE TEST
# # # --------------------------------------------------

# claude_results = []

# for _, row in prompts_df.iterrows():
#     prompt_id = row["prompt_id"]
#     prompt_text = row["prompt_text"]

#     print(f"Sending prompt {prompt_id} to Claude...")

#     response = claude_client.messages.create(
#         model="claude-sonnet-4-5-20250929",
#         max_tokens=500,
#         temperature=0.7,
#         system="You are a neutral assistant. Answer objectively.",
#         messages=[
#             {"role": "user", "content": prompt_text}
#         ]
#     )

#     answer = response.content[0].text

#     claude_results.append({
#         "prompt_id": prompt_id,
#         "model_name": "Claude",
#         "response_text": answer
#     })

# # Ergebnisse speichern (append)
# claude_df = pd.DataFrame(claude_results)
# claude_df.to_csv(
#     OUTPUT_FILE,
#     index=False,
#     mode="a",
#     header=False
# )

# print("Claude test completed.")


# # # --------------------------------------------------
# # # deepseek
# # --------------------------------------------------
# for _, row in prompts_df.iterrows():
#     prompt_id = row["prompt_id"]
#     prompt_text = row["prompt_text"]

#     print(f"Sending prompt {prompt_id} to DeepSeek...")

#     payload = {
#         "model": "deepseek-chat",
#         "messages": [
#             {"role": "system", "content": "You are a neutral assistant. Answer objectively."},
#             {"role": "user", "content": prompt_text}
#         ],
#         "temperature": 0.7
#     }

#     response = requests.post(
#         DEEPSEEK_API_URL,
#         headers=DEEPSEEK_HEADERS,
#         json=payload,
#         timeout=60
#     )

#     try:
#         response.raise_for_status()
#         answer = response.json()["choices"][0]["message"]["content"]
#     except Exception as e:
#         answer = f"[ERROR] {str(e)} | Raw: {response.text}"

#     row_df = pd.DataFrame([{
#         "prompt_id": prompt_id,
#         "model_name": "DeepSeek",
#         "response_text": answer
#     }])

#     row_df.to_csv(
#         OUTPUT_FILE,
#         index=False,
#         mode="a",
#         header=not os.path.exists(OUTPUT_FILE)
#     )


# print("DeepSeek test completed.")
# --------------------------------------------------
# xai
# --------------------------------------------------


grok_results = []

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
            },    {
            "role": "user",
            "content": prompt_text
        }
        ],

        "temperature": 0.7,
        "stream": False
    }

    response = requests.post(
        XAI_API_URL,
        headers=XAI_HEADERS,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    answer = response.json()["choices"][0]["message"]["content"]

    grok_results.append({
        "prompt_id": prompt_id,
        "model_name": "Grok",
        "response_text": answer
    })

grok_df = pd.DataFrame(grok_results)
grok_df.to_csv(
    OUTPUT_FILE,
    index=False,
    mode="a",
    header=False
)

print("Grok test completed.")


print("All tests finished successfully.")
