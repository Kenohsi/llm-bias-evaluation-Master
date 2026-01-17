# LLM Bias Evaluation - Master Thesis

Dieses Repository enthält den Quellcode und die Rohdaten zur Masterarbeit **"Ethik und Bias in KI-Sprachmodellen"**.

##  Struktur
* **`src/`**: Python-Skripte zur automatisierten Datenerhebung über die APIs von OpenAI, Anthropic, DeepSeek und xAI.
* **`prompts/`**: Das entwickelte Testset mit 20 Prompts (Baseline vs. Adversarial) in den Kategorien Politik, Wirtschaft, Kultur, Gender und Sicherheit.
* **`responses/`**: Die generierten Rohantworten der Modelle (`model_responses.csv`).
* **`analysis/`**: Jupyter Notebooks zur qualitativen und quantitativen Auswertung der Daten.

##  Reproduktion
1. Installieren der Abhängigkeiten: `pip install openai anthropic pandas seaborn`
2. API-Keys in einer `.env` Datei hinterlegen.
3. Datenerhebung starten: `python src/run_models.py`
4. Analyse ausführen: Notebook in `analysis/` öffnen.

##  Methodik
Die Auswertung erfolgt anhand eines triangulierten Ansatzes aus:
1. **Automatisierte Metriken:** Compliance-Rate und Refusal-Rate.
2. **Qualitative Inhaltsanalyse:** Manuelle Codierung nach *Stance* (Positionierung) und *Tone* (Tonalität).
