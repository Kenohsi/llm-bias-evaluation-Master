# Ethik und Bias in KI-Sprachmodellen
### Ein empirischer Vergleich ausgewählter Large Language Models

**Masterarbeit** | Kenan Husic | FH Campus Wien – MSc Software Design and Engineering
**Betreuer:** DI Dr. techn Mugdim Bublin | **Eingereicht:** 13.01.2026

---

## Überblick

Dieses Repository enthält den vollständigen Quellcode, die Rohdaten und die Analyseergebnisse zur Masterarbeit. Untersucht wird, ob und wie vier Large Language Models (LLMs) bei ethisch zugespitzten Prompts unterschiedliches Compliance- und Stilverhalten zeigen.

**Untersuchte Modelle:**
| Modell | Anbieter | Version |
|--------|----------|---------|
| ChatGPT | OpenAI | GPT-4.1-mini |
| Claude | Anthropic | Claude Sonnet |
| DeepSeek | DeepSeek AI | deepseek-chat |
| Grok | xAI | grok-4 |

---

## Zentrale Ergebnisse

| Modell | Compliance-Rate | Adversarial-Compliance | Ø Wörter | Stil |
|--------|----------------|----------------------|----------|------|
| **Grok** | **100 %** | **100 %** | 794 | Zustimmend / Aggressiv |
| ChatGPT | 95 % | 90 % | 300 | Ausgewogen |
| Claude | 85 % | 70 % | 239 | Selektiv / Vorsichtig |
| DeepSeek | 75 % | 50 % | 289 | Zustimmend / Bürokratisch |

---

## Struktur

```
llm-bias-evaluation/
├── src/
│   └── run_tests.py          # Datenabruf via OpenAI, Anthropic, DeepSeek, xAI APIs
├── prompts/
│   └── prompts.csv           # 20 Prompts (10 Kategorien × Baseline + Adversarial)
├── responses/
│   ├── model_responses.csv           # Rohdaten: 80 Modellantworten
│   └── model_responses_classified.csv # Klassifizierte Antworten
├── analysis/
│   └── analysis.ipynb        # Vollständiges Analyse-Notebook (ausgeführt, mit Outputs)
└── requirements.txt
```

### Prompt-Kategorien
Politik (Zensur) · Wirtschaft (System) · Kultur (Familie) · Gender (Identität) · Sicherheit (Überwachung) · Migration · Geschichte (Kolonialismus) · Klimawandel · Justiz (Strafe) · KI-Rechte

---

## Methodik

Die Auswertung basiert auf einem triangulierten Ansatz:

**1. Automatisierte Klassifikation**
- **Hard Refusal:** Explizite Ablehnung (<250 Wörter + Ablehnungsphrase)
- **Soft Refusal:** Diplomatisches Ausweichen (Hedging-Phrasen)
- **Compliance:** Vollständige Aufgabenerfüllung

**2. Keyword-basierte Stilanalyse**
- Aggressive/alarmistische Sprachdichte
- Institutionell-bürokratische Sprachdichte
- Sycophancy-Score (Übernahme des Prompt-Framings)

**3. Weitere Metriken**
- Kosinus-Ähnlichkeit (TF-IDF) zwischen Modellen
- Sentiment-Analyse (VADER Compound Score)
- Antwortlängenverteilung

---

## Visualisierungen

Das Notebook erzeugt folgende Abbildungen:

| Datei | Inhalt |
|-------|--------|
| `barchart_refusal_rate.png` | Compliance & Refusal-Rate pro Modell |
| `baseline_vs_adversarial.png` | Compliance-Vergleich: Baseline vs. Adversarial |
| `category_compliance_heatmap.png` | Compliance pro Thema & Modell |
| `refusal_types_stacked.png` | Hard vs. Soft Refusal im Detail |
| `word_count_boxplot.png` | Antwortlängenverteilung |
| `keyword_heatmap.png` | Signalworthäufigkeit (lexikalische Analyse) |
| `radar_chart_final.png` | Empirisches Modellprofil (5 Dimensionen) |
| `cosine_similarity.png` | TF-IDF Kosinus-Ähnlichkeit zwischen Modellen |
| `sentiment_analysis.png` | VADER Sentiment Baseline vs. Adversarial |

---

## Reproduktion

### 1. Abhängigkeiten installieren
```bash
pip install openai anthropic pandas matplotlib seaborn nltk scikit-learn python-dotenv requests
```

### 2. API-Keys konfigurieren
```bash
# .env Datei im Projekt-Root erstellen:
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
DEEPSEEK_API_KEY=...
XAI_API_KEY=...
```

### 3. Datenerhebung (optional – Rohdaten bereits vorhanden)
```bash
python src/run_tests.py
```

### 4. Analyse ausführen
```bash
jupyter notebook analysis/analysis.ipynb
```

---

## Literatur (Auswahl)

- Floridi, L., & Cowls, J. (2019). A unified framework of five principles for AI in society. *Harvard Data Science Review*, 1(1). https://doi.org/10.1162/99608f92.8cd550d1
- Nielsen, A. (2020). *Practical Fairness*. O'Reilly Media.
- Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 35*, 27730–27744.
- European Parliament & Council. (2024). Regulation (EU) 2024/1689 (AI Act).
