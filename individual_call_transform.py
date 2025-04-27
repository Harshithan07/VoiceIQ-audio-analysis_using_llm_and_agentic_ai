import streamlit as st
import pandas as pd
import textwrap
import requests
from datetime import datetime
import json
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

# -------------------------------------------------------------
# Class Overview:
# 
# This class provides end-to-end functionality for summarizing 
# customer support call transcripts, extracting key insights, 
# generating sentiment charts, and exporting structured reports.
# -------------------------------------------------------------

# Main Functions:

# 1. summarize_from_csv
#    - Summarizes a call transcript from a CSV file and generates JSON and HTML reports.

# 2. generate_sentiment_charts
#    - Creates sentiment distribution and fluctuation charts based on the conversation transcript.

# 3. fig_to_base64
#    - Converts a matplotlib figure into a Base64-encoded image string for embedding into HTML reports.

# 4. export_summary_html
#    - Exports the call analysis summary and sentiment charts into a styled HTML report.

# 5. extract_keywords_llm
#    - Extracts 3–5 important single-word keywords from the generated call summary using an LLM.

# 6. extract_intent_llm
#    - Classifies the customer's main intent from the call summary into predefined intent categories.

# 7. extract_sentiment_llm
#    - Determines the sentiment (positive, neutral, or negative) for either the customer or representative.

# 8. save_summary_json
#    - Saves the summarized call information and metadata into a structured JSON file 
#      (to support further dashboard development and trend analysis).

# --- Summarizer Class ---
class OpenRouterTranscriptSummarizer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://chat.openrouter.ai"
        }
        self.model = "openchat/openchat-3.5-0106"

    def summarize_from_csv(self, csv_input, text_column="Text", max_tokens=512, call_id="call_001", representative="Agent_Unknown"):
        if not isinstance(csv_input, str):
            csv_input.seek(0)
        df = pd.read_csv(csv_input)
        text = " ".join(df[text_column].dropna().tolist())[:3000]
        prompt = (
            "Summarize this customer support conversation into 2–3 meaningful sentences. "
            "Focus on what the customer wanted and how the representative responded.\n\n"
            f"Transcript:\n{text}"
        )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }

        summaries = {}
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            result = response.json()

            if isinstance(result, dict) and "choices" in result:
                summary_text = result["choices"][0]["message"]["content"].strip()
                keywords = self.extract_keywords_llm(summary_text)
                intent = self.extract_intent_llm(summary_text)
                sentiment_customer = self.extract_sentiment_llm(summary_text, speaker="customer")
                sentiment_representative = self.extract_sentiment_llm(summary_text, speaker="representative")

                summaries[self.model] = {
                    "summary": summary_text,
                    "keywords": keywords,
                    "intent": intent,
                    "sentiment_customer": sentiment_customer,
                    "sentiment_representative": sentiment_representative
                }

                self.save_summary_json(
                    call_id=call_id,
                    representative=representative,
                    summary=summary_text,
                    keywords=keywords,
                    intent=intent,
                    sentiment_customer=sentiment_customer,
                    sentiment_representative=sentiment_representative,
                    model=self.model
                )

                chart_base64_1, chart_base64_2 = self.generate_sentiment_charts(df)

                self.export_summary_html(
                    summaries[self.model],
                    file_path=f"./reports/html/{call_id}_{self.model.replace('/', '_')}.html",
                    title=call_id,
                    chart_base64_list=[chart_base64_1, chart_base64_2]
                )

        except Exception as e:
            summaries[self.model] = {
                "summary": f"Request failed: {str(e)}",
                "keywords": [],
                "intent": "N/A",
                "sentiment_customer": "N/A",
                "sentiment_representative": "N/A"
            }

        return summaries

    def generate_sentiment_charts(self, df):
        sentiments = []
        for _, row in df.iterrows():
            content = row['Text']
            prompt = (
                f"Classify the sentiment of the following customer support utterance:\n\n"
                f"{content}\n\nChoose one of: positive, neutral, negative."
            )
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10
            }
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload)
                result = response.json()
                label = result['choices'][0]['message']['content'].strip().lower()
                if "positive" in label:
                    label = "positive"
                elif "negative" in label:
                    label = "negative"
                elif "neutral" in label:
                    label = "neutral"
                else:
                    label = "neutral"
            except:
                label = 'neutral'
            sentiments.append(label)

        df['Sentiment'] = sentiments
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        df['SentimentScore'] = df['Sentiment'].map(sentiment_map)
        df['Exchange'] = range(1, len(df) + 1)

        # Chart 1: Distribution
        # Define custom color palette
        palette = {
            'positive': '#2ecc71',   # emerald green
            'neutral':  '#3498db',   # peter river blue
            'negative': '#e74c3c'    # alizarin red
        }

        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sns.countplot(data=df, x='Speaker', hue='Sentiment', ax=ax1, palette=palette)
        ax1.set_title('Overall Sentiment Distribution by Speaker', fontsize=14, weight='bold')
        ax1.set_ylabel('Number of Turns', fontsize=12)
        ax1.set_xlabel('Speaker', fontsize=12)
        ax1.tick_params(axis='x', labelsize=11)
        ax1.tick_params(axis='y', labelsize=11)
        ax1.legend(title='Sentiment', title_fontsize=11, fontsize=10)

        chart_base64_1 = self.fig_to_base64(fig1)
        plt.close(fig1)

        # Chart 2: Fluctuation
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for speaker, color in zip(['Representative', 'Customer'], ['orange', 'blue']):
            speaker_df = df[df['Speaker'] == speaker]
            ax2.plot(speaker_df['Exchange'], speaker_df['SentimentScore'], label=f'{speaker} Trend', color=color)
            ax2.scatter(speaker_df['Exchange'], speaker_df['SentimentScore'], color=color, alpha=0.6)
        ax2.axhline(0, color='gray', linestyle='--', label='Neutral Baseline')
        ax2.set_title('Sentiment Fluctuation Over Conversation by Speaker')
        ax2.set_xlabel('Conversation Exchange')
        ax2.set_ylabel('Sentiment Score (-1 to 1)')
        ax2.legend()
        chart_base64_2 = self.fig_to_base64(fig2)
        plt.close(fig2)

        return chart_base64_1, chart_base64_2

    def fig_to_base64(self, fig):
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        return encoded


    def export_summary_html(self, content: dict, file_path: str, title: str, chart_base64_list=[]):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        import random
        import datetime

        # Metadata generation
        call_id = f"CALL-{random.randint(1000, 9999)}"
        customer_id = f"CUST-{random.randint(10000, 99999)}"
        representative_id = random.choice(["Agent_A", "Agent_B", "Agent_C", "Agent_D"])
        today = datetime.date.today()

        html = f"""
        <html>
        <head>
            <title>Call Analysis Report – {content['intent'].replace('_', ' ').title()}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metadata-box {{
                    background-color: #f4f4f4;
                    border-left: 5px solid #0077b6;
                    padding: 10px 20px;
                    margin-bottom: 20px;
                    font-size: 14px;
                }}
                .quick-insights {{
                    background-color: #fdfcdc;
                    border-left: 5px solid #ffb703;
                    padding: 15px 20px;
                    margin-bottom: 20px;
                    font-size: 15px;
                }}
                summary {{
                    font-weight: bold;
                    margin-top: 15px;
                    font-size: 16px;
                }}
                details {{
                    margin-bottom: 20px;
                    border: 1px solid #ccc;
                    padding: 10px;
                    border-radius: 4px;
                    background: #fafafa;
                }}
                button {{
                    background-color: #0077b6;
                    color: white;
                    padding: 10px 16px;
                    border: none;
                    border-radius: 4px;
                    font-size: 14px;
                    cursor: pointer;
                }}
                button:hover {{
                    background-color: #005f86;
                }}
            </style>
            <script>
                function printPage() {{
                    window.print();
                }}
            </script>
        </head>
        <body>
            <h1>📞 Call Analysis Report</h1>
            <div class="metadata-box">
                <b>Call ID:</b> {call_id} |
                <b>Date:</b> {today} |
                <b>Customer ID:</b> {customer_id} |
                <b>Representative:</b> {representative_id}
            </div>

            <div class="quick-insights">
                <b>Quick Insights:</b>
                <ul>
                    <li><b>Intent:</b> {content['intent'].replace('_', ' ').title()}</li>
                    <li><b>Customer Sentiment:</b> {content['sentiment_customer'].title()}</li>
                    <li><b>Representative Sentiment:</b> {content['sentiment_representative'].title()}</li>
                    <li><b>Keywords:</b> {', '.join(content['keywords'])}</li>
                </ul>
            </div>

            <details open>
                <summary>📝 Full Summary</summary>
                <p>{content['summary']}</p>
            </details>

            <details open>
                <summary>📊 Charts</summary>
        """

        for chart_base64 in chart_base64_list:
            html += f"<img src='data:image/png;base64,{chart_base64}' style='max-width:100%; margin-bottom:20px;'><br>"

        html += """
            </details>
            <button onclick="printPage()">🖨️ Print to PDF</button>
        </body>
        </html>
        """

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html)





    def extract_keywords_llm(self, summary_text):
        prompt = (
            "Extract 3–5 important keywords from this summary. Each keyword has to be a single word, not a phrase. **Do not extract more than 5 important keywords**"
            "Return only a comma-separated list, no explanation.\n\n"
            f"Summary:\n{summary_text}"
        )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            result = response.json()
            keyword_str = result['choices'][0]['message']['content'].strip()
            return [kw.strip() for kw in keyword_str.split(',') if kw.strip()]
        except:
            return []

    def extract_intent_llm(self, summary_text):
        prompt = (
            "Classify the customer's intent from this summary.\n"
            "Choose one of the following intents: Refill_Request, Billing_Issue, Medication_Change, Delivery_Status, "
            "Side_Effect, Doctor_Contact, General_Inquiry.\n"
            f"Summary:\n{summary_text}\nIntent:"
        )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except:
            return "General_Inquiry"

    def extract_sentiment_llm(self, summary_text, speaker):
        prompt = (
            f"Classify the sentiment of the {speaker} in this summary.\n"
            "Only respond with ONE of the following words:\n"
            "- positive\n"
            "- neutral\n"
            "- negative\n\n"
            "DO NOT explain. Just reply with one word only.\n"
            "Example Output:\npositive\n\n"
            f"Summary:\n{summary_text}\nSentiment:"
        )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 20,
            "temperature": 0
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            result = response.json()
            return result['choices'][0]['message']['content'].strip().lower()
        except:
            return "neutral"

    def save_summary_json(self, call_id, representative, summary, keywords, intent,
                      sentiment_customer, sentiment_representative, model, df):
        import random

        # Generate customer ID
        customer_id = f"CUST-{random.randint(10000, 99999)}"

        # Extract ending sentiment from customer's last utterance
        customer_df = df[df['Speaker'].str.lower() == "customer"]
        if not customer_df.empty:
            ending_sentiment = customer_df.iloc[-1]['Sentiment']
        else:
            ending_sentiment = "neutral"

        result = {
            "call_id": call_id,
            "customer_id": customer_id,
            "representative": representative,
            "summary": summary,
            "keywords": keywords,
            "intent": intent,
            "sentiment_customer": sentiment_customer,
            "sentiment_representative": sentiment_representative,
            "sentiment_ending": ending_sentiment,
            "model": model,
            "timestamp": datetime.now().isoformat()
        }

        os.makedirs("./reports/json", exist_ok=True)
        path = f"./reports/json/{call_id}_{model.replace('/', '_')}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)