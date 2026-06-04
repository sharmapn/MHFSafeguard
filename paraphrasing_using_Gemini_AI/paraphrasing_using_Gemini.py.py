import os
import google.generativeai as genai
import re
import sqlite3
import json
import time
import nltk
#from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

# Download sentence tokenizer
nltk.download('punkt')




# Load API key
api_key = ""
if not api_key:
    raise ValueError("API key not found. Please set your API_KEY in the environment.")

# Configure Gemini API
genai.configure(api_key=api_key)

# Define generation config
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Define model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    safety_settings=[
        {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ],
)

# Database path
db_path = "../databases/dec-24/all_datasets_labelled.db"

# Full system prompt with Part 3
system_prompt = """
Task: Generate diverse sentences for machine learning training.
Input Sentence: {sentence}
Keyword(s): {keywords}
Label: {label}

Instructions:

PART 1 – Contextually Aligned Paraphrases:
1. Generate 10 diverse sentences that are structurally and stylistically different from the input.
2. Each sentence must retain the keyword(s) exactly as they appear.
3. The context and function of the keyword(s) should remain the same as in the input sentence.
4. Do not copy the wording, phrasing, or sentence structure from the input sentence.
5. The output should be plausible, emotionally coherent, and human-like.

PART 2 – Diverse Situations (Same Context):
6. Generate 10 sentences where the keyword(s) continue to represent self-harm, suicide, or harm to others (matching the label), but place the sentence in a different real-life scenario or setting.
7. The new sentences should vary the physical location, time of day, people involved, or emotional triggers — but the keyword(s) must still be used as a method or action of self-harm, suicide, or harming others.
8. The sentences must not reuse wording from the input and should describe unique, realistic situations.

PART 3 – Distraction Sentence Generation:
9. Assess whether the given keyword is inherently dangerous or could promote harmful behavior even in non-harmful contexts.
10. If the keyword is inherently dangerous (e.g., poisons, weapons, drugs, methods or actions of self-harm), do not generate distraction sentences. Respond with:
    {{"status": "dangerous", "message": "This keyword is too dangerous to be used in distraction sentences."}}
11. If the keyword is safe for neutral or positive contexts, generate 10 non-harmful, everyday sentences using the keyword.
12. The sentences must NOT imply self-harm, suicide, violence, criminal behavior, accidents, death, or any harmful acts.
13. Focus on using the keyword in neutral or positive settings such as cooking, school, work, hobbies, or sports.

Output Format:
Return a JSON object:
{{
  "similar_context": [ list of 10 paraphrased sentences preserving keyword usage and similar scenario ],
  "diverse_situations_same_context": [ list of 10 sentences using keywords in the same self-harm or suicide context but in different life situations ],
  "distraction_sentences": {{ "status": "dangerous" OR "safe", "sentences": [ list of 10 distraction sentences if safe ] }}
}}
"""

def call_google_gemini(sentence, keywords, label):
    """Generate paraphrased and distraction sentences using Gemini."""
    chat = model.start_chat(history=[])
    prompt = system_prompt.format(sentence=sentence, keywords=', '.join(keywords), label=label)

    try:
        response = chat.send_message(prompt)

        # Log the raw response
        print("\n🔍 Raw Gemini Response:\n", response.text)

        clean_text = re.sub(r"```(json)?", "", response.text).strip()
        parsed = json.loads(clean_text)
        return parsed

    except genai.types.generation_types.BlockedPromptException as e:
        print(f"Prompt blocked: {e}")
    except json.JSONDecodeError:
        print("\n❌ JSON Decode Error. Raw Response:")
        print(response.text)
    except Exception as e:
        print(f"General error: {e}")
    return {}

def read_update_database():
    """Read labeled rows from the database and insert Gemini-generated paraphrases."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = """
        SELECT id, sentence, label, keyword 
        FROM MH_forum_388_sentences 
        WHERE label IN (
            'Suicide or Self Harm Ideation', 
            'Method or action of Suicide, Self-Harm or Harming others'
        )
        AND id NOT IN (SELECT id FROM paraphrases4)

        UNION ALL

        SELECT id, sentence, first_label AS label, keywords_phrases_method_action AS keyword 
        FROM SuicideAndDepressionDetectionKaggleDataset_classified_sentences 
        WHERE first_label IN (
            'Suicide or Self Harm Ideation', 
            'Method or action of Suicide, Self-Harm or Harming others'
        )
        AND id NOT IN (SELECT id FROM paraphrases4)

        ORDER BY id
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    for row in rows:
        try:
            original_id, sentence, label, keyword_str = row

            if not sentence or not label or not keyword_str:
                print(f"Skipping post {original_id} due to missing data.")
                continue

            required_keywords = [k.strip().lower() for k in keyword_str.split(',') if k.strip()]
            print(f"\nProcessing ID {original_id}, Keywords: {required_keywords}")

            paraphrase_result = call_google_gemini(sentence.strip(), required_keywords, label)

            print(f"\nOriginal Sentence [ID {original_id}]: {sentence}")
            total_inserted = 0

            # Insert similar_context and diverse_situations_same_context
            for context_type in ["similar_context", "diverse_situations_same_context"]:
                paraphrases = paraphrase_result.get(context_type, [])
                if not isinstance(paraphrases, list):
                    continue

                print(f"\nGenerated Paraphrases ({context_type.replace('_', ' ').title()}):")
                for i, para in enumerate(paraphrases, 1):
                    if not para or not isinstance(para, str):
                        continue
                    print(f"  {i}. {para}")

                    cursor.execute(
                        "INSERT INTO paraphrases4 (id, sentence, paraphrases, label, keywords, context_type, gemini_raw_response) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (original_id, sentence, para.strip(), 'Method or action of Suicide, Self-Harm or Harming others', keyword_str, context_type, paraphrase_result)
                    )
                    total_inserted += 1

            # Insert distraction_sentences if available
            distraction_info = paraphrase_result.get("distraction_sentences", {})
            if distraction_info.get("status") == "safe":
                distraction_sentences = distraction_info.get("sentences", [])
                for i, para in enumerate(distraction_sentences, 1):
                    if not para or not isinstance(para, str):
                        continue
                    print(f"  Distraction {i}: {para}")

                    cursor.execute(
                        "INSERT INTO paraphrases4 (id, sentence, paraphrases, label, keywords, context_type) VALUES (?, ?, ?, ?, ?, ?)",
                        (original_id, sentence, para.strip(), 'Not Suicide', keyword_str, "distraction_sentences")
                    )
                    total_inserted += 1
            else:
                print(f"Distraction sentences not generated for ID {original_id} (keyword flagged as dangerous).")

            if total_inserted:
                conn.commit()
                print(f"✅ Inserted {total_inserted} sentences for post ID {original_id}")
            else:
                print(f"⚠️ No valid sentences returned for ID {original_id}")

        except Exception as e:
            print(f"❌ Error processing post ID {row[0]}: {e}")
            time.sleep(2)
            continue

    conn.close()
    print("Database connection closed.")

if __name__ == "__main__":
    read_update_database()
