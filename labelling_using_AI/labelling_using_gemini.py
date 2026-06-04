
import sqlite3
import os
import time
import json
import google.generativeai as genai
from dotenv import load_dotenv
import re

# additional imports for those messages that gave error
import sqlite3
import json
import re
import time
import nltk
from bs4 import BeautifulSoup
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
MAX_WORDS_PER_CHUNK = 1000  # Can adjust based on API token limitss

# new script to classify theos emessages whoch were not classified

## 227036
#select count (distinct post_id) from SuicideAndDepressionDetectionKaggleDataset_classified_sentences
## 232074
#select count (distinct id) from SuicideAndDepressionDetectionKaggleDataset
## 3000 less

# Load environment variables
load_dotenv()
api_key = "AIzaSyCz7Sq8EJeGZ2WOFmC4RAULMBf5A5xPfKg" #os.getenv("API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set your API_KEY in the environment.")

# Configure Google Gemini API
genai.configure(api_key=api_key)

# Define model and generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", #"gemini-1.5-flash", gemini-1.5-pro",
    generation_config=generation_config,
    safety_settings=[
        {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ],
)

# Database configuration
db_path = "../databases/dec-24/all_datasets_labelled.db"
source_db_table = "SuicideAndDepressionDetectionKaggleDataset"
classified_table = "SuicideAndDepressionDetectionKaggleDataset_classified_sentences"
#classified_table="SuicideAndDepressionDetectionKaggleDataset_classified_sentences"

# System prompt for the model
system_prompt = """
Analyze the following mental health forum message sentence-by-sentence and output the results in JSON format.

First **Preprocess the text**:
   - Remove all HTML tags and decode any HTML entities (e.g., &amp;, &lt;, etc.).
   - Ensure the text is clean, readable, and properly encoded in UTF-8.
   - Split the cleaned text into **sentences**, even across paragraphs.
   - If the text is very long, process it in **chunks of up to 1000 words** (based on complete sentences).

Then, structure each sentence's analysis in the JSON response with:
{
    "sentence": "The text of each sentence",
    "first_label": "Assign a primary label for the sentence from ['Methods or actions of Suicide, Self Harm or harming others' - any method, action, technique or way mentioned in the sentence, 'Suicide or Self Harm Ideation' - only if the sentence author has suicide or self harm ideation, not for others they may be referred to in the sentence and do not include ideation from the past, 'Depression' - only if the sentence author has depresion tendencies and mentions this in the sentence, not depression of others they are referring to in the sentence and do not include depression from the past, 'Advice' - anyone giving any sort of advice against self harm and suicide, or all the rest sentences as 'Not Suicide post']. Apart from 'Methods or actions of Suicide, Self Harm or harming others' and 'Not Suicide post', the remaining labels should only be assigned based on if they are directly the thoughts of the sentence author, not other's thoughts shared in the sentence or any ideation from the past",
    
    "keywords_phrases_method_action": "From the sentence, identify any keywords or phrases that are 'Methods or actions of suicide, self-harm, or harming others' or 'Suicide or Self Harm Ideation'", 
    "keywords_phrases_ideation": "From the sentence, identify any keywords or phrases that are 'Suicide or Self Harm Ideation'", 
    "mid_label" : "classify the keywords_phrases_method_action according to the following 'Suicide Methods or actions' or 'Methods or actions of Self-Harm or harming others' categories where applicable, or create a category and sub-category if you coem across a new one:",
    
    methods = {
        "Suicide Methods or actions": {
            "Firearms": ["Handguns", "Rifles", "Shotguns", "Other firearms"],
            "Sharp Objects": ["Knives", "Razor blades", "Scissors", "Glass", "Other sharp objects"],
            "Suffocation": ["Hanging", "Strangulation", "Suffocation by plastic bag"],
            "Jumping": ["From height (building, bridge, cliff)", "In front of moving vehicle"],
            "Poisoning": [
                "Overdose of prescription medication",
                "Overdose of illicit drugs",
                "Ingestion of household chemicals",
                "Plant-based poisons",
                "Animal-based poisons",
                "Other poisons"
            ],
            "Drowning": ["Immersion in water (ocean, river, lake, bathtub)"],
            "Gas Poisoning": ["Carbon monoxide poisoning", "Other gas poisoning"],
            "Other Violent Methods": [
                "Blunt force trauma",
                "Fire",
                "Electrical shock",
                "Exposure to extreme temperatures"
            ],
            "Starvation": ["Deliberate food deprivation"],
            "Dehydration": ["Deliberate water deprivation"],
            "Medical Refusal": ["Withholding necessary medical treatment"],
            "Exposure to Elements": ["Exposure to extreme weather conditions"]
            "Other method": ["Include any other method not listed above"]
        },
        "Methods or actions of Self-Harm or harming others": {
            "Cutting and Scratching": ["Using sharp objects to injure skin"],
            "Burning": ["Using fire", "Hot objects", "Chemicals to injure skin"],
            "Hitting": ["Punching", "Slapping", "Head banging"],
            "Hair Pulling": ["Trichotillomania"],
            "Skin Picking": ["Dermatillomania"],
            "Biting": ["Self-inflicted bite wounds"],
            "Bone Breaking": ["Intentionally breaking bones"],
            "Object Insertion": ["Inserting objects into body orifices"],
            "Tattooing or Piercing": ["Without proper sterilization"],
            "Chemical Burns": ["Using chemicals to burn skin"]
            "Other method or action ": ["Include any other method or action not listed above"]
        }
    },
    "next_label" : "classify the keywords_phrases_ideation" into 'General Suicidal Ideation (No Specific Action)', 'Passive Suicidal Ideation (No Intention or Plan)', 'Self-Harm Ideation (No Action Stated)', 'Thoughts of Death Without Suicidal Intent', or 'Existential Crisis / Hopelessness', or create a category and sub-category if needed",
    

    "second_label": "Secondary classification from the following: 'Wish to be Dead' - This suicidal ideation is about a wish to be dead or not alive anymore, or a wish to fall asleep and not wake up. For example, a person would say - I wish I were dead., 'Non-specific Active Suicidal Thoughts' - This suicidal ideation is about general non-specific thoughts of wanting to end ones life or committing suicide without ways to kill oneself or associated methods, intent, or plan. For example, a person would say - I have thought about killing myself., 'Active Suicidal Ideation with Any Methods (Not Plan) without Intent to Act' - This suicidal ideation is about thoughts of suicide with at least one method but without a specific plan, time, place, and method details. For example, a person would say - I thought about taking an overdose, but I never made a specific plan as to when, where, or how I would actually do it…and I would never go through with it., 'Active Suicidal Ideation with Some Intent to Act, without Specific Plan' - This suicidal ideation is about active thoughts of killing oneself and having some intent to act on such thoughts. For example, a person would say - I have thought about killing myself and I definitely will do something about it. , 'Active Suicidal Ideation with Specific Plan and Intent' - This suicidal ideation is about thoughts of killing oneself with details of a fully or partially worked out plan and having some intent to carry it out. For example, a person would say - I will end all my sufferings tomorrow with those little pills.' , 'Not suicide sentence' - all other sentences "
}

Provide the analysis for each sentence in this JSON format.
"""


# Updated System Prompt for Gemini
# system_prompt = """
# You are analyzing raw mental health forum posts. These posts may contain HTML tags, special characters, inconsistent formatting, or be very long.

# Your tasks:
# 1. **Preprocess the text**:
#    - Remove all HTML tags and decode any HTML entities (e.g., &amp;, &lt;, etc.).
#    - Ensure the text is clean, readable, and properly encoded in UTF-8.
#    - Split the cleaned text into **sentences**, even across paragraphs.
#    - If the text is very long, process it in **chunks of up to 1000 words** (based on complete sentences).

# 2. **Analyze each sentence** and return the analysis in JSON format.

# For each sentence, return a dictionary structured like this:
# {
#     "sentence": "The cleaned and clearly split sentence.",
#     "first_label": "Assign one primary label from: ['Methods or actions of Suicide, Self Harm or harming others', 'Suicide or Self Harm Ideation', 'Depression', 'Advice', 'Not Suicide post'] — follow the detailed rules below.",
#     "keywords_phrases_method_action": "List any relevant phrases indicating methods/actions of suicide, self-harm, or harm to others.",
#     "keywords_phrases_ideation": "List phrases indicating suicidal or self-harm ideation.",
#     "mid_label": "Classify method/action keywords into the following categories or create one: [Suicide Methods or actions, Methods or actions of Self-Harm or harming others] with appropriate sub-categories.",
#     "next_label": "Classify ideation keywords into one of the following or a new category: ['General Suicidal Ideation (No Specific Action)', 'Passive Suicidal Ideation (No Intention or Plan)', 'Self-Harm Ideation (No Action Stated)', 'Thoughts of Death Without Suicidal Intent', 'Existential Crisis / Hopelessness']",
#     "second_label": "Assign one of the following: ['Wish to be Dead', 'Non-specific Active Suicidal Thoughts', 'Active Suicidal Ideation with Any Methods (Not Plan) without Intent to Act', 'Active Suicidal Ideation with Some Intent to Act, without Specific Plan', 'Active Suicidal Ideation with Specific Plan and Intent', 'Not suicide sentence']"
# }

# Make sure to process and return results for each sentence individually, in a JSON list format.
# """


def call_google_gemini(post_text):
    # Start a chat session with the model
    chat_session = model.start_chat(
        history=[{"role": "user", "parts": ["I am working on moderating a mental health forum. " + system_prompt]}]
    )

    try:
        # Send message and include system prompt once
        response = chat_session.send_message(post_text)

        if response:
            #print('####\nResponse Text: ', response.text)

             # Clean response text
            raw_text = response.text
            #print(f"####\nRaw Response Text: {raw_text}")  # Log raw text for debugging

            # Remove backticks and language hints
            clean_text = re.sub(r"```(json)?", "", raw_text).strip()

             # Attempt to parse JSON
            parsed_result = json.loads(clean_text)
       
        return parsed_result  # Expecting a list of dictionaries, one per sentence

    except genai.types.generation_types.BlockedPromptException as e:
        print(f"Prompt blocked due to: {e}")
        return []
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON response from API.")
        return []
    except Exception as e:
        print(f"Error calling API: {e}")
        return []
    

def call_google_gemini2(post_text, retries=3):
    response = None  # Initialize the response variable
    for attempt in range(retries):
        try:
            # Start a chat session
            chat_session = model.start_chat(
                history=[{"role": "user", "parts": ["I am working on moderating a mental health forum. " + system_prompt]}]
            )
            # Send the message
            response = chat_session.send_message(post_text)

            # Check if response is None
            if response is None:
                print(f"Attempt {attempt + 1}: Received no response from the API.")
                continue  # Retry

            # Clean and parse the response
            clean_text = response.text.strip("```json").strip("```").strip()
            parsed_result = json.loads(clean_text)
            return parsed_result

        except json.JSONDecodeError as json_error:
            print(f"JSON Decode Error: {json_error}")
            if response:
                print(f"Response Text: {response.text}")
        except genai.types.generation_types.BlockedPromptException as blocked_error:
            print(f"Prompt Blocked: {blocked_error}")
        except Exception as general_error:
            print(f"Error calling API on attempt {attempt + 1}: {general_error}")
            if response:
                print(f"Partial Response Text: {response.text}")
        # Exponential backoff for retries
        time.sleep(2 ** attempt)

    print(f"Failed to process post after {retries} retries.")
    return []  # Return an empty list if all attempts fail

# Function to read the database and update the classified_sentences table

# select max(post_id) from SuicideAndDepressionDetectionKaggleDataset_classified_sentences
# select * from SuicideAndDepressionDetectionKaggleDataset where id  = 126963



def clean_html_and_encode(text):
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text(separator=" ")
    # Ensure UTF-8 encoding
    return cleaned_text.encode('utf-8', errors='replace').decode('utf-8')

def split_into_paragraphs_and_sentences(text):
    paragraphs = re.split(r'\n{2,}', text.strip())
    all_sentences = []
    for para in paragraphs:
        if para.strip():
            try:
                sentences = sent_tokenize(para.strip())
                all_sentences.extend(sentences)
            except Exception:
                sentences = re.split(r'(?<=[.!?]) +', para.strip())
                all_sentences.extend(sentences)
    return all_sentences

def chunk_sentences(sentences, max_words=MAX_WORDS_PER_CHUNK):
    chunks, current_chunk, word_count = [], [], 0
    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, word_count = [], 0
        current_chunk.append(sentence)
        word_count += len(words)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def read_update_database():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Ensure the failed log table exists
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS failed_classification_log (
                post_id INTEGER PRIMARY KEY,
                raw_text TEXT,
                error_message TEXT
            )
        """)

        # SELECT id, text
        # FROM SuicideAndDepressionDetectionKaggleDataset
        # WHERE id NOT IN (
        #     SELECT post_id FROM SuicideAndDepressionDetectionKaggleDataset_classified_sentences
        # )
        # ORDER BY id ASC

        cursor.execute(f"""
            SELECT id, text 
            FROM {source_db_table} 
            WHERE id NOT IN (
                SELECT post_id FROM {classified_table}
            ) 
            AND id > 51721       
            ORDER BY id ASC
        """)
        # 1156
        # until 1156 are introductory posts, so we can skip them

        # cursor.execute(f"""
        #     SELECT s.id, s.text
        #     FROM {source_db_table} s
        #     LEFT JOIN {classified_table} c ON s.id = c.post_id
        #     WHERE c.post_id IS NULL
        #     ORDER BY s.id ASC
        # """)

        rows = cursor.fetchall()

        for row in rows:
            try:
                post_id = row[0]
                post_text = row[1]

                if not post_text:
                    print(f"Skipping empty post ID {post_id}")
                    continue

                print(f'\n#### Processing post ID {post_id}')

                try:
                    sentence_classifications = call_google_gemini(post_text.strip())
                    if not sentence_classifications:
                        print(f"No classifications returned for post ID {post_id}")
                        print("Response:" + str(sentence_classifications))
                        # we insert the resulst into the database
                        cursor.execute(
                                f"INSERT INTO {classified_table} (post_id, sentence, first_label, keywords_phrases_method_action, keywords_phrases_ideation, mid_label, next_label, second_label, full_response) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (post_id, post_text.strip(), "Error", "", "", "", "", "", sentence_classifications)
                            )
                        conn.commit()
                        print(f"Inserted classified sentence for post ID {post_id}")

                        continue
                    
                    for sentence_data in sentence_classifications:
                        try:
                            sentence = sentence_data.get("sentence", "")
                            if not sentence:
                                continue

                            first_label = sentence_data.get("first_label", "")
                            # json.dumps(sentence_data.get("keywords_phrases_method_action", []) or [])
                            keywords_phrases_method_action = json.dumps(sentence_data.get("keywords_phrases_method_action", []) or [])
                            keywords_phrases_ideation = json.dumps(sentence_data.get("keywords_phrases_ideation", []) or [])
                            mid_label = json.dumps(sentence_data.get("mid_label", []) or [])
                            next_label = json.dumps(sentence_data.get("next_label", []) or [])
                            second_label = sentence_data.get("second_label", "")
                            full_response = json.dumps(sentence_data or [])

                            cursor.execute(
                                f"INSERT INTO {classified_table} (post_id, sentence, first_label, keywords_phrases_method_action, keywords_phrases_ideation, mid_label, next_label, second_label, full_response) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (post_id, sentence, first_label, keywords_phrases_method_action, keywords_phrases_ideation, mid_label, next_label, second_label, full_response)
                            )
                            conn.commit()
                            print(f"Inserted classified sentence for post ID {post_id}")
                        except Exception as e:
                            print(f"Error inserting into database for post ID {post_id}: {e}")
                            conn.rollback()
                            continue


                except Exception as api_error:
                    print(f"Error calling Google Gemini API for post ID {post_id}: {api_error}")
                    time.sleep(5)
                    continue

            except Exception as row_error:
                print(f"Error processing row: {row_error}")
                continue

    except sqlite3.Error as db_error:
        print(f"Database error during query: {db_error}")
    finally:
        conn.close()
        print("Database connection closed")


def read_update_database_main_one_used_for_all():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:                                                                  #7745 # 11434 32860 # 35864 72300 77736 92389 108411 113846 127917 162008 209477 225194
        cursor.execute(f"SELECT id, text FROM {source_db_table} WHERE ID > 226041  ORDER BY id ASC")
        rows = cursor.fetchall()

        for row in rows:
            try:
                post_id = row[0]
                post_text = row[1]
                
                if not post_text:
                    print(f"Skipping empty post ID {post_id}")
                    continue

                print(f'\n#### Processing post ID {post_id}: {post_text.strip()}')
                
                try:
                    sentence_classifications = call_google_gemini(post_text.strip())
                    if not sentence_classifications:
                        print(f"No classifications returned for post ID {post_id}")
                        print("Response:" + str(sentence_classifications))
                        # we insert the resulst into the database
                        cursor.execute(
                                f"INSERT INTO {classified_table} (post_id, sentence, first_label, keywords_phrases_method_action, keywords_phrases_ideation, mid_label, next_label, second_label, full_response) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (post_id, post_text.strip(), "Error", "", "", "", "", "", sentence_classifications)
                            )
                        conn.commit()
                        print(f"Inserted classified sentence for post ID {post_id}")

                        continue
                    
                    for sentence_data in sentence_classifications:
                        try:
                            sentence = sentence_data.get("sentence", "")
                            if not sentence:
                                continue

                            first_label = sentence_data.get("first_label", "")
                            # json.dumps(sentence_data.get("keywords_phrases_method_action", []) or [])
                            keywords_phrases_method_action = json.dumps(sentence_data.get("keywords_phrases_method_action", []) or [])
                            keywords_phrases_ideation = json.dumps(sentence_data.get("keywords_phrases_ideation", []) or [])
                            mid_label = json.dumps(sentence_data.get("mid_label", []) or [])
                            next_label = json.dumps(sentence_data.get("next_label", []) or [])
                            second_label = sentence_data.get("second_label", "")
                            full_response = json.dumps(sentence_data or [])

                            cursor.execute(
                                f"INSERT INTO {classified_table} (post_id, sentence, first_label, keywords_phrases_method_action, keywords_phrases_ideation, mid_label, next_label, second_label, full_response) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (post_id, sentence, first_label, keywords_phrases_method_action, keywords_phrases_ideation, mid_label, next_label, second_label, full_response)
                            )
                            conn.commit()
                            print(f"Inserted classified sentence for post ID {post_id}")
                        except Exception as e:
                            print(f"Error inserting into database for post ID {post_id}: {e}")
                            conn.rollback()
                            continue


                except Exception as api_error:
                    print(f"Error calling Google Gemini API for post ID {post_id}: {api_error}")
                    time.sleep(5)
                    continue

            except Exception as row_error:
                print(f"Error processing row: {row_error}")
                continue

    except sqlite3.Error as db_error:
        print(f"Database error during initial query: {db_error}")
    finally:
        conn.close()
        print("Database connection closed")

# Function to create the table if it doesn’t already exist
def create_classified_sentences_table(db_path, source_table, classified_table):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {classified_table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id INTEGER NOT NULL,
        sentence TEXT NOT NULL,
        first_label TEXT,
        keywords_phrases_method_action JSON,  -- Store as JSON array
        keywords_phrases_ideation JSON,  -- Store as JSON array
        mid_label JSON,  -- Store as JSON array
        next_label JSON,  -- Store as JSON array
        second_label TEXT,
        full_response JSON,
        FOREIGN KEY (post_id) REFERENCES {source_table}(id) ON DELETE CASCADE
    );
    """

    try:
        cursor.execute(create_table_query)
        conn.commit()
        print(f"Table '{classified_table}' created successfully.")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

# Main function
if __name__ == "__main__":
    create_classified_sentences_table(db_path, source_db_table, classified_table)
    read_update_database()
