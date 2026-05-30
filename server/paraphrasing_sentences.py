from transformers import *
import sqlite3

# 15 Oct 2024 ....Main paraphrasing script used

#Pegasus Transformer
#In this section, we'll use the Pegasus transformer architecture model that was fine-tuned for paraphrasing instead of summarization. 
# # To instantiate the model, we need to use PegasusForConditionalGeneration as it's a form of text generation:

model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")

#Next, let's make a general function that takes a model, its tokenizer, the target sentence and returns the paraphrased text:

def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5, num_beams=5):
  # tokenize the text to be form of a list of token IDs
  inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
  # generate the paraphrased sentences
  outputs = model.generate(
    **inputs,
    num_beams=num_beams,
    num_return_sequences=num_return_sequences,
  )
  # decode the generated sentences using the tokenizer to get them back to text
  return tokenizer.batch_decode(outputs, skip_special_tokens=True)
# We also add the possibility of generating multiple paraphrased sentences by passing num_return_sequences to the model.generate() method.

# We also set num_beams so we generate the paraphrasing using beam search. Setting it to 5 will allow the model to look ahead for five possible words to keep the most likely hypothesis at each time step and choose the one that has the overall highest probability.

# I highly suggest you check this blog post to learn more about the parameters of the model.generate() method.

# Let's use the function now:

# Connect to the SQLite database
conn = sqlite3.connect('../databases/11-Oct-24/all_datasets_labelled.db')
cursor = conn.cursor()

# Read data from the "original" table
cursor.execute("SELECT id, sentence, label FROM MH_forum_388_sentences where label != 'Not Suicide post' ")
rows = cursor.fetchall()

# Insert paraphrased sentences into the "passphrases" table
for row in rows:
    original_id = row[0]
    sentence = row[1]
    label = row[2]

    #sentence = "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences."
    paraphrases = get_paraphrased_sentences(model, tokenizer, sentence, num_beams=10, num_return_sequences=10)
    print(paraphrases)

    if paraphrases is not None:
        for para_phrase_tuple in paraphrases:
            # Extract only the paraphrased sentence from the tuple (ignore score)
            # Retrieve the first field
            #first_field = para_phrase_tuple[0]
            print("Input_sentence: ", sentence)
            print("\tPassphrases:  ", para_phrase_tuple)
            #para_phrase = para_phrase_tuple[0]
            cursor.execute("INSERT INTO paraphrases2 (id, sentence, paraphrases, label) VALUES (?, ?, ?, ?)", (original_id, sentence, para_phrase_tuple, label))
            conn.commit()
# Commit the transaction and close the connection
conn.commit()
conn.close()
