from datasets import load_dataset
import re
import pandas as pd

# Read the formatted text file
with open("MLqa.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Regex pattern to capture questions and answers
qa_pairs = re.findall(
    r"(\d+)\)\s*(.*?)\s*\n(.*?)(?=\n\d+\)|\Z)",  # Captures (ID, Question, Answer)
    text,
    re.DOTALL
)

# Cleaning extracted QA pairs
cleaned_qa_pairs = []
for id_, question, answer in qa_pairs:
    question = question.strip()

    # Remove explicit '[Answer]' markers
    answer = re.sub(r"\[Answer\]", "", answer).strip()

    cleaned_qa_pairs.append((int(id_), question, answer))

# Convert to DataFrame
df = pd.DataFrame(cleaned_qa_pairs, columns=["ID", "Question", "Answer"])

# Save to CSV
df.to_csv("ml_questions.csv", index=False)

# Display DataFrame
print(df)

with open("NLPqa.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Regex pattern to capture questions and answers
qa_pairs = re.findall(
    r"(\d+)\.\s*(.*?)\s*\[answer\]\s*(.*?)(?=\n\d+\.|\Z)",  # Captures (ID, Question, Answer)
    text,
    re.DOTALL
)

# Cleaning extracted QA pairs
cleaned_qa_pairs = []
for id_, question, answer in qa_pairs:
    question = question.strip()
    answer = answer.strip()

    cleaned_qa_pairs.append((int(id_), question, answer))

# Convert to DataFrame
df_ = pd.DataFrame(cleaned_qa_pairs, columns=["ID", "Question", "Answer"])

# Save to CSV
df_.to_csv("nlp_questions.csv", index=False)

# Display DataFrame
df_.head()

final_df = pd.concat([df, df_], ignore_index=True)

# Save to CSV
final_df.to_csv("combined_questions.csv", index=False)

# Display final DataFrame
final_df

ML_ds = load_dataset("mjphayes/machine_learning_questions", split = "train")

df = pd.DataFrame(ML_ds)

df = df.drop("__index_level_0__", axis=1)

extra_qa_df = pd.read_csv("combined_questions.csv")
extra_qa_df = extra_qa_df.drop("ID", axis = 1)
extra_qa_df = extra_qa_df.rename(columns={"Question": "question", "Answer": "answer"})
Final_df = pd.concat([df, extra_qa_df], ignore_index=True)

Final_df.to_csv("final_df.csv", index=False)