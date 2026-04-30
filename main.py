import spacy
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

resumes_folder = "resumes"

def preprocess(text):
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            tokens.append(token.lemma_)
    return " ".join(tokens)

# Load and clean job description
with open("job.txt", encoding="utf-8") as f:
    job_description = f.read()

job_clean = preprocess(job_description)

scores = []

for filename in os.listdir(resumes_folder):
    file_path = os.path.join(resumes_folder, filename)

    with open(file_path, encoding="utf-8") as f:
        resume_text = f.read()

    resume_clean = preprocess(resume_text)

    # Vectorize both texts together
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_clean, job_clean])

    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    score = similarity[0][0] * 100

    scores.append((filename, score))

# Sort results
scores.sort(key=lambda x: x[1], reverse=True)

print("Resume Rankings:")
for rank, (filename, score) in enumerate(scores, start=1):
    print(f"Rank {rank}: {filename} - Match Score: {score:.2f}%")