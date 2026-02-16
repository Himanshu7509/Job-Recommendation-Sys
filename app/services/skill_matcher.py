import spacy
from spacy.matcher import PhraseMatcher
from app.core.config import DATA_DIR

nlp = spacy.load("en_core_web_sm")

def load_skill_db():
    with open(f"{DATA_DIR}/skills.txt", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f)

SKILL_DB = load_skill_db()

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(skill) for skill in SKILL_DB]
matcher.add("SKILLS", patterns)