import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string, random

# Embedding model import
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")
    EMBEDDINGS_AVAILABLE = False

# Ensure required nltk resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# ---------------- Embedding Model (Load Once) ---------------- #
_embedder = None

def getEmbedder():
    """Lazy load the embedding model"""
    global _embedder
    if _embedder is None and EMBEDDINGS_AVAILABLE:
        print("Loading embedding model (one-time setup)...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

# ---------------- Skill Normalization (CRITICAL) ---------------- #
SKILL_MAP = {
    " ml ": " machine learning ",
    " ai ": " artificial intelligence ",
    " ds ": " data science ",
    " cv ": " computer vision ",
    " nlp ": " natural language processing ",
    " nn ": " neural network ",
    " dl ": " deep learning ",
    " rl ": " reinforcement learning ",
    " api ": " application programming interface ",
    " ui ": " user interface ",
    " ux ": " user experience ",
    " db ": " database ",
    " aws ": " amazon web services ",
    " gcp ": " google cloud platform ",
}

def normalizeSkills(text):
    """Expand common abbreviations before preprocessing"""
    text = " " + text.lower() + " "  # Add spaces for boundary matching
    for abbr, full in SKILL_MAP.items():
        text = text.replace(abbr, full)
    return text.strip()

# ---------------- Preprocessing ---------------- #
def preprocess(text):
    """Preprocess text: normalize skills, tokenize, lemmatize, filter"""
    # First normalize skills
    text = normalizeSkills(text)
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords, numbers, punctuation, and very short tokens
    filtered_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalnum()
        and word not in stop_words
        and not word.isdigit()
        and len(word) > 2
    ]
    return ' '.join(filtered_tokens)

# ---------------- Term Contribution Helper ---------------- #
def getTopContributingTerms(query_vec, doc_vec, feature_names, top_n=3):
    contributions = query_vec.toarray()[0] * doc_vec.toarray()[0]
    top_indices = np.argsort(contributions)[::-1]
    top_terms = [feature_names[i] for i in top_indices if contributions[i] > 0][:top_n]
    return top_terms

# ---------------- Embedding Cache ---------------- #
_internship_embeddings_cache = {}

def buildInternshipEmbeddings(internships, cache_key="default"):
    """Build and cache embeddings for internships using PREPROCESSED text"""
    global _internship_embeddings_cache
    
    embedder = getEmbedder()
    if embedder is None:
        return None
    
    # Check cache
    if cache_key in _internship_embeddings_cache:
        return _internship_embeddings_cache[cache_key]
    
    # Build embeddings from PREPROCESSED text (same as TF-IDF)
    texts = [
        preprocess(f'{i["title"]} {i["description"]} {i["required_skills"]}')
        for i in internships
    ]
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    
    # Cache it
    _internship_embeddings_cache[cache_key] = embeddings
    return embeddings

def clearEmbeddingCache():
    """Clear the embedding cache (useful if internships change)"""
    global _internship_embeddings_cache
    _internship_embeddings_cache = {}

# ---------------- Recommendation Engine (Hybrid) ---------------- #
def recommendInternship(student, internships, top_n=5, use_embeddings=True):
    if not internships:
        return []

    # Combine resume skills + interests
    query_raw = student.get("skills", "") + " " + student.get("interests", "")
    
    # Preprocess query (same for both TF-IDF and embeddings)
    query = preprocess(query_raw)

    # Preprocess internship docs (same for both TF-IDF and embeddings)
    docs = [
        preprocess(f'{i["title"]} {i["description"]} {i["required_skills"]}')
        for i in internships
    ]

    # Fallback: no useful resume text
    if not query.strip():
        scores = []
        for intern in internships:
            score = (
                (intern.get("popularity", 0) / 100) * 0.5 +
                (intern.get("rating", 0) / 5) * 0.3 +
                (intern.get("company_prestige", 0) / 10) * 0.2
            )
            scores.append((score, intern, "Recommended based on general popularity and ratings."))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_n]

    # ---------------- TF-IDF Similarity ---------------- #
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,  # CRITICAL: Don't drop rare terms (resumes are small docs)
    )
    tfidf_matrix = vectorizer.fit_transform(docs + [query])
    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]
    tfidf_similarities = cosine_similarity(query_vec, doc_vecs)[0]

    # ---------------- Semantic Embeddings ---------------- #
    semantic_similarities = None
    if use_embeddings and EMBEDDINGS_AVAILABLE:
        try:
            embedder = getEmbedder()
            if embedder is not None:
                # Build or retrieve cached internship embeddings (using preprocessed text)
                internship_embeddings = buildInternshipEmbeddings(internships)
                
                # Compute query embedding using PREPROCESSED text (not raw)
                query_embedding = embedder.encode(
                    query,  # Use preprocessed query, not query_raw
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                # Compute semantic similarities
                semantic_similarities = cosine_similarity(
                    [query_embedding],
                    internship_embeddings
                )[0]
        except Exception as e:
            print(f"Warning: Embedding computation failed: {e}")
            semantic_similarities = None

    # Normalize stipend
    all_stipends = [intern.get("stipend", 0) for intern in internships]
    max_stipend = max(all_stipends) if all_stipends else 1
    if max_stipend == 0:
        max_stipend = 1

    scores = []
    feature_names = vectorizer.get_feature_names_out()

    for i, intern in enumerate(internships):
        tfidf_sim = tfidf_similarities[i]
        semantic_sim = semantic_similarities[i] if semantic_similarities is not None else 0
        
        # ---------------- Hybrid Similarity Logic ---------------- #
        if semantic_similarities is not None:
            if tfidf_sim < 0.05:
                # Weak TF-IDF match → rely on semantics
                final_sim = semantic_sim
                method = "semantic"
            else:
                # Strong TF-IDF match → hybrid approach
                final_sim = (tfidf_sim * 0.6) + (semantic_sim * 0.4)
                method = "hybrid"
        else:
            # No embeddings available → use TF-IDF only
            final_sim = tfidf_sim
            method = "tfidf"

        # Weighted scoring system
        score = (
            final_sim * 0.55 +
            (intern.get("popularity", 0) / 100) * 0.15 +
            min(intern.get("stipend", 0) / max_stipend, 1) * 0.15 +
            (intern.get("rating", 0) / 5) * 0.10 +
            (intern.get("company_prestige", 0) / 10) * 0.05
        )

        # ---------------- Build Explanation ---------------- #
        explanation_lines = []
        
        if method == "hybrid":
            top_terms = getTopContributingTerms(query_vec, doc_vecs[i], feature_names)
            if top_terms:
                explanation_lines.append(f"• Hybrid Match: Keywords ({', '.join(top_terms)}) + Semantic meaning")
            else:
                explanation_lines.append("• Hybrid Match: Textual and semantic alignment")
            explanation_lines.append(f"• TF-IDF Similarity: {tfidf_sim:.3f}")
            explanation_lines.append(f"• Semantic Similarity: {semantic_sim:.3f}")
            
        elif method == "semantic":
            explanation_lines.append("• Semantic Match: Inferred from resume meaning (no exact keyword overlap)")
            explanation_lines.append(f"• Semantic Similarity: {semantic_sim:.3f}")
            
        else:  # tfidf only
            top_terms = getTopContributingTerms(query_vec, doc_vecs[i], feature_names)
            if tfidf_sim > 0.05 and top_terms:
                explanation_lines.append(f"• Keyword Match: {', '.join(top_terms)}")
            elif tfidf_sim > 0.05:
                explanation_lines.append("• General textual match with your resume")
            else:
                explanation_lines.append("• No strong skill match, ranked by popularity and ratings")
            explanation_lines.append(f"• TF-IDF Similarity: {tfidf_sim:.3f}")

        explanation_lines.append(f"• Popularity Score: {intern.get('popularity', 0)}/100")
        explanation_lines.append(f"• Prestige Score: {intern.get('company_prestige', 0)}/10")

        explanation = "\n".join(explanation_lines)
        scores.append((score, intern, explanation))

    # Sort by score, add slight randomness to break ties
    scores = sorted(scores, key=lambda x: (x[0], random.random()), reverse=True)
    return scores[:top_n]