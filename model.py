import re
import json
import numpy as np
import pandas as pd
import joblib
import torch
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
DOMAIN_SKILLS = {
    "data science": {
        "data", "dataset", "analysis", "analytics", "statistics", "visualization", 
        "pandas", "numpy", "sql", "eda", "insights", "business insights", "hypothesis",
        "data cleaning", "feature engineering", "reporting", "dashboard", "tableau", 
        "power bi", "data wrangling", "time series", "forecasting", "data pipeline", 
        "data mining"
    },
    "machine learning": {
        "model", "training", "prediction", "classifier", "accuracy", "precision", 
        "recall", "neural", "deep learning", "regression", "supervised", "unsupervised", 
        "reinforcement", "hyperparameter", "optimization", "tensorflow", "pytorch", 
        "scikit-learn", "nlp", "computer vision", "recommendation", "clustering", 
        "anomaly detection", "ensemble", "gradient boosting", "transformer"
    },
    "web development": {
        "frontend", "backend", "api", "rest", "html", "css", "javascript", "responsive",
        "web app", "server", "node.js", "express", "django", "flask", "angular", "vue", 
        "typescript", "authentication", "authorization", "database", "mongodb", "graphql",
        "deployment", "ci/cd", "microservices", "seo", "websocket", "react"
    },
    "cybersecurity": {
        "security", "vulnerability", "attack", "intrusion", "network", "firewall", 
        "malware", "pentest", "risk", "compliance", "ethical hacking", "forensics", 
        "incident response", "encryption", "access control", "zero trust", 
        "cloud security", "ids", "ips", "threat detection", "phishing", "audit", "policy"
    },
    "general": {
        "programming", "software", "development", "debugging", "testing", 
        "documentation", "version control", "collaboration", "agile", "devops", 
        "problem solving", "system design", "architecture"
    }
}
VALID_INTENTS = {
    "data science", "machine learning", " web development", "cybersecurity"
}

class IntentClassifier:
    def __init__(self, model_name='all-MiniLM-L6-v2', model_path=None):
        self.encoder = SentenceTransformer(model_name)
        self.encoder.eval()
        if model_path:
            data = joblib.load(model_path)
            self.classifier = data['classifier']
            self.label_encoder = data['label_encoder']
            self.scaler = data['scaler']
        else:
            self.classifier = None
            self.label_encoder = None
            self.scaler = StandardScaler()
    
    def createEmbeddings(self, texts: List[str]) -> np.ndarray:
        for param in self.encoder.parameters():
            param.requires_grad = False
        with torch.no_grad():
            embeddings = self.encoder.encode(texts, convert_to_tensor=False, show_progress_bar=False)
        return embeddings
    
    def train(self, X_train: List[str], y_train: List[str], x_val: List[str] = None, y_val: List[str] = None):
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_train)
        print("creating sentence embeddings")
        X_train_emb = self.createEmbeddings(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train_emb)
        print("Training classifier")
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(128,),  
            activation='relu',
            solver='adam',
            max_iter=300,  
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        self.classifier.fit(X_train_scaled, y_encoded)
        
        if x_val and y_val:
            x_val_emb = self.createEmbeddings(x_val)
            x_val_scaled = self.scaler.transform(x_val_emb)
            y_val_encoded = self.label_encoder.transform(y_val)
            y_pred = self.classifier.predict(x_val_scaled)
            accuracy = accuracy_score(y_val_encoded, y_pred)
            print(f"Validation Accuracy: {accuracy:.4f}")
            print("\nClassification report:")
            print(classification_report(y_val_encoded, y_pred, target_names=self.label_encoder.classes_))
        return self
    
    def predict(self, texts: List[str]) -> List[str]:
        embeddings = self.createEmbeddings(texts)
        scaled = self.scaler.transform(embeddings)
        predictions = self.classifier.predict(scaled)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        embedding = self.createEmbeddings([text])
        scaled = self.scaler.transform(embedding)
        probs = self.classifier.predict_proba(scaled)[0]
        return {cls: prob for cls, prob in zip(self.label_encoder.classes_, probs)}
    
    def save(self, path: str):
        joblib.dump({
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler
        }, path)
        print(f"model saved to {path}")
    
    def load(self, path: str):
        data = joblib.load(path)
        self.classifier = data['classifier']
        self.label_encoder = data['label_encoder']
        self.scaler = data['scaler']
        return self

class ModeClassifier:
    def __init__(self):
        self.keyword_rules = {
            'remote': ['remote', 'work from home', 'wfh', 'online', 'virtual', 
                      'distributed', 'telecommute', 'work from anywhere'],
            'onsite': ['onsite', 'office', 'in-person', 'on-site', 'in office', 
                      'physical', 'location', 'based in'],
            'hybrid': ['hybrid', 'flexible', 'partial remote', 'partially remote', 
                      'part remote', 'partly onsite', 'partly remote', 
                      'mix of remote and site', 'some remote']
        }
    
    def extractFeatures(self, text: str) -> np.ndarray:
        text_lower = text.lower()
        features = []
        for mode, keywords in self.keyword_rules.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            features.append(count)
        features.append(len(text.split()))
        features.append(1 if 'remote' in text_lower else 0)
        features.append(1 if 'hybrid' in text_lower else 0)
        return np.array(features).reshape(1, -1)
    
    def predict(self, text: str) -> str:
        features = self.extractFeatures(text)
        remote_score = features[0][0]
        onsite_score = features[0][1]
        hybrid_score = features[0][2]
        
        if hybrid_score > max(remote_score, onsite_score):
            return 'hybrid'
        elif remote_score > onsite_score:
            return 'remote'
        elif onsite_score > 0:
            return 'onsite'
        else:
            return 'hybrid'
        
class SkillExtractor:
    def __init__(self, skill_dict_path='skill_dictionary.json'):
        try:
            with open(skill_dict_path, 'r') as f:
                self.skill_list = json.load(f)
        except FileNotFoundError:
            self.skill_list = self.getDefaultSkills()

        self.skill_variations = self.createSkillVariations()

        self.aliases = {
            'ml': 'machine learning',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            'ai': 'artificial intelligence',
            'js': 'javascript',
            'reactjs': 'react',
            'nodejs': 'node.js',
            'python3': 'python',
            'tensorflow2': 'tensorflow',
            'tf': 'tensorflow',
            'pytorch': 'pytorch'
        }

        self.WEAK_SKILLS = {"ai", "ml", "machine learning", "artificial intelligence"}

        #Domain-to-skill inference
        self.domain_skill_inference = {
            'data science': ['python', 'pandas', 'sql', 'data analysis', 'visualization'],
            'machine learning': ['python', 'tensorflow', 'pytorch', 'scikit-learn', 'deep learning'],
            'web development': ['javascript', 'react', 'html', 'css', 'node.js', 'api'],
            'cybersecurity': ['linux', 'networking', 'security', 'penetration testing']
        }

        self.action_to_skill = {
            'build': ['development', 'programming'],
            'design': ['ui/ux', 'frontend', 'design'],
            'analyze': ['data analysis', 'analytics'],
            'train': ['machine learning', 'model training'],
            'deploy': ['devops', 'deployment'],
            'test': ['testing', 'qa'],
            'secure': ['security', 'cybersecurity'],
            'predict': ['machine learning', 'statistics'],
            'develop': ['development', 'programming'],
            'implement': ['programming', 'development'],
            'create': ['development', 'programming'],
            'optimize': ['optimization', 'performance'],
            'manage': ['management', 'project management'],
            'research': ['research', 'analysis']
        }
        self.skill_embeddings = {}
        self.skill_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.SKILL_MATCH_THRESHOLD = 0.55
        self.initializeSkillEmbeddings()

    def getDefaultSkills(self):
        return [
            'python', 'javascript', 'react', 'node.js', 'html', 'css',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch',
            'sql', 'mongodb', 'api', 'rest', 'pandas', 'numpy',
            'data analysis', 'statistics', 'visualization',
            'security', 'networking', 'linux', 'testing', 'deployment'
        ]

    def createSkillVariations(self) -> Dict[str, List[str]]:
        variations = {}
        for skill in self.skill_list:
            base = skill.lower().strip()
            if not base:
                continue
            skill_vars = [
                base,
                base.replace(' ', '-'),
                base.replace(' ', '_'),
                base.replace(' ', ''),
                ''.join(word[0] for word in base.split()) if ' ' in base else base
            ]
            variations[skill] = list(set(skill_vars))
        return variations

    def normalizeText(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[.,!?;:()\[\]{}]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def initializeSkillEmbeddings(self):
        print("Initializing skill embeddings...")

        skill_anchors = {}
        for skill in self.skill_list:
            s = skill.lower()
            if s in ['regression', 'forecasting', 'prediction']:
                anchor = "linear regression forecasting prediction statistical modeling"
            elif s in ['clustering', 'segmentation']:
                anchor = "kmeans clustering segmentation unsupervised learning"
            elif s in ['nlp', 'natural language processing']:
                anchor = "text nlp sentiment bert language processing"
            elif s in ['docker', 'kubernetes', 'deployment']:
                anchor = "docker kubernetes container deployment devops"
            elif s in ['react', 'frontend']:
                anchor = "react frontend javascript ui components"
            else:
                anchor = skill

            skill_anchors[skill] = anchor

        texts = list(skill_anchors.values())
        embeddings = self.skill_encoder.encode(texts, convert_to_numpy=True)

        for skill, emb in zip(skill_anchors.keys(), embeddings):
            self.skill_embeddings[skill] = emb

    def extract(self, text: str, return_scores: bool = False) -> List[str]:
        normalized_txt = self.normalizeText(text)
        words = normalized_txt.split()
        found_skills = {}

        for skill, variations in self.skill_variations.items():
            max_score = 0
            for variation in variations:
                if variation in words:
                    score = 1.0
                elif re.search(r'\b' + re.escape(variation) + r'\b', normalized_txt):
                    score = 0.9
                elif variation in normalized_txt:
                    score = 0.7
                else:
                    continue
                max_score = max(max_score, score)

            if max_score > 0.6:
                found_skills[skill] = max_score

        for alias, full in self.aliases.items():
            if alias in normalized_txt and full in self.skill_variations:
                found_skills[full] = max(found_skills.get(full, 0), 0.8)

        sorted_skills = sorted(found_skills.items(), key=lambda x: x[1], reverse=True)
        extracted = [s for s, _ in sorted_skills]

        if len(extracted) == 1 and extracted[0].lower() in self.WEAK_SKILLS:
            extracted = []

        return extracted if not return_scores else sorted_skills

    def extractWithEmbeddings(self, text: str, threshold: float = 0.55) -> List[str]:
        query_emb = self.skill_encoder.encode([text], convert_to_numpy=True)[0]

        matches = []
        for skill, emb in self.skill_embeddings.items():
            sim = np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8
            )
            if sim > threshold:
                matches.append((skill, sim))

        matches.sort(key=lambda x: x[1], reverse=True)

        filtered = [
            skill for skill, score in matches
            if skill.lower() not in self.WEAK_SKILLS or len(matches) > 2
        ]

        return filtered[:10]

    def extractWithContext(self, text: str, intent: str = None) -> List[str]:
        regex_skills = self.extract(text, return_scores=False)
        embed_skills = self.extractWithEmbeddings(text, self.SKILL_MATCH_THRESHOLD)
        all_skills = set(regex_skills)
        all_skills.update(embed_skills)

        #action verb inference
        normalized = self.normalizeText(text)
        for word in normalized.split():
            if word in self.action_to_skill:
                all_skills.update(self.action_to_skill[word])

        #intent-based inference
        if intent and intent != "general" and len(all_skills) < 3:
            if intent in self.domain_skill_inference:
                all_skills.update(self.domain_skill_inference[intent][:2])

        #filter valid dictionary skills
        skill_list_lower = {s.lower() for s in self.skill_list}
        valid_skills = [s for s in all_skills if s.lower() in skill_list_lower]

        return valid_skills[:5]

class InternshipRecommender:
    def __init__(self, internship_df, encoder_model='all-MiniLM-L6-v2'):
        self.df = internship_df.copy()
        self.GENERIC_SKILLS = {"web", "js", "javascript", "html", "css", "python"}
        self.DOMAIN_SKILL_WEIGHT = 1.5
        self.GENERIC_SKILL_WEIGHT = 0.3
        
        if 'domain' not in self.df.columns:
            self.df['domain'] = self.infer_domain(self.df)
        
        self.encoder = SentenceTransformer(encoder_model)
        self.skill_extractor = SkillExtractor()
        print("Computing internship embeddings...")
        self.precomputeEmbeddings()
    
    def infer_domain(self, df):
        def categorize(title, skills):
            text = (str(title) + ' ' + str(skills)).lower()
            if any(k in text for k in ['neural', 'deep learning', 'model', 'training', 'ml']):
                return 'machine learning'
            elif any(k in text for k in ['data', 'analytics', 'statistics', 'dashboard']):
                return 'data science'
            elif any(k in text for k in ['web', 'frontend', 'react', 'api']):
                return 'web development'
            elif any(k in text for k in ['cyber', 'security', 'linux', 'pentest']):
                return 'cybersecurity'
            else:
                return 'general'
        
        return df.apply(lambda row: categorize(row['title'], row['skills']), axis=1)
    
    def precomputeEmbeddings(self):
        if 'description' not in self.df.columns:
            self.df['description'] = ''
        
        domain_anchors = {
            'data science': ' involving data analysis, machine learning, statistics, python',
            'machine learning': ' involving model training, neural networks, deep learning, hyperparameters',
            'web development': ' involving frontend backend development, APIs, JavaScript, React',
            'cybersecurity': ' involving network security, penetration testing, Linux, encryption'
        }
        
        self.df['enriched_title'] = self.df.apply(
            lambda row: str(row['title']) + domain_anchors.get(row['domain'], ''), 
            axis=1
        )
        self.df['combined_text'] = (
            self.df['enriched_title'].fillna('') + ' ' +
            self.df['skills'].fillna('') + ' ' +
            self.df['description'].fillna('')
        )
        
        self.df['text_embedding'] = list(
            self.encoder.encode(
                self.df['combined_text'].tolist(),
                show_progress_bar=False,
                convert_to_numpy=True
            )
        )
        
        all_skills = set()
        for skill_list in self.df['skills'].dropna():
            for skill in str(skill_list).split(','):
                all_skills.add(skill.strip().lower())
        self.all_skills = sorted(list(all_skills))
        
        self.df['skill_vector'] = self.df['skills'].apply(
            lambda x: self.createSkillVector(str(x))
        )
        
        self.text_embeddings = np.stack(self.df['text_embedding'].values)
        self.skill_vectors = np.stack(self.df['skill_vector'].values)
    
    def createSkillVector(self, skills_text: str) -> np.ndarray:
        skills = str(skills_text).split(',')
        skills = [s.strip().lower() for s in skills if s.strip()]
        vector = np.zeros(len(self.all_skills))
        for i, skill in enumerate(self.all_skills):
            if any(skill in s or s in skill for s in skills):
                vector[i] = 1
        return vector
    
    def getUserRepresentation(self, query: str, user_skills: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        text_embedding = self.encoder.encode([query], convert_to_numpy=True)[0]
        if user_skills:
            skill_vector = self.createSkillVector(','.join(user_skills))
        else:
            skill_vector = np.zeros(self.skill_vectors.shape[1])
        return text_embedding, skill_vector
    
    def cosineSimilarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    

    def recommend(self, query: str, predicted_intent: str, intent_conf: float,
                  top_k: int = 5, alpha: float = 0.7, beta: float = 0.3) -> List[Dict]:

        #Context-aware skill extraction
        user_skills = self.skill_extractor.extractWithContext(query, predicted_intent)
        user_text_emb, user_skill_vec = self.getUserRepresentation(query, user_skills)

        #STRICT DOMAIN FILTERING (MANDATORY)
        if predicted_intent != "general":
            # Primary: exact domain match
            domain_filtered = self.df[self.df['domain'] == predicted_intent].copy()

            # Secondary: related domains
            if len(domain_filtered) < top_k:
                related_domains = self.getRelatedDomains(predicted_intent)
                for rd in related_domains:
                    if len(domain_filtered) < top_k:
                        related_df = self.df[self.df['domain'] == rd]
                        domain_filtered = pd.concat(
                            [domain_filtered, related_df]
                        ).drop_duplicates()

            #Last fallback general
            if len(domain_filtered) < 3:
                general_df = self.df[self.df['domain'] == 'general']
                domain_filtered = pd.concat(
                    [domain_filtered, general_df]
                ).drop_duplicates()
        else:
            # General intent  prioritize general domain
            domain_filtered = self.df[self.df['domain'] == 'general'].copy()
            if len(domain_filtered) < top_k:
                domain_filtered = self.df

        #Absolute fallback
        if len(domain_filtered) == 0:
            domain_filtered = self.df

        similarities = []

        for idx, row in domain_filtered.iterrows():
            global_idx = self.df.index.get_loc(idx)

            text_sim = self.cosineSimilarity(
                user_text_emb, self.text_embeddings[global_idx]
            )

            
            skill_sim = 0
            if user_skill_vec.sum() > 0:
                weighted_intersection = 0
                weighted_union = 0

                for i, val in enumerate(user_skill_vec):
                    if val == 1 or self.skill_vectors[global_idx][i] == 1:
                        skill_name = self.all_skills[i]
                        weight = (
                            self.GENERIC_SKILL_WEIGHT
                            if skill_name in self.GENERIC_SKILLS
                            else self.DOMAIN_SKILL_WEIGHT
                        )
                        weighted_union += weight
                        if val == 1 and self.skill_vectors[global_idx][i] == 1:
                            weighted_intersection += weight

                skill_sim = weighted_intersection / (weighted_union + 1e-8)

            combined_sim = alpha * text_sim + beta * skill_sim

            #DOMAIN MATCH BOOST
            if row['domain'] == predicted_intent:
                combined_sim *= 1.15

            #CROSS-DOMAIN PENALTY
            elif predicted_intent != "general":
                combined_sim *= 0.65

            similarities.append((idx, combined_sim, text_sim, skill_sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, combined_sim, text_sim, skill_sim in similarities[:top_k]:
            row = self.df.iloc[self.df.index.get_loc(idx)]

            explanation = self.generateExplanation(
                row, user_skills, combined_sim, text_sim, skill_sim, predicted_intent
            )

            results.append({
                'title': row['title'],
                'company': row.get('company', 'N/A'),
                'skills': row['skills'],
                'mode': row.get('mode', 'N/A'),
                'domain': row.get('domain', 'N/A'),
                'similarity_score': round(combined_sim, 3),
                'text_similarity': round(text_sim, 3),
                'skill_similarity': round(skill_sim, 3),
                'matched_skills': list(
                    set(user_skills) &
                    set(str(row['skills']).lower().split(','))
                ),
                'explanation': explanation
            })

        return results

    def getRelatedDomains(self, domain: str) -> List[str]:
        """Get related domains for fallback matching"""
        related_map = {
            'data science': ['machine learning', 'general'],
            'machine learning': ['data science', 'general'],
            'web development': ['general'],
            'cybersecurity': ['general']
        }
        return related_map.get(domain, ['general'])

    
    def generateExplanation(self, internship, user_skills, combined_sim, 
                          text_sim, skill_sim, intent) -> List[str]:
        explanations = []
        
        if text_sim > 0.7:
            explanations.append("High semantic match with internship description")
        elif text_sim > 0.5:
            explanations.append('Moderate semantic match')
        
        if skill_sim > 0.6:
            explanations.append('Strong skill overlap')
        elif skill_sim > 0.3:
            explanations.append("Partial skill match")
        
        matched = set(user_skills) & set(str(internship['skills']).lower().split(','))
        if matched:
            explanations.append(f"Exact skill matches: {', '.join(matched)}")
        
        if internship.get('domain') == intent:
            explanations.append(f"Aligned with {intent} domain")
        
        return explanations
    
class InternshipPipeline:
    def __init__(self, internships_file='D:\projects\intern_recommend_0.0\dataset_collection\scraped_internship.csv'):
        self.df = pd.read_csv(internships_file)
        self.DOMAIN_KEYWORDS = {
            "data science": ["data", "analysis", "statistics", "visualization", "sql", 
                            "pandas", "numpy", "eda", "insights", "hypothesis", 
                            "time series", "forecasting", "feature engineering"],
            "machine learning": ["model", "training", "regression", "classification", 
                                "neural", "clustering", "deep learning", "prediction",
                                "algorithm", "hyperparameter", "tensorflow", "pytorch",
                                "unsupervised", "supervised"],
            "web development": ["frontend", "backend", "react", "api", "ui", "css",
                               "javascript", "html", "server", "node", "express",
                               "django", "deployment", "rest", "graphql"],
            "cybersecurity": ["security", "attack", "vulnerability", "encryption", 
                             "siem", "pentest", "firewall", "malware", "intrusion",
                             "network", "threat", "forensics", "incident response"]
        }
        print("Initializing pipeline components...")
        
        self.intent_classifier = IntentClassifier()
        self.intent_classifier.load('intent_classifier_V1.pkl')
        
        self.mode_classifier = ModeClassifier()
        self.skill_extractor = SkillExtractor()
        self.recommender = InternshipRecommender(self.df)
        
        # NEW: Domain anchors for semantic intent override
        self.domain_anchors = {
            'data science': 'data analysis statistics pandas visualization insights dashboard',
            'machine learning': 'model training neural network deep learning prediction algorithm',
            'web development': 'frontend backend api javascript react html css server',
            'cybersecurity': 'security vulnerability penetration testing firewall encryption network'
        }
        
        # Pre-compute anchor embeddings
        print("Computing domain anchor embeddings...")
        self.domain_embeddings = {
            domain: self.recommender.encoder.encode([text], convert_to_numpy=True)[0]
            for domain, text in self.domain_anchors.items()
        }
        self.MIN_EVIDENCE =0.40 #Confindence threshold
        print("Pipeline initialized successfully")
    
    def normalizeSemantic(self, sim: float) -> float:
        #Typical observed range: 0.20-0.55
        if sim < 0.2:
            return 0.0
        elif sim > 0.6:
            return 1.0
        else:
            return (sim - 0.2) / (0.6 - 0.2)

    def computeDomainEvidence(self, query: str, intent_probs: Dict[str, float], extracted_skills: List[str]) -> Dict[str, float]:
        """Domain-agnostic evidence fusion from multiple signals."""
        query_emb = self.recommender.encoder.encode([query], convert_to_numpy=True)[0]
        evidence = {}
        
        for domain in self.domain_embeddings:
            # Signal 1: Classifier confidence
            clf_score = intent_probs.get(domain, 0)
            
            # Signal 2: Semantic similarity to domain anchor (NORMALIZED)
            raw_sim = np.dot(query_emb, self.domain_embeddings[domain]) / (
                np.linalg.norm(query_emb) * np.linalg.norm(self.domain_embeddings[domain]) + 1e-8
            )
            sem_score = self.normalizeSemantic(raw_sim)
            
            # Signal 3: Skill coverage
            domain_skills = DOMAIN_SKILLS.get(domain, set())
            if domain_skills and extracted_skills:
                skill_overlap = len(
                    set(s.lower() for s in extracted_skills) & domain_skills
                ) / max(len(domain_skills), 1)
            else:
                skill_overlap = 0
            
            # Weighted evidence fusion
            evidence[domain] = (
                0.45 * clf_score +
                0.35 * sem_score +
                0.20 * skill_overlap
            )
            
        
        return evidence
    
    def generateSummary(self, results: Dict) -> str:
        intent = results['intent']['predicted']
        mode = results['mode']
        skills = results['skills']['extracted'][:5]
        num_recs = len(results['recommendations'])
        summary = (
            f"Based on your query, I've identified:\n"
            f"• Primary Domain: {intent}\n"
            f"• Preferred Mode: {mode}\n"
            f"• Key Skills: {', '.join(skills) if skills else 'None detected'}\n"
            f"• Found {num_recs} matching internships\n"
            f"• Top recommendation: {results['recommendations'][0]['title']} "
            f"(Score: {results['recommendations'][0]['similarity_score']})"
        )
        return summary
    
    
    def processQuery(self, query: str, top_k: int = 5) -> Dict:
        if isinstance(query, tuple):
            query = query[0]
        
        results = {
            'query': query,
            'timestamp': pd.Timestamp.now().isoformat(),
            'decision_reasoning': None
        }
        
        # Step 1: Get classifier probabilities (PRIMARY DECISION)
        intent_probs = self.intent_classifier.predict_proba(query)
        
        # Classifier decides intent, not evidence
        predicted_intent = max(intent_probs, key=intent_probs.get)
        base_confidence = intent_probs[predicted_intent]
        
        results['base_intent'] = {
            'predicted': predicted_intent,
            'confidence': round(base_confidence, 4),
            'all_probs': {k: round(v, 4) for k, v in intent_probs.items()}
        }
        
        # Step 2: Extract skills WITH INTENT (for domain-specific inference)
        extracted_skills = self.skill_extractor.extractWithContext(query, predicted_intent)
        
        results['skills'] = {
            'extracted': extracted_skills,
            'count': len(extracted_skills)
        }
        
        # Step 3: Compute evidence only for ADJUSTING confidence
        evidence = self.computeDomainEvidence(query, intent_probs, extracted_skills)
        
        #Adjust confidence based on evidence (only if domain matches)
        adjusted_confidence = base_confidence
        if predicted_intent != "general" and predicted_intent in evidence:
            # Blend: 70% classifier confidence, 30% evidence support
            adjusted_confidence = 0.7 * base_confidence + 0.3 * evidence[predicted_intent]
            results['decision_reasoning'] = f"confidence_adjusted_by_evidence"
        else:
            results['decision_reasoning'] = f"classifier_primary"
        
        # Step 4: Determine confidence level never force general
        if adjusted_confidence >= 0.65:
            confidence_level = "HIGH"
        elif adjusted_confidence >= 0.50:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        #never downgrade to general unless classifier says so
        # (The classifier already decided predicted_intent)
        
        results['intent'] = {
            'predicted': predicted_intent,  # From classifier
            'confidence': round(adjusted_confidence, 4),
            'confidence_level': confidence_level,
            'evidence_support': {k: round(v, 4) for k, v in evidence.items()} if evidence else {},
            'base_classifier_prob': round(base_confidence, 4)
        }
        
        # Step 5: Mode classification
        predicted_mode = self.mode_classifier.predict(query)
        results['mode'] = predicted_mode
        
        # Step 6: Recommendations with STRICT domain filtering
        recommendations = self.recommender.recommend(
            query, predicted_intent, adjusted_confidence, top_k=top_k
        )
        results['recommendations'] = recommendations
        
        # Step 7: Generate summary
        results['summary'] = self.generateSummary(results)
        
        return results


def evaluatePipeline(test_queries: List, pipeline: InternshipPipeline):
    results = []
    
    for test in test_queries:
        if isinstance(test, dict):
            query = test.get('query')
            expected_intent = test.get('expected_intent')
        else:
            query = test[0]
            expected_intent = test[1] if len(test) > 1 else None
        
        output = pipeline.processQuery(query)
        
        result = {
            'query': query,
            'predicted_intent': output['intent']['predicted'],
            'expected_intent': expected_intent,
            'intent_correct': expected_intent == output['intent']['predicted'] 
                            if expected_intent else None,
            'num_skills_extracted': len(output['skills']['extracted']),
            'top_recommendation_score': output['recommendations'][0]['similarity_score'] 
                                       if output['recommendations'] else 0,
            'output': output
        }
        results.append(result)
    
    #calculate hit rates
    valid_results = [r for r in results if r['expected_intent'] is not None]
    
    if valid_results:
        #hit@K metrics
        def top_k_hit(recs, expected_intent, k=3):
            if not recs or not expected_intent:
                return False
            
            expected_norm = expected_intent.lower()
            expected_words = expected_norm.split()
            
            for r in recs[:k]:
                title = str(r.get('title', '')).lower()
                skills = str(r.get('skills', '')).lower()
                domain = str(r.get('domain', '')).lower()
                
                #check for matches
                for word in expected_words:
                    if (word in title or word in skills or word in domain or
                        expected_norm in title or expected_norm in skills):
                        return True
            return False
        
        hit_1 = sum(top_k_hit(r['output']['recommendations'], r['expected_intent'], 1) 
                   for r in valid_results) / len(valid_results)
        hit_3 = sum(top_k_hit(r['output']['recommendations'], r['expected_intent'], 3) 
                   for r in valid_results) / len(valid_results)
        hit_5 = sum(top_k_hit(r['output']['recommendations'], r['expected_intent'], 5) 
                   for r in valid_results) / len(valid_results)
        
        print("\n" + "="*60)
        print("IMPROVED RECOMMENDATION SYSTEM EVALUATION")
        print("="*60)
        print(f"Total test queries: {len(valid_results)}")
        print(f"\nRecommendation Quality:")
        print(f"  Hit@1: {hit_1:.2%} ")
        print(f"  Hit@3: {hit_3:.2%} ")
        print(f"  Hit@5: {hit_5:.2%} ")
        
        #Intent accuracy
        intent_correct = [r['intent_correct'] for r in valid_results 
                         if r['intent_correct'] is not None]
        if intent_correct:
            intent_accuracy = sum(intent_correct) / len(intent_correct)
            print(f"\nIntent Classification:")
            print(f"  Pipeline Accuracy: {intent_accuracy:.2%}  ")
        
        #skill extraction success rate
        skill_success = sum(1 for r in valid_results 
                          if r['num_skills_extracted'] > 0) / len(valid_results)
        print(f"\nSkill Extraction:")
        print(f"Success Rate: {skill_success:.2%}")
        print(f"Avg Skills/Query: {np.mean([r['num_skills_extracted'] for r in valid_results]):.2f}")
        
        print("\n" + "="*60)
        print("SAMPLE RESULTS")
        print("="*60)
        
        #show improvements
        for i, r in enumerate(valid_results[:3], 1):
            print(f"\n[Example {i}]")
            print(f"Query: {r['query'][:70]}...")
            print(f"Expected: {r['expected_intent']} | "f"Predicted: {r['predicted_intent']} | "f"Correct: {'✓' if r['intent_correct'] else 'X'}")
            print(f"Skills Extracted: {r['num_skills_extracted']}")
            if r['output']['recommendations']:
                top = r['output']['recommendations'][0]
                print(f"Top Match: {top['title'][:50]} "
                      f"(Domain: {top.get('domain', 'N/A')}, "
                      f"Score: {top['similarity_score']:.3f})")
    
    return results

def trainIntentClassifier(intent_data_file=r'D:\projects\intern_recommend_0.0\dataset_collection\new_intent_dataset.csv', model_save_path='intent_classifier_V1.pkl'):
    """Train the intent classifier"""
    df = pd.read_csv(intent_data_file)
    from sklearn.model_selection import train_test_split
    
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    classifier = IntentClassifier()
    classifier.train(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        val_df['text'].tolist(),
        val_df['label'].tolist()
    )
    
    joblib.dump({
        'classifier': classifier.classifier,
        'scaler': classifier.scaler,
        'label_encoder': classifier.label_encoder
    }, model_save_path)
    
    print(f'Intent classifier saved to {model_save_path}')
    return classifier
