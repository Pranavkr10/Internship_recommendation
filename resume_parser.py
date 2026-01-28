import nltk
import numpy as np
import re
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

#SKILL DATABASE Categorized skills for better matching
SKILL_DATABASE = {
    "programming": ["python", "java", "javascript", "c++", "c#", "go", "rust", 
                    "swift", "kotlin", "typescript", "ruby", "php", "scala", "r"],
    "data_science": ["pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", 
                     "keras", "spark", "hadoop", "tableau", "powerbi", "matplotlib",
                     "seaborn", "machine learning", "deep learning", "data analysis",
                     "data visualization", "statistics"],
    "web_dev": ["react", "angular", "vue", "django", "flask", "node.js", "nodejs",
                "express", "spring", "laravel", "fastapi", "html", "css", "bootstrap",
                "tailwind", "webpack", "redux", "next.js", "nextjs"],
    "cloud_devops": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", 
                     "jenkins", "ansible", "git", "ci/cd", "gitlab", "github actions",
                     "circleci", "nginx", "linux"],
    "databases": ["mysql", "postgresql", "mongodb", "redis", "cassandra", 
                  "elasticsearch", "oracle", "sqlite", "sql", "nosql", "dynamodb"],
    "mobile": ["android", "ios", "react native", "flutter", "xamarin", "swift", "kotlin"],
    "soft_skills": ["leadership", "communication", "teamwork", "problem-solving", 
                    "critical-thinking", "adaptability", "time management", "creativity"]
}

CRITICAL_SKILLS = ["python", "aws", "react", "sql", "docker", "kubernetes", "machine learning", "api", "git", "agile"]

def preprocess(text):
    #Input validation guard
    if not text or not isinstance(text, str):
        return ""
    
    # Strip and check for empty after stripping
    text = text.strip()
    if not text:
        return ""
    
    try:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        
        filtered_tokens = [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word.isalnum()
            and word not in stop_words
            and not word.isdigit()
            and len(word) > 2
        ]
        return ' '.join(filtered_tokens)
    except Exception as e:
        print(f"Error in preprocess: {e}")
        return ""

def extractStructuredResumeData(resume_text):
    if not resume_text or not resume_text.strip():
        return {
            "skills": [],
            "skills_detailed": [],
            "skill_categories": {},
            "experience": [],
            "education": [],
            "skill_gaps": {},
            "raw_text": ""
        }
    
    #Extract skills using pattern matching
    skills_found = []
    skill_categories = {}
    resume_lower = resume_text.lower()
    
    for category, skill_list in SKILL_DATABASE.items():
        category_skills = []
        for skill in skill_list:
            #Multiple matching patterns to catch variations
            patterns = [
                r'\b' + re.escape(skill) + r'\b',
                r'\b' + re.escape(skill.replace('-', ' ')) + r'\b',
                r'\b' + re.escape(skill.replace('.', '')) + r'\b'
            ]
            for pattern in patterns:
                if re.search(pattern, resume_lower, re.IGNORECASE):
                    skills_found.append({
                        "skill": skill,
                        "category": category,
                        "confidence": 0.9 if category != "soft_skills" else 0.7
                    })
                    category_skills.append(skill)
                    break
        
        if category_skills:
            skill_categories[category] = category_skills
    
  
    experience_patterns = [
        r'(\d+)\s*\+?\s*years?\s*(?:of)?\s*(?:experience|exp)',
        r'worked\s+(?:as|at)\s+([a-zA-Z\s]+)',
        r'(?:intern|internship|employee|engineer|developer|analyst)\s+at\s+([a-zA-Z\s]+)'
    ]
    
    experience = []
    for pattern in experience_patterns:
        matches = re.finditer(pattern, resume_text, re.IGNORECASE)
        for match in matches:
            experience.append(match.group().strip())
    
    
    education_keywords = ['bachelor', 'master', 'phd', 'bs', 'ms', 'btech', 
                          'mtech', 'university', 'college', 'degree']
    education = []
    sentences = resume_text.split('.')
    for sent in sentences:
        if any(keyword in sent.lower() for keyword in education_keywords):
            education.append(sent.strip())
    
    
    skill_gaps = analyzeSkillGaps(skills_found)
    
    return {
        "skills": list({s["skill"] for s in skills_found}),
        "skills_detailed": skills_found,
        "skill_categories": skill_categories,
        "experience": list(set(experience))[:5],
        "education": list(set(education))[:3],
        "skill_gaps": skill_gaps,
        "raw_text": resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text
    }

def analyzeSkillGaps(found_skills):
    found_skill_names = [s["skill"] for s in found_skills]
    missing = [skill for skill in CRITICAL_SKILLS if skill not in found_skill_names]
    
    coverage_score = len(set(found_skill_names) & set(CRITICAL_SKILLS)) / len(CRITICAL_SKILLS)
    
    return {
        "missing_critical": missing[:5],
        "coverage_score": round(coverage_score * 100, 1),
        "suggested_skills": missing[:3],
        "total_found": len(found_skill_names),
        "total_critical": len(CRITICAL_SKILLS)
    }

def scoreResumeAts(resume_text):
    if not resume_text or not resume_text.strip():
        return {
            "ats_score": 0,
            "grade": "F",
            "improvements": ["Please provide a resume for analysis"]
        }
    
    total_score = 100
    
    #Check for ATS red flags (penalties)
    red_flags = [
        (r'\bobjective\b', -10, "Remove outdated 'Objective' section"),
        (r'references available upon request', -5, "Remove 'References available' line"),
        (r'\bhobbies\b', -3, "Remove 'Hobbies' section unless relevant"),
        (r'http://|https://', -5, "Too many URLs can break ATS")
    ]
    
    improvements = []
    
    for pattern, penalty, message in red_flags:
        if re.search(pattern, resume_text, re.IGNORECASE):
            total_score += penalty
            improvements.append(message)
    
    #Check for green flags (bonuses)
    green_flags = [
        (r'\d+\s*%', 15, "Great! Includes quantifiable percentages"),
        (r'\$\d+[KkMm]?', 15, "Great! Shows monetary impact"),
        (r'increased|decreased|improved|reduced|optimized|enhanced', 10, "Good use of action verbs"),
    ]
    
    for pattern, bonus, message in green_flags:
        if re.search(pattern, resume_text, re.IGNORECASE):
            total_score += bonus
    
    #Check section completeness
    required_sections = ["experience", "education", "skills", "projects"]
    sections_found = 0
    for section in required_sections:
        if re.search(fr'\b{section}\b', resume_text, re.IGNORECASE):
            sections_found += 1
    
    if sections_found < 3:
        improvements.append(f"Add missing sections - found only {sections_found}/4 key sections")
        total_score -= (4 - sections_found) * 5
    
    total_score += (sections_found / len(required_sections)) * 10
    
    #Check word count
    word_count = len(resume_text.split())
    if word_count < 300:
        improvements.append("Resume too short - add more details (aim for 400-700 words)")
        total_score -= 15
    elif word_count > 900:
        improvements.append("Resume too long - keep it concise (aim for 400-700 words)")
        total_score -= 10
    
    #Check for action verbs
    action_verbs = ["achieved", "built", "created", "developed", "implemented", 
                    "improved", "increased", "led", "managed", "optimized"]
    action_verb_count = sum(1 for verb in action_verbs if verb in resume_text.lower())
    
    if action_verb_count < 3:
        improvements.append(f"Add more action verbs (found {action_verb_count}, aim for 5+)")
        total_score -= 5
    
    #Check for quantifiable achievements
    if not re.search(r'\d+\s*%|\$\d+|\d+\s*(?:x|times)', resume_text):
        improvements.append("Add quantifiable achievements (e.g., 'Improved by 30%', 'Reduced costs by $5K')")
        total_score -= 10
    
    #Ensure score is within bounds
    final_score = max(0, min(100, total_score))
    
    #Determine grade
    if final_score >= 85:
        grade = "A"
    elif final_score >= 70:
        grade = "B"
    elif final_score >= 50:
        grade = "C"
    elif final_score >= 30:
        grade = "D"
    else:
        grade = "F"
    
    return {
        "ats_score": round(final_score, 1),
        "grade": grade,
        "improvements": improvements[:5] if improvements else ["Resume looks good!"]
    }

def getTopContributingTerms(query_vec, doc_vec, feature_names, top_n=3):
    """Get terms that contribute most to similarity score"""
    contributions = query_vec.toarray()[0] * doc_vec.toarray()[0]
    top_indices = np.argsort(contributions)[::-1]
    top_terms = [feature_names[i] for i in top_indices if contributions[i] > 0][:top_n]
    return top_terms

def calculateSkillOverlap(resume_skills, internship_required_skills):
    resume_skills_set = set([s.lower() for s in resume_skills])
    required_skills = re.split(r'[,\|/]', internship_required_skills.lower())
    required_skills = [s.strip() for s in required_skills if s.strip()]
    
    overlap = resume_skills_set & set(required_skills)
    match_count = len(overlap)
    total_required = len(required_skills) if required_skills else 1
    
    match_percentage = (match_count / total_required) * 100
    missing = [skill for skill in required_skills if skill not in resume_skills_set]
    
    return {
        "match_count": match_count,
        "match_percentage": round(match_percentage, 1),
        "missing_skills": missing[:5],
        "matched_skills": list(overlap)[:5]
    }

def recommendInternship(student, internships, top_n=5):
    """
    FIX #1: Fixed list + string concatenation issue
    FIX #4: Added TF-IDF empty vocabulary protection
    FIX #5: Normalized return tuple to always have 5 values
    """
    if not internships:
        return []
    
    # Convert skills and interests lists to space-separated strings
    skills_text = " ".join(student.get("skills", []))
    interests_text = " ".join(student.get("interests", []))
    query_raw = f"{skills_text} {interests_text}".strip()
    
    # If raw_resume is available, use that instead
    if "raw_resume" in student and student["raw_resume"]:
        query_raw = student["raw_resume"]
    
    resume_data = extractStructuredResumeData(query_raw)
    ats_analysis = scoreResumeAts(query_raw)
    
    query = preprocess(query_raw)
    docs = [
        preprocess(f'{i["title"]} {i["description"]} {i["required_skills"]}')
        for i in internships
    ]
    
    # FIX #4: Filter out empty documents and check for empty query
    docs = [d for d in docs if d.strip()]
    
    # Fallback: no useful resume text or all docs are empty
    if not query.strip() or not docs:
        scores = []
        for intern in internships:
            score = (
                (intern.get("popularity", 0) / 100) * 0.5 +
                (intern.get("rating", 0) / 5) * 0.3 +
                (intern.get("company_prestige", 0) / 10) * 0.2
            )
            explanation = "No resume data found - ranked by popularity.\n"
            explanation += f"Popularity: {intern.get('popularity', 0)}/100\n"
            explanation += f"Rating: {intern.get('rating', 0)}/5\n"
            explanation += "Tip: Upload your resume for personalized recommendations!"
            
            # FIX #5: Return 5-tuple even in fallback mode
            scores.append((
                score,
                intern,
                explanation,
                {},  # Empty resume_data
                {"ats_score": 0, "grade": "N/A", "improvements": ["No resume provided"]}
            ))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_n]
    
    #TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(docs + [query])
    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]
    similarities = cosine_similarity(query_vec, doc_vecs)[0]
    
    #Normalize stipend
    all_stipends = [intern.get("stipend", 0) for intern in internships]
    max_stipend = max(all_stipends) if all_stipends else 1
    if max_stipend == 0:
        max_stipend = 1
    
    scores = []
    feature_names = vectorizer.get_feature_names_out()
    
    for i, intern in enumerate(internships):
        sim = similarities[i]
        
        #Calculate skill overlap
        skill_overlap_data = calculateSkillOverlap(
            resume_data["skills"], 
            intern.get("required_skills", "")
        )
        skill_fit = skill_overlap_data["match_percentage"] / 100
        
        #Weighted scoring system
        score = (
            sim * 0.45 +                                           
            skill_fit * 0.25 +                                     
            (intern.get("popularity", 0) / 100) * 0.12 +          
            min(intern.get("stipend", 0) / max_stipend, 1) * 0.10 + 
            (intern.get("rating", 0) / 5) * 0.05 +                
            (intern.get("company_prestige", 0) / 10) * 0.03       
        )
        explanation_parts = []
        
        #Header with match quality
        if skill_fit > 0.7:
            explanation_parts.append("EXCELLENT MATCH")
        elif skill_fit > 0.4:
            explanation_parts.append("GOOD MATCH")
        else:
            explanation_parts.append("POTENTIAL OPPORTUNITY")
        
        #Skill analysis
        explanation_parts.append(f"\nSkill Match: {skill_overlap_data['match_count']} skills ({skill_overlap_data['match_percentage']}%)")
        
        if skill_overlap_data["matched_skills"]:
            explanation_parts.append(f"   ✓ Matching: {', '.join(skill_overlap_data['matched_skills'][:3])}")
        
        if skill_overlap_data["missing_skills"] and len(skill_overlap_data["missing_skills"]) > 0:
            missing_display = ', '.join(skill_overlap_data["missing_skills"][:3])
            explanation_parts.append(f"   Missing: {missing_display}")
        
        #Keyword contribution
        top_terms = getTopContributingTerms(query_vec, doc_vecs[i], feature_names, 3)
        if top_terms:
            explanation_parts.append(f"\n Key Matches: {', '.join(top_terms)}")
        
        #Metrics
        explanation_parts.append(f"\nMetrics:")
        explanation_parts.append(f"   • Similarity Score: {sim:.2f}")
        explanation_parts.append(f"   • Popularity: {intern.get('popularity', 0)}/100")
        explanation_parts.append(f"   • Company Prestige: {intern.get('company_prestige', 0)}/10")
        
        #Resume quality feedback
        if ats_analysis["grade"] in ["C", "D", "F"]:
            explanation_parts.append(f"\nResume Quality: {ats_analysis['grade']} ({ats_analysis['ats_score']}/100)")
            explanation_parts.append(f"   Tip: {ats_analysis['improvements'][0] if ats_analysis['improvements'] else 'Improve your resume'}")
        
        explanation = "\n".join(explanation_parts)
        scores.append((score, intern, explanation, resume_data, ats_analysis))
    
    #Sort by score with slight randomness to break ties
    scores = sorted(scores, key=lambda x: (x[0], random.random()), reverse=True)
    return scores[:top_n]

def generateResumeSuggestions(resume_data, target_role="software engineer"):
    suggestions = {
        "add_sections": [],
        "improve_sections": [],
        "keyword_optimizations": [],
        "skill_development": []
    }
    
    #Role specific keyword suggestions
    role_keywords = {
        "software engineer": ["agile", "scrum", "rest api", "microservices", "cicd"],
        "data scientist": ["machine learning", "predictive modeling", "etl", "data visualization"],
        "web developer": ["responsive design", "rest api", "typescript", "testing"],
        "data analyst": ["sql", "excel", "tableau", "data visualization", "statistics"]
    }
    
    target_keywords = role_keywords.get(target_role.lower(), role_keywords["software engineer"])
    
    #Check for missing keywords
    resume_text_lower = resume_data.get("raw_text", "").lower()
    missing_keywords = [kw for kw in target_keywords if kw not in resume_text_lower]
    
    if missing_keywords:
        suggestions["keyword_optimizations"].append(
            f"Add {target_role} keywords: {', '.join(missing_keywords[:3])}"
        )
    
    #Suggest quantifiable achievements
    if not re.search(r'\d+%|\$\d+|\d+\s*(?:x|times)', resume_text_lower):
        suggestions["improve_sections"].append(
            "Add quantifiable achievements (e.g., 'Improved performance by 30%')"
        )
    
    #Check project section
    if not re.search(r'\bprojects?\b', resume_text_lower, re.IGNORECASE):
        suggestions["add_sections"].append(
            "Add a 'Projects' section with GitHub links"
        )
    
    #Skill gap recommendations
    if resume_data["skill_gaps"]["missing_critical"]:
        suggestions["skill_development"].append(
            f"Consider learning: {', '.join(resume_data['skill_gaps']['missing_critical'][:3])}"
        )
    
    return suggestions


def generateCoverLetterTemplate(resume_data, internship):
    skills = resume_data.get("skills", [])
    top_skills = ', '.join(skills[:3]) if skills else "relevant technical skills"
    
    template = f"""Dear Hiring Manager,

I am excited to apply for the {internship.get('title', 'internship position')} at {internship.get('company', 'your company')}. 
With my background in {top_skills}, I am confident in my ability to contribute to your team.

Key qualifications that align with your requirements:
1. Technical expertise in {skills[0] if skills else 'modern technologies'}
2. Demonstrated experience through academic and personal projects
3. Strong foundation in {', '.join(skills[1:3]) if len(skills) > 2 else 'computer science fundamentals'}

I am particularly drawn to this opportunity because of {internship.get('company', 'your company')}'s 
reputation for innovation and excellence in the field.

I would welcome the opportunity to discuss how my skills and enthusiasm can contribute to your team.

Sincerely,
[Your Name]
"""
    return template

def processResumeComplete(resume_text, internships_data, target_role="software engineer"):
    """
    Complete enhanced resume processing pipeline
    Returns all analysis, recommendations, and suggestions
    """
    parsed_resume = extractStructuredResumeData(resume_text)
    ats_score = scoreResumeAts(resume_text)
    #Create student profile
    student_profile = {
        "skills": parsed_resume["skills"],
        "interests": [],
        "raw_resume": resume_text
    }

    recommendations = recommendInternship(student_profile, internships_data, top_n=5)
    suggestions = generateResumeSuggestions(parsed_resume, target_role)
    #Generate cover letter for top match
    cover_letter = ""
    if recommendations and len(recommendations) > 0:
        top_internship = recommendations[0][1]
        cover_letter = generateCoverLetterTemplate(parsed_resume, top_internship)
    
    return {
        "parsed_resume": parsed_resume,
        "ats_analysis": ats_score,
        "recommendations": [(score, intern, explanation) for score, intern, explanation, _, _ in recommendations],
        "skill_gaps": parsed_resume["skill_gaps"],
        "improvement_suggestions": suggestions,
        "cover_letter_template": cover_letter
    }