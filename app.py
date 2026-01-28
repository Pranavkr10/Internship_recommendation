from flask import Flask, render_template, request, jsonify
from model import InternshipPipeline
from base import recommendInternship
from PyPDF2 import PdfReader
import traceback
import numpy as np
import mysql.connector
from mysql.connector import Error
from roadmap import ROADMAPS

try:
    from resume_parser import (
        processResumeComplete, 
        extractStructuredResumeData, 
        scoreResumeAts,
        recommendInternship as resumeRecommendInternship,
        generateResumeSuggestions,
        generateCoverLetterTemplate
    )
    HAS_RESUME_PARSER = True
    print("Resume parser loaded successfully")
except ImportError as e:
    print(f"Warning: Resume parser not available. Chatbot will still work. Error: {e}")
    HAS_RESUME_PARSER = False
    #Creating dummy functions to avoid errors
    def processResumeComplete(*args, **kwargs):
        return {"error": "Resume parser not installed"}
    def extractStructuredResumeData(*args, **kwargs):
        return {"error": "Resume parser not installed"}
    def scoreResumeAts(*args, **kwargs):
        return {"error": "Resume parser not installed"}
    def resumeRecommendInternship(*args, **kwargs):
        return []
    def generateResumeSuggestions(*args, **kwargs):
        return {}
    def generateCoverLetterTemplate(*args, **kwargs):
        return ""

app = Flask(__name__)

MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',      
    'password': 'qwerty',      
    'database': 'internship_db',
    'port': 3306
}

# Initialize the NLP pipeline (model.py)
pipeline = InternshipPipeline(
    internships_file=r"D:\projects\intern_recommend_0.0\dataset_collection\scraped_internship.csv"
)

def get_db_connection():
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def fetch_internships_from_db(limit=500):
    connection = get_db_connection()
    if not connection:
        print("Warning: Could not connect to database for internships")
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, title, company, skills, location, mode, source,
                   popularity, rating, company_prestige
            FROM internships
            ORDER BY popularity DESC
            LIMIT %s
        """, (limit,))

        rows = cursor.fetchall()
        cursor.close()
        connection.close()

        # Convert DB schema to recommender schema
        internships = []
        for r in rows:
            # FIX 3: Map company_prestige VARCHAR to numeric score
            prestige_raw = (r.get("company_prestige") or "").lower()
            prestige_map = {
                "high": 9,
                "medium": 6,
                "low": 3
            }
            prestige_score = prestige_map.get(prestige_raw, 5)  # default to 5
            
            internships.append({
                "id": r["id"],
                "title": r["title"],
                "company": r["company"],
                "required_skills": r["skills"] or "",  # KEY: Map skills -> required_skills
                "description": "",  # DB doesn't have this field
                "location": r.get("location", ""),
                "mode": r.get("mode", "onsite"),
                "source": r.get("source", ""),
                "popularity": r.get("popularity", 0) or 0,
                "rating": float(r["rating"]) if r["rating"] is not None else 0,
                "company_prestige": prestige_score  # Now properly mapped
            })

        print(f"✓ Loaded {len(internships)} internships from database")
        return internships

    except Error as e:
        print(f"Database error while fetching internships: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {traceback.format_exc()}")
        return []
    
# ==================== CHATBOT ROUTES (model.py) ====================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chatbot endpoint using model.py"""
    data = request.get_json()
    query = data.get("message", "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        result = pipeline.processQuery(query)

        #Convert numpy floats to Python floats
        confidence = result["intent"]["confidence"]
        if hasattr(confidence, 'item'):  
            confidence = float(confidence)

        top_recommendation = None
        if result["recommendations"]:
            rec = result["recommendations"][0]
            if isinstance(rec, (list, tuple)) and len(rec) > 0:
                score = rec[0]
                if hasattr(score, 'item'):  
                    score = float(score)
                    top_recommendation = (score,) + rec[1:]
                else:
                    top_recommendation = rec

        response = {
            "intent": result["intent"]["predicted"],
            "confidence": confidence,
            "skills": result["skills"]["extracted"],
            "top_recommendation": top_recommendation,
            "summary": result["summary"]
        }

        return jsonify(response)
    
    except Exception as e:
        print(f"Chat error: {traceback.format_exc()}")
        return jsonify({"error": f"Chat processing failed: {str(e)}"}), 500


# ==================== RESUME PARSER ROUTES ====================

@app.route("/api/resume/upload", methods=["POST"])
def upload_resume():
    """Upload and parse resume (PDF/TXT only)"""
    if not HAS_RESUME_PARSER:
        return jsonify({
            "error": "Resume parser not available. Please check if all dependencies are installed.",
            "hint": "Run: pip install nltk scikit-learn"
        }), 503
    
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded. Please select a PDF or TXT file."}), 400
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    filename = file.filename.lower()
    if not (filename.endswith('.pdf') or filename.endswith('.txt')):
        return jsonify({"error": "Only PDF and TXT files are supported"}), 400
    
    try:
        if filename.endswith('.pdf'):
            text = extractTextFromPdf(file)
        else: 
            text = extractTextFromTxt(file)
        
        if not text or len(text.strip()) < 50:
            return jsonify({
                "error": "File appears to be empty or too short (minimum 50 characters required)"
            }), 400
        parsed_resume = extractStructuredResumeData(text)
        ats_score = scoreResumeAts(text)
        
        return jsonify({
            "success": True,
            "file_name": file.filename,
            "text_length": len(text),
            "text_preview": text[:500] + "..." if len(text) > 500 else text,
            "parsed_resume": parsed_resume,
            "ats_score": ats_score
        })
        
    except Exception as e:
        print(f"Error parsing resume: {traceback.format_exc()}")
        return jsonify({
            "error": f"Failed to parse resume: {str(e)}",
            "hint": "Please ensure the file is readable and contains valid text"
        }), 500


@app.route("/api/resume/analyze", methods=["POST"])
def analyze_resume():
    """Complete resume analysis with recommendations (IMPROVEMENT 1: Now uses DB)"""
    if not HAS_RESUME_PARSER:
        return jsonify({
            "error": "Resume parser not available",
            "hint": "Please check if all dependencies are installed"
        }), 503
    
    data = request.get_json()
    resume_text = data.get("resume_text", "").strip()

    if not resume_text:
        return jsonify({"error": "Resume text is empty"}), 400
    
    if len(resume_text) < 50:
        return jsonify({"error": "Resume text too short (minimum 50 characters)"}), 400

    try:
        #Using DB internships for analysis
        db_internships = fetch_internships_from_db(limit=500)
        internships_source = db_internships if db_internships else pipeline.internships
        
        #Use processResumeComplete from resume_parser.py
        result = processResumeComplete(
            resume_text=resume_text,
            internships_data=internships_source,  
            target_role=data.get("target_role", "software engineer")
        )
        
        return jsonify({
            "success": True,
            "analysis": result,
            "data_source": "database" if db_internships else "csv_fallback"
        })
        
    except Exception as e:
        print(f"Error analyzing resume: {traceback.format_exc()}")
        return jsonify({
            "error": f"Analysis failed: {str(e)}",
            "hint": "Please check the resume text format"
        }), 500


@app.route("/api/resume/recommend", methods=["POST"])
def resume_recommend():
    """Get personalized recommendations based on resume (with DB integration)"""
    if not HAS_RESUME_PARSER:
        return jsonify({"error": "Resume parser not available"}), 503
    
    data = request.get_json()
    resume_text = data.get("resume_text", "").strip()
    top_n = int(data.get("top_n", 5)) 
    use_db = data.get("use_db", True)  
    
    if not resume_text:
        return jsonify({"error": "Resume text required"}), 400
    
    try:
        parsed_resume = extractStructuredResumeData(resume_text)
        ats_analysis = scoreResumeAts(resume_text)
        student_profile = {
            "skills": parsed_resume["skills"],
            "interests": [],
            "raw_resume": resume_text
        }
        if use_db:
            db_internships = fetch_internships_from_db(limit=500)
            internships_source = db_internships if db_internships else pipeline.internships
            data_source = "database" if db_internships else "csv_fallback"
        else:
            internships_source = pipeline.internships
            data_source = "csv"

        recommendations = resumeRecommendInternship(
            student_profile, 
            internships_source, 
            top_n=top_n
        )

        formatted_recs = []
        company_counts = {}
        
        for score, intern, explanation, resume_data, ats_data in recommendations:
            formatted_recs.append({
                "score": float(score) if hasattr(score, 'item') else score,
                "internship": intern,
                "explanation": explanation,
                "skill_match": resume_data.get("skills", []),
                "ats_score": ats_data.get("ats_score", 0)
            })
            
            # Count companies efficiently
            company = intern.get("company", "").strip()
            if company:
                company_counts[company] = company_counts.get(company, 0) + 1
        
        # Build recommended companies list
        recommended_companies = [
            {"name": company, "count": count} 
            for company, count in company_counts.items()
        ]
        
        return jsonify({
            "success": True,
            "recommendations": formatted_recs,
            "recommended_companies": recommended_companies,
            "data_source": data_source,
            "total_internships_searched": len(internships_source),
            "parsed_resume": parsed_resume,
            "ats_analysis": ats_analysis
        })
        
    except Exception as e:
        print(f"Error in resume recommendation: {traceback.format_exc()}")
        return jsonify({"error": f"Recommendation failed: {str(e)}"}), 500


@app.route("/api/resume/suggestions", methods=["POST"])
def get_resume_suggestions():
    """Get improvement suggestions for resume"""
    if not HAS_RESUME_PARSER:
        return jsonify({"error": "Resume parser not available"}), 503
    
    data = request.get_json()
    resume_text = data.get("resume_text", "").strip()
    target_role = data.get("target_role", "software engineer")
    
    if not resume_text:
        return jsonify({"error": "Resume text required"}), 400
    
    try:
        parsed_resume = extractStructuredResumeData(resume_text)
        suggestions = generateResumeSuggestions(parsed_resume, target_role)
        
        return jsonify({
            "success": True,
            "suggestions": suggestions,
            "skill_gaps": parsed_resume.get("skill_gaps", {}),
            "skills_found": parsed_resume.get("skills", [])
        })
        
    except Exception as e:
        print(f"Error generating suggestions: {traceback.format_exc()}")
        return jsonify({"error": f"Failed to generate suggestions: {str(e)}"}), 500


@app.route("/api/resume/cover-letter", methods=["POST"])
def generate_cover_letter():
    """Generate cover letter template"""
    if not HAS_RESUME_PARSER:
        return jsonify({"error": "Resume parser not available"}), 503
    
    data = request.get_json()
    resume_text = data.get("resume_text", "").strip()
    internship_data = data.get("internship", {})
    
    if not resume_text:
        return jsonify({"error": "Resume text required"}), 400
    
    if not internship_data:
        return jsonify({"error": "Internship data required"}), 400
    
    try:
        parsed_resume = extractStructuredResumeData(resume_text)
        cover_letter = generateCoverLetterTemplate(parsed_resume, internship_data)
        
        return jsonify({
            "success": True,
            "cover_letter": cover_letter,
            "skills_used": parsed_resume.get("skills", [])[:3]
        })
        
    except Exception as e:
        print(f"Error generating cover letter: {traceback.format_exc()}")
        return jsonify({"error": f"Failed to generate cover letter: {str(e)}"}), 500

@app.route("/api/resume/recommend-db", methods=["POST"])
def resume_recommend_db():
    """Get recommendations exclusively from MySQL database (real companies only)"""
    if not HAS_RESUME_PARSER:
        return jsonify({"error": "Resume parser not available"}), 503
    
    data = request.get_json()
    resume_text = data.get("resume_text", "").strip()
    top_n = int(data.get("top_n", 5))  # FIX 2: Safe conversion

    if not resume_text:
        return jsonify({"error": "Resume text required"}), 400

    try:
        # Parse resume
        parsed_resume = extractStructuredResumeData(resume_text)
        ats_analysis = scoreResumeAts(resume_text)

        student_profile = {
            "skills": parsed_resume["skills"],
            "interests": [],
            "raw_resume": resume_text
        }

        # Fetch from database only
        db_internships = fetch_internships_from_db(limit=500)
        
        if not db_internships:
            return jsonify({
                "error": "No internships found in database",
                "hint": "Check database connection or data availability"
            }), 404

        # Get recommendations
        recommendations = resumeRecommendInternship(
            student_profile,
            db_internships,
            top_n=top_n
        )

        # Format results with efficient company counting
        formatted = []
        company_info = {}  # Store company details

        for score, intern, explanation, resume_data, ats_data in recommendations:
            formatted.append({
                "score": float(score) if hasattr(score, "item") else score,
                "internship": intern,
                "explanation": explanation,
                "skill_match": resume_data.get("skills", []),
                "ats_score": ats_data.get("ats_score", 0)
            })

            company = intern.get("company", "").strip()
            if company:
                if company not in company_info:
                    company_info[company] = {
                        "name": company,
                        "location": intern.get("location", ""),
                        "mode": intern.get("mode", ""),
                        "count": 0
                    }
                company_info[company]["count"] += 1

        # Convert to list
        companies = list(company_info.values())

        return jsonify({
            "success": True,
            "recommendations": formatted,
            "recommended_companies": companies,
            "data_source": "mysql_database",
            "total_searched": len(db_internships),
            "ats_analysis": ats_analysis,
            "parsed_resume": parsed_resume
        })
        
    except Exception as e:
        print(f"Error in DB recommendation: {traceback.format_exc()}")
        return jsonify({"error": f"DB recommendation failed: {str(e)}"}), 500


# ==================== DATABASE ROUTES ====================

@app.route("/api/health")
def health():
    """Health check endpoint with DB internship count (IMPROVEMENT 2: Fixed double connection)"""
    # IMPROVEMENT 2: Connect only once and close properly
    conn = get_db_connection()
    db_ok = conn is not None
    db_internship_count = 0
    
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM internships")
            db_internship_count = cursor.fetchone()[0]
            cursor.close()
        except Exception as e:
            print(f"Error counting internships: {e}")
        finally:
            conn.close()
    
    return jsonify({
        "status": "ok", 
        "chatbot": "active",
        "model_loaded": pipeline is not None,
        "resume_parser": HAS_RESUME_PARSER,
        "mysql": "connected" if db_ok else "unavailable",
        "db_internships_count": db_internship_count,
        "csv_internships_count": len(pipeline.internships) if pipeline else 0
    })


@app.route("/api/db/internships", methods=["GET"])
def get_internships_db():
    """Get internships from MySQL database"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, title, company, skills, location, mode, source, 
                   popularity, rating, company_prestige 
            FROM internships
            ORDER BY popularity DESC
            LIMIT 100
        """)
        
        internships = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return jsonify({
            "success": True,
            "count": len(internships),
            "internships": internships
        })
        
    except Error as e:
        print(f"Database error: {e}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    except Exception as e:
        print(f"Error: {traceback.format_exc()}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/api/db/internships/<int:internship_id>", methods=["GET"])
def get_internship_by_id(internship_id):
    """Get specific internship by ID from MySQL"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM internships WHERE id = %s", (internship_id,))
        
        internship = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if not internship:
            return jsonify({"error": "Internship not found"}), 404
        
        return jsonify({
            "success": True,
            "internship": internship
        })
        
    except Error as e:
        print(f"Database error: {e}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@app.route("/api/db/internships/search", methods=["GET"])
def search_internships_db():
    """Search internships in MySQL database"""
    try:
        keyword = request.args.get('q', '').strip()
        if not keyword:
            return jsonify({"error": "Search query required"}), 400
        
        connection = get_db_connection()
        if not connection:
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = connection.cursor(dictionary=True)
        search_pattern = f"%{keyword}%"
        cursor.execute("""
            SELECT id, title, company, skills, location, mode, source, 
                   popularity, rating, company_prestige 
            FROM internships
            WHERE title LIKE %s 
               OR company LIKE %s 
               OR skills LIKE %s
               OR location LIKE %s
            ORDER BY popularity DESC
            LIMIT 50
        """, (search_pattern, search_pattern, search_pattern, search_pattern))
        
        internships = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return jsonify({
            "success": True,
            "count": len(internships),
            "keyword": keyword,
            "internships": internships
        })
        
    except Error as e:
        print(f"Database error: {e}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@app.route("/api/db/internships/recommend", methods=["GET"])
def recommend_from_db():
    """Get recommendations from MySQL database based on skills"""
    try:
        skills = request.args.get('skills', '').strip()
        limit = int(request.args.get('limit', 10))
        
        if not skills:
            return jsonify({"error": "Skills parameter required"}), 400
        
        connection = get_db_connection()
        if not connection:
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = connection.cursor(dictionary=True)
        
        skill_list = [s.strip() for s in skills.split(',')]
        recommendations = []
        
        for skill in skill_list[:5]:  
            search_pattern = f"%{skill}%"
            cursor.execute("""
                SELECT id, title, company, skills, location, mode, source, 
                       popularity, rating, company_prestige 
                FROM internships
                WHERE skills LIKE %s
                ORDER BY popularity DESC
                LIMIT %s
            """, (search_pattern, limit))
            
            results = cursor.fetchall()
            for result in results:
                match_score = 0
                for s in skill_list:
                    if s.lower() in result['skills'].lower():
                        match_score += 1
                result['match_score'] = match_score
                recommendations.append(result)
        
        #Remove duplicates and sort by match score
        unique_recommendations = []
        seen_ids = set()
        for rec in recommendations:
            if rec['id'] not in seen_ids:
                seen_ids.add(rec['id'])
                unique_recommendations.append(rec)
        

        unique_recommendations.sort(
            key=lambda x: (x['match_score'], x['popularity']), 
            reverse=True
        )
        
        cursor.close()
        connection.close()
        
        return jsonify({
            "success": True,
            "count": len(unique_recommendations),
            "skills_searched": skill_list,
            "internships": unique_recommendations[:limit]
        })
        
    except Error as e:
        print(f"Database error: {e}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    except Exception as e:
        print(f"Error: {traceback.format_exc()}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# ==================== UTILITY FUNCTIONS ====================
def fetch_internships_from_db(limit=500):
    """
    Fetch internships from MySQL and convert to format expected by recommender.
    Maps DB schema (skills) -> recommender schema (required_skills)
    """
    connection = get_db_connection()
    if not connection:
        print("Warning: Could not connect to database for internships")
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, title, company, skills, location, mode, source,
                   popularity, rating, company_prestige
            FROM internships
            ORDER BY popularity DESC
            LIMIT %s
        """, (limit,))

        rows = cursor.fetchall()
        cursor.close()
        connection.close()

        # Convert DB schema to recommender schema
        internships = []
        for r in rows:
            # FIX 3: Map company_prestige VARCHAR to numeric score
            prestige_raw = (r.get("company_prestige") or "").lower()
            prestige_map = {
                "high": 9,
                "medium": 6,
                "low": 3
            }
            prestige_score = prestige_map.get(prestige_raw, 5)  # default to 5
            
            internships.append({
                "id": r["id"],
                "title": r["title"],
                "company": r["company"],
                "required_skills": r["skills"] or "",  # KEY: Map skills -> required_skills
                "description": "",  # DB doesn't have this field
                "location": r.get("location", ""),
                "mode": r.get("mode", "onsite"),
                "source": r.get("source", ""),
                "popularity": r.get("popularity", 0) or 0,
                "rating": float(r["rating"]) if r["rating"] is not None else 0,
                "company_prestige": prestige_score  # Now properly mapped
            })

        print(f"✓ Loaded {len(internships)} internships from database")
        return internships

    except Error as e:
        print(f"Database error while fetching internships: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {traceback.format_exc()}")
        return []
    
def extractTextFromPdf(file):
    """Extract text from PDF file"""
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Failed to read PDF: {str(e)}")


def extractTextFromTxt(file):
    """Extract text from TXT file"""
    try:
        content = file.read()
        if isinstance(content, bytes):
            return content.decode('utf-8', errors='ignore')
        return str(content)
    except Exception as e:
        raise Exception(f"Failed to read TXT file: {str(e)}")

# ==================== ROADMAP.PY ====================
@app.route("/roadmap", methods=["POST"])
def roadmap():
    data = request.get_json() or {}
    key = (data.get("track") or "").strip().lower()

    roadmap = ROADMAPS.get(key, ROADMAPS.get("general"))
    return jsonify({
        "track": key,
        "title": roadmap["title"],
        "duration": roadmap["duration"],
        "steps": roadmap["steps"],
        "resources": roadmap["resources"]
    })

# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Flask Application Starting...")
    print(f"✓ Chatbot (model.py): Active")
    print(f"{'✓' if HAS_RESUME_PARSER else '✗'} Resume Parser: {'Active' if HAS_RESUME_PARSER else 'Inactive'}")
    
    # IMPROVEMENT 2: Connect once and close properly
    conn = get_db_connection()
    db_ok = conn is not None
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM internships")
            count = cursor.fetchone()[0]
            print(f"✓ MySQL Database: Connected ({count} internships loaded)")
            cursor.close()
        except:
            print(f"✓ MySQL Database: Connected (unable to count internships)")
        finally:
            conn.close()
    else:
        print(f"✗ MySQL Database: Not Connected")
    
    print("="*60 + "\n")
    app.run(debug=True)