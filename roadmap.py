ROADMAPS = {
    "data science": {
        "title": "Data Scientist / Analyst Roadmap",
        "duration": "4–8 months",
        "steps": [
            "Master SQL fundamentals and practice with real datasets",
            "Learn Python data stack: Pandas, NumPy, Matplotlib/Seaborn",
            "Build 2–3 EDA projects with clear insights and visualizations",
            "Get comfortable with dashboard tools (Tableau or Power BI)",
            "Learn basic statistics (hypothesis testing, distributions)",
            "Create portfolio with business insights and storytelling",
            "Apply to 20+ roles, focusing on industries matching your projects"
        ],
        "resources": [
            {"name": "Kaggle Datasets", "url": "https://www.kaggle.com/datasets", "type": "Practice"},
            {"name": "Google Data Analytics Certificate (Coursera)", "url": "https://www.coursera.org/professional-certificates/google-data-analytics", "type": "Course"},
            {"name": "Mode SQL Tutorial", "url": "https://mode.com/sql-tutorial/", "type": "SQL"},
            {"name": "Tableau Public", "url": "https://public.tableau.com/", "type": "Portfolio"},
            {"name": "Coursera: Python for Everybody", "url": "https://www.coursera.org/specializations/python", "type": "Python"}
        ]
    },
    "machine learning": {
        "title": "Machine Learning Engineer / ML Scientist Roadmap",
        "duration": "6–12 months",
        "steps": [
            "Solidify Python programming and algorithms",
            "Learn scikit-learn end-to-end (data prep → model → evaluation)",
            "Build 3 ML projects (classification, regression, clustering)",
            "Deep dive into deep learning with TensorFlow/PyTorch",
            "Deploy a model using Flask/FastAPI or cloud (AWS SageMaker, GCP AI Platform)",
            "Master MLOps basics: versioning, CI/CD, monitoring",
            "Contribute to open source or publish on GitHub",
            "Prepare for coding interviews (LeetCode) and ML system design"
        ],
        "resources": [
            {"name": "Coursera: Andrew Ng ML Specialization", "url": "https://www.coursera.org/specializations/machine-learning-introduction", "type": "Course"},
            {"name": "Fast.ai Practical Deep Learning", "url": "https://course.fast.ai/", "type": "Course"},
            {"name": "MLOps Zoomcamp", "url": "https://github.com/DataTalksClub/mlops-zoomcamp", "type": "Course"},
            {"name": "Hugging Face NLP Course", "url": "https://huggingface.co/learn/nlp-course", "type": "NLP"},
            {"name": "Kaggle Competitions", "url": "https://www.kaggle.com/competitions", "type": "Practice"}
        ]
    },
    "web development": {
        "title": "Full-Stack Web Developer Roadmap",
        "duration": "5–9 months",
        "steps": [
            "HTML/CSS/JavaScript fundamentals (build 5 small UI projects)",
            "Learn React.js with state management (Redux/Context)",
            "Build a full-stack project (MERN or similar) with authentication",
            "Master backend development (Node.js/Express or Python/Django)",
            "Database design (SQL & NoSQL) and optimization",
            "Learn deployment (Docker, AWS/GCP, CI/CD pipelines)",
            "Practice system design for web applications",
            "Create portfolio with 3+ deployed projects"
        ],
        "resources": [
            {"name": "The Odin Project", "url": "https://www.theodinproject.com/", "type": "Full Course"},
            {"name": "Full Stack Open (University of Helsinki)", "url": "https://fullstackopen.com/en/", "type": "Course"},
            {"name": "freeCodeCamp Certifications", "url": "https://www.freecodecamp.org/learn/", "type": "Certification"},
            {"name": "Frontend Mentor", "url": "https://www.frontendmentor.io/", "type": "Practice"},
            {"name": "MDN Web Docs", "url": "https://developer.mozilla.org/", "type": "Reference"}
        ]
    },
    "cybersecurity": {
        "title": "Cybersecurity Analyst / Pentester Roadmap",
        "duration": "6–12 months",
        "steps": [
            "Networking fundamentals (TCP/IP, DNS, HTTP, firewalls)",
            "Linux command line and scripting (Bash/Python)",
            "Get hands-on with security tools (Wireshark, Nmap, Metasploit)",
            "Complete TryHackMe/HTB beginner paths",
            "Learn vulnerability assessment and penetration testing methodology",
            "Study compliance standards (ISO 27001, NIST, GDPR)",
            "Get entry-level cert (CompTIA Security+, CEH practical)",
            "Participate in CTF competitions and bug bounty programs"
        ],
        "resources": [
            {"name": "TryHackMe Learning Paths", "url": "https://tryhackme.com/paths", "type": "Hands-on"},
            {"name": "IBM Cybersecurity Analyst Certificate (Coursera)", "url": "https://www.coursera.org/professional-certificates/ibm-cybersecurity-analyst", "type": "Course"},
            {"name": "The Cyber Mentor - Ethical Hacking", "url": "https://academy.tcm-sec.com/p/practical-ethical-hacking-the-complete-course", "type": "Course"},
            {"name": "OWASP Web Security Testing Guide", "url": "https://owasp.org/www-project-web-security-testing-guide/", "type": "Reference"},
            {"name": "Hack The Box", "url": "https://www.hackthebox.com/", "type": "Practice"}
        ]
    },
    "general": {
        "title": "Software Engineering Fundamentals",
        "duration": "Ongoing",
        "steps": [
            "Master at least one programming language deeply",
            "Learn data structures and algorithms (LeetCode/HackerRank)",
            "Practice version control (Git/GitHub) and collaboration workflows",
            "Understand software development lifecycle (Agile/Scrum)",
            "Learn debugging, testing, and clean code principles",
            "Study system design basics (scalability, databases, APIs)",
            "Build communication and documentation skills",
            "Create a strong LinkedIn/GitHub profile"
        ],
        "resources": [
            {"name": "LeetCode 75 Study Plan", "url": "https://leetcode.com/studyplan/leetcode-75/", "type": "Practice"},
            {"name": "GitHub Student Developer Pack", "url": "https://education.github.com/pack", "type": "Tools"},
            {"name": "Google Tech Dev Guide", "url": "https://techdevguide.withgoogle.com/", "type": "Guide"},
            {"name": "System Design Primer (GitHub)", "url": "https://github.com/donnemartin/system-design-primer", "type": "Reference"},
            {"name": "freeCodeCamp Interview Prep", "url": "https://www.freecodecamp.org/learn/coding-interview-prep/", "type": "Interview"}
        ]
    }
}

def get_roadmap(role: str):
    if not role:
        return []
    if role in ROADMAPS:
        return ROADMAPS[role]
    #partial match
    for key in ROADMAPS:
        if key.lower() in role.lower() or role.lower() in key.lower():
            return ROADMAPS[key]
    return []