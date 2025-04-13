from flask import Flask, render_template, request, jsonify,redirect,url_for
import os
import json
import cv2
import pytesseract as ts
import pdfplumber as pdp
from PIL import Image
from werkzeug.utils import secure_filename
import cv2
import pytesseract as ts
import json
import pdfplumber as pdp
import magic
import os
import numpy as np
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")

# Load BERT Model
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Configure Google Gemini
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Global variables
resume_json_path = ""
jd_json_path = ""
resume_text = ""
jd_text = ""

def save_to_json(file_path, extracted_text):
    output_filename = os.path.splitext(os.path.basename(file_path))[0] + "_resume.json"
    output_path = os.path.join(os.path.dirname(file_path), output_filename)
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump({"file_name": os.path.basename(file_path), "extracted_text": extracted_text}, json_file, indent=4, ensure_ascii=False)
    return output_path

def extract_text_from_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return ts.image_to_string(img).strip()

def extract_text_from_pdf(file_path):
    text = ""
    with pdp.open(file_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text.strip() + " "
    return text

def process_resume(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".pdf"}:
        return extract_text_from_pdf(file_path)
    elif ext in {".jpg", ".jpeg", ".png"}:
        return extract_text_from_image(file_path)
    else:
        raise ValueError("Unsupported file format")

def load_extracted_text(json_path):
    if not json_path or not os.path.exists(json_path):
        return ""  # Return empty string instead of raising an error
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data.get("extracted_text", "")
    except UnicodeDecodeError:
        with open(json_path, "r", encoding="ISO-8859-1") as file:
            data = json.load(file)
        return data.get("extracted_text", "")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return "Handling POST request"
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global resume_json_path, jd_json_path, resume_text, jd_text

    if 'resume' not in request.files or 'job_description' not in request.files:
        return jsonify({'error': 'Missing files'}), 400

    resume_file = request.files['resume']
    jd_file = request.files['job_description']

    if resume_file.filename == "" or jd_file.filename == "":
        return jsonify({'error': 'No selected file'}), 400

    resume_filename = secure_filename(resume_file.filename)
    jd_filename = secure_filename(jd_file.filename)

    upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)

    resume_path = os.path.join(upload_folder, resume_filename)
    jd_path = os.path.join(upload_folder, jd_filename)

    resume_file.save(resume_path)
    jd_file.save(jd_path)

    try:
        resume_text = process_resume(resume_path)
        jd_text = extract_text_from_pdf(jd_path)

        # Debug print (optional)
        print("Resume Text (preview):", resume_text[:300])
        print("JD Text (preview):", jd_text[:300])

        # Handle empty text cases
        if not resume_text.strip() or not jd_text.strip():
            return jsonify({'error': 'Text data is missing, please upload files again'}), 400

        resume_json_path = save_to_json(resume_path, resume_text)
        jd_json_path = save_to_json(jd_path, jd_text)

        return redirect(url_for('scoring'))

    except Exception as e:
        print(f"Error processing files: {e}")
        return jsonify({'error': str(e)}), 500


# Industry Keywords
industry_keywords = {
    "AI/ML": [
        "machine learning", "artificial intelligence", "deep learning", "neural networks",
        "computer vision", "natural language processing", "NLP", "reinforcement learning",
        "generative AI", "predictive modeling", "supervised learning", "unsupervised learning",
        "semi-supervised learning", "clustering", "decision trees", "random forest",
        "support vector machines", "GANs", "transformers", "BERT", "GPT", "LLM", "gradient boosting",
        "hyperparameter tuning", "feature engineering", "data augmentation", "backpropagation",
        "optimization algorithms", "convolutional neural networks", "recurrent neural networks",
        "Bayesian networks", "Markov models", "autoencoders", "word embeddings", "attention mechanism", "scikit learn",
        "pandas"
    ],

    "Web Development": [
        "HTML", "CSS", "JavaScript", "React", "Angular", "Vue.js", "Node.js", "Express.js",
        "Next.js", "Svelte", "Bootstrap", "Tailwind CSS", "SASS", "TypeScript", "jQuery",
        "REST APIs", "GraphQL", "AJAX", "WebSockets", "SSR", "CSR", "PWA", "Django", "Flask",
        "Spring Boot", "ASP.NET", "Ruby on Rails", "Laravel", "WordPress", "Shopify",
        "Frontend Development", "Backend Development", "Full-Stack Development"
    ],

    "App Development": [
        "Android", "iOS", "Flutter", "React Native", "Swift", "Kotlin", "Dart", "Objective-C",
        "Jetpack Compose", "SwiftUI", "Xamarin", "Cordova", "Ionic", "Firebase", "Google Play Store",
        "App Store Optimization (ASO)", "Mobile UI/UX", "Push Notifications", "Geolocation",
        "Offline Storage", "Bluetooth Connectivity", "Cross-Platform Development"
    ],

    "Software Development": [
        "Java", "Python", "C#", "C++", "Rust", "Go", "Ruby", "Perl", "Shell Scripting",
        "Version Control", "Git", "GitHub", "GitLab", "CI/CD", "Docker", "Kubernetes",
        "Agile Development", "Scrum", "Test-Driven Development", "Microservices",
        "Cloud Computing", "AWS", "Azure", "Google Cloud Platform", "DevOps", "System Architecture"
    ],

    "Cyber Security": [
        "penetration testing", "ethical hacking", "firewall", "intrusion detection", "SIEM",
        "zero trust", "ransomware", "phishing", "malware analysis", "forensics", "SOC",
        "incident response", "threat intelligence", "public key infrastructure (PKI)",
        "VPN", "IDS/IPS", "OWASP", "vulnerability assessment", "red teaming", "blue teaming",
        "CISO", "zero-day exploits", "cloud security", "identity management", "SIEM", "python"
    ],

    "Healthcare": [
        "electronic health records", "EHR", "clinical trials", "medical imaging", "telemedicine",
        "pharmaceuticals", "biotechnology", "FDA compliance", "ICD-10", "HL7", "FHIR", "HIPAA",
        "medical diagnostics", "genomics", "precision medicine", "health informatics",
        "public health", "epidemiology", "robotic surgery", "medical AI"
    ],

    "Finance": [
        "investment banking", "risk management", "financial modeling", "trading",
        "stock market", "hedge funds", "derivatives", "blockchain", "cryptocurrency",
        "DeFi", "fintech", "robo-advisors", "algorithmic trading", "portfolio management",
        "credit risk", "Basel III", "AML compliance", "forensic accounting", "regulatory compliance",
        "quantitative finance", "venture capital", "private equity"
    ],

    "Marketing": [
        "SEO", "SEM", "digital marketing", "content marketing", "influencer marketing",
        "social media marketing", "Facebook Ads", "Google Ads", "email marketing",
        "brand positioning", "customer segmentation", "market research", "consumer behavior",
        "conversion rate optimization", "A/B testing", "growth hacking", "public relations (PR)",
        "media buying", "advertising", "lead generation", "copywriting"
    ],

    "Data Science": [
        "data analysis", "big data", "data engineering", "ETL", "data warehousing",
        "data visualization", "Power BI", "Tableau", "Apache Spark", "Hadoop",
        "feature selection", "dimensionality reduction", "EDA", "time series analysis",
        "data governance", "data quality", "database management", "SQL", "NoSQL", "data pipelines", "python", "pandas",
        "statistics"
    ],

    "Business Analysis": [
        "business intelligence", "KPI tracking", "OKRs", "market analysis", "financial analysis",
        "data-driven decision making", "process optimization", "requirement gathering",
        "stakeholder management", "cost-benefit analysis", "data storytelling", "case study analysis",
        "business process modeling", "product management", "competitive analysis", "ROI analysis",
        "customer journey mapping", "business strategy"
    ],

    "Blockchain & Web3": [
        "Ethereum", "Solidity", "smart contracts", "NFTs", "DeFi", "crypto wallets",
        "DApps", "Layer 2 solutions", "Polygon", "Solana", "Web3.js", "Metamask",
        "decentralized governance", "blockchain security", "hashing algorithms",
        "staking", "consensus mechanisms", "Bitcoin", "hyperledger", "permissioned blockchain"
    ],

    "Cloud Computing": [
        "AWS", "Azure", "Google Cloud", "cloud storage", "serverless computing",
        "Kubernetes", "Docker", "CI/CD", "Terraform", "cloud security", "hybrid cloud",
        "multi-cloud strategy", "virtualization", "IAM", "cloud-native applications",
        "microservices architecture", "edge computing", "cloud automation"
    ],

    "DevOps": [
        "CI/CD", "Jenkins", "Ansible", "Terraform", "Kubernetes", "Docker",
        "GitOps", "infrastructure as code", "monitoring and logging", "prometheus",
        "grafana", "ELK stack", "deployment automation", "SRE", "containerization",
        "cloud orchestration", "agile methodologies"
    ],

    "Internet of Things (IoT)": [
        "IoT sensors", "edge computing", "M2M communication", "wireless networks",
        "5G", "smart home", "embedded systems", "real-time analytics", "low-power networks",
        "cyber-physical systems", "digital twins", "IoT security"
    ],

    "Robotics": [
        "robotics engineering", "autonomous systems", "ROS", "sensor fusion",
        "robot perception", "manipulators", "human-robot interaction", "swarm robotics",
        "drones", "robotic process automation (RPA)", "motion planning", "SLAM"
    ]
}
def match_resume_with_jd(resume_text, jd_text, bert_model):
    """
    Matches resume text with job description text using TF-IDF and BERT similarity.

    Args:
        resume_text (str): Extracted resume text.
        jd_text (str): Extracted job description text.
        bert_model (SentenceTransformer): Preloaded BERT model for embedding similarity.

    Returns:
        float: Final resume-to-JD match score (0-100%).
    """
    # TF-IDF Similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    tfidf_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    # BERT Semantic Similarity
    resume_embedding = bert_model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = bert_model.encode(jd_text, convert_to_tensor=True)
    bert_score = util.pytorch_cos_sim(resume_embedding, jd_embedding).item()

    # Normalize scores using MinMaxScaler
    scores = np.array([[tfidf_score], [bert_score]])
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()

    # Weighted ATS Score
    final_score = (normalized_scores[0] * 0.40) + (normalized_scores[1] * 0.60)

    return round(final_score * 100, 2)


# **Step 1: TF-IDF Cosine Similarity**
def compute_tfidf_similarity(resume_text, keywords):
    vectorizer = TfidfVectorizer(vocabulary=keywords)
    tfidf_matrix = vectorizer.fit_transform([resume_text, " ".join(keywords)])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
# Flatten keywords list
all_keywords = [kw for keywords in industry_keywords.values() for kw in keywords]
all_keywords = [word.lower() for word in all_keywords]
all_keywords = list(set(all_keywords))
tfidf_score = compute_tfidf_similarity(resume_text, all_keywords)


# print(f"TF-IDF Cosine Similarity Score: {tfidf_score:.2f}")

# **Step 2: Gemini Keyword Matching**
def gemini_keyword_matching(text, industry_keywords):
    prompt = f"""
    Identify industry-specific keywords from the following resume text. 
    Also find more keywords related to the industry outside of given list.
    Interpret the job title and match only the keywords relevant to it .

    Keywords: {', '.join(industry_keywords)}.

    Resume:
    {text}

    Return matched keywords separated by commas.
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    matched_keywords = response.text.strip().split(", ")
    return matched_keywords


matched_keywords = gemini_keyword_matching(resume_text, all_keywords)
match_percentage = len(matched_keywords) / (len(set(matched_keywords) | set(all_keywords)))


# print(f"Matched Keywords: {matched_keywords[1:]}")
# print(f"Gemini Keyword Match Score: {match_percentage:.2f}")

# **Step 3: BERT Semantic Similarity**
def compute_bert_similarity(resume_text, keywords):
    resume_embedding = bert_model.encode(resume_text)
    keyword_embedding = bert_model.encode(" ".join(keywords))
    return cosine_similarity([resume_embedding], [keyword_embedding])[0][0]


bert_score = compute_bert_similarity(resume_text, all_keywords)
# print(f"BERT Semantic Similarity Score: {bert_score:.2f}")

import re
import json

def get_gemini_feedback(resume_text, job_description, ats_score):
    genai.configure(api_key=GEMINI_API_KEY)

    prompt = f"""
    You are an ATS Resume Optimization Expert. Based on the resume and job description below,
    provide feedback for improving the resume to optimize ATS performance.

    The ATS score is {ats_score:.2f}%.

    ONLY return valid JSON (no explanations, no extra text).

    === Job Description ===
    {job_description}

    === Resume ===
    {resume_text}

    Respond in this exact JSON format:

    {{
      "overall_assessment": "Brief summary of strengths & weaknesses.",
      "missing_keywords": ["keyword1", "keyword2", "keyword3"],
      "formatting_issues": ["Issue 1", "Issue 2", "Issue 3"],
      "content_improvements": ["Suggestion 1", "Suggestion 2", "Suggestion 3"],
      "final_score_estimate": 
    }}
    """

    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    raw_text = response.text.strip()

    try:
        # Try parsing directly
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to extract JSON substring using regex
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Fallback dummy
        return {
            "overall_assessment": "Error parsing response.",
            "missing_keywords": [],
            "formatting_issues": [],
            "content_improvements": [],
            "final_score_estimate": ats_score
        }


import numpy as np
# Reshape for sklearn transformer
scores = np.array([[tfidf_score], [match_percentage], [bert_score]])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
normalized_scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()

import re
def extract_numeric_score(response_text):
    """Extracts the first valid numeric score (0-100) from the response."""
    numbers = re.findall(r'\d+', response_text)  # Find all numbers in response
    if numbers:
        score = int(numbers[0])  # Take the first number found
        return min(max(score, 0), 100)  # Ensure score is within 0-100
    return 0  # Default to 0 if no valid number is found

def rate_skills(resume_text, job_description):
    prompt = f"""
    You are an expert ATS Resume Evaluator. Analyze the resume against the job description.
    Rate the relevance of the candidate's skills to the job description on a scale of 0 to 100.

    === Job Description ===
    {job_description}

    === Resume ===
    {resume_text}

    Provide only a numeric score (0-100) without any.
    Just give the score , no other word other than that .
    """
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    return extract_numeric_score(response.text)


def rate_readability(resume_text):
    prompt = f"""
    You are an ATS Resume Formatting Expert. Analyze the resume for readability, formatting, 
    structure, and ease of understanding. Consider bullet points, font consistency, section clarity, and spacing.

    Rate the resume’s readability on a scale of 0 to 100.

    === Resume ===
    {resume_text}

    Provide only a numeric score (0-100).
    Just give the score , no other word other than that .
    """
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    return extract_numeric_score(response.text)


def rate_vocabulary(resume_text):
    prompt = f"""
    You are a Resume Language Specialist. Analyze the resume for professional vocabulary, 
    grammatical accuracy, and the use of strong action verbs.

    Rate the vocabulary and grammar quality on a scale of 0 to 100.

    === Resume ===
    {resume_text}

    Provide only a numeric score (0-100) .
    Just give the score , no other word other than that .
    """
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    return extract_numeric_score(response.text)


def rate_experience(resume_text, job_description):
    prompt = f"""
    You are an ATS Resume Evaluator. Analyze the experience section of the resume against 
    the job description. Consider relevance, duration, and quality of past roles.

    Rate the candidate's experience alignment with the job on a scale of 0 to 100.

    === Job Description ===
    {job_description}

    === Resume ===
    {resume_text}

    Provide only a numeric score (0-100) .
    Just give the score , no other word other than that .
    """
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    return extract_numeric_score(response.text)


def rate_projects(resume_text, job_description):
    prompt = f"""
    You are a Resume Project Reviewer. Analyze the relevance and impact of the projects 
    mentioned in the resume based on the job description.

    Rate the project relevance and quality on a scale 0 to 100.

    === Job Description ===
    {job_description}

    === Resume ===
    {resume_text}

    Provide only a numeric score (0-100).
    Just give the score , no other word other than that .
    """
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    return extract_numeric_score(response.text)


def rate_keywords(resume_text, job_description):
    prompt = f"""
    You are an ATS Optimization Expert. Analyze how many important job-related keywords 
    from the job description are present in the resume.

    Rate the keyword match on a scale of 0 to 100.

    === Job Description ===
    {job_description}

    === Resume ===
    {resume_text}

    Provide only a numeric score (0-100).
    Just give the score , no other word other than that .
    """
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    return extract_numeric_score(response.text)

def evaluate_resume(resume_text, job_description):
    scores = {
        "skills": [rate_skills(resume_text, job_description)],
        "readability": [rate_readability(resume_text)],
        "vocabulary": [rate_vocabulary(resume_text)],
        "experience": [rate_experience(resume_text, job_description)],
        "projects": [rate_projects(resume_text, job_description)],
        "keywords": [rate_keywords(resume_text, job_description)],

    }

    with open("../resume_evaluation.json", "w") as f:
        json.dump(scores, f, indent=4)

    return scores


params = evaluate_resume(resume_text,jd_text)
# Final ATS Score
ats_score = (normalized_scores[0] * 0.3) + (normalized_scores[1] * 0.4) + (normalized_scores[2] * 0.3)
feedback = get_gemini_feedback(resume_text,jd_text,ats_score)
# Print results
# print(f"Transformed TF-IDF Score: {scores[0]:}")
# print(f"Transformed Gemini Score: {scores[1]:}")
# print(f"Transformed BERT Score: {scores[2]:}")
#match_jd = match_resume_with_jd(resume_text, jd_text, bert_model)
#print(f"matching based on job description : {match_jd}%")
#print(f"Final ATS Score: {float(ats_score) * 100:.2f}%")
# Save the improvement feedback to a JSON file

@app.route('/scoring', methods=['GET', 'POST'])
def scoring():
    global resume_text, jd_text

    if not resume_text or not jd_text:
        return jsonify({'error': 'Text data is missing, please upload files again'}), 400

    # Get evaluation results
    params = evaluate_resume(resume_text, jd_text)
    keywords = list(map(str, params.get("keywords", [])))
    skills = list(map(str, params.get("skills", [])))
    readability = list(map(str, params.get("readability", [])))
    experience = list(map(str, params.get("experience", [])))
    vocabulary = list(map(str, params.get("vocabulary", [])))
    projects = list(map(str, params.get("projects", [])))

    # Match score calculation
    match_score = match_resume_with_jd(resume_text, jd_text, bert_model)

    # Compute ATS score and get feedback
    ats_score = (normalized_scores[0] * 0.2) + (normalized_scores[1] * 0.4) + (normalized_scores[2] * 0.4)
    ats_score = round(ats_score * 100, 2)
    feedback = get_gemini_feedback(resume_text, jd_text, ats_score)
    overall_assessment = feedback.get("overall_assessment")
    # Extract feedback parameters

    missing_keywords = feedback.get("missing_keywords", [])
    formatting_issues = feedback.get("formatting_issues", [])
    content_improvements = feedback.get("content_improvements", [])
    final_score_estimate = feedback.get("final_score_estimate", "N/A")

    return render_template(
        'result.html',
        match_score=ats_score,
        job_match_score=match_score,
        skills=skills,
        readability=readability,
        vocabulary=vocabulary,
        experience=experience,
        projects=projects,
        keywords=keywords,
        overall_assessment=feedback.get("overall_assessment", ""),
        missing_keywords=feedback.get("missing_keywords", []),
        formatting_issues=feedback.get("formatting_issues", []),
        content_improvements=feedback.get("content_improvements", []),
        final_score_estimate=feedback.get("final_score_estimate", 0)
    )

if __name__ == '__main__':
    app.run(debug=True)