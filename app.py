from flask import Flask, render_template, request, redirect, url_for, session, abort
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
from flask import flash
import os
import openai
from flask_login import login_required, login_user, current_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from flask_login import LoginManager, UserMixin
from models import db, User, Resume, ResumeInput, UsageLog
import docx
from PyPDF2 import PdfReader
from striprtf.striprtf import rtf_to_text
from werkzeug.utils import secure_filename


# Initialize the Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://dennischarles:Camastra12%40@localhost/career_coach'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize database and migration manager
db.init_app(app)
migrate = Migrate(app, db)

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/")
def home():
    current_year = datetime.now().year
    return render_template("home.html", current_year=current_year)

@app.route("/admin")
def admin_dashboard():
    resume_count = Resume.query.count()
    generate_count = UsageLog.query.filter_by(action="generate_resume").count()
    return render_template("admin_dashboard.html", resume_count=resume_count, generate_count=generate_count)

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'rtf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file_path, ext):
    if ext == 'pdf':
        text = ""
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.replace('\x00', '')
    elif ext == 'docx':
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.replace('\x00', '')
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().replace('\x00', '')
    elif ext == 'rtf':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return rtf_to_text(f.read()).replace('\x00', '')
    else:
        raise ValueError("Unsupported file type")


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_resume():
    if request.method == "POST":
        # 1. Validate file existence
        if 'resume' not in request.files:
            return "No file part", 400

        file = request.files['resume']
        if file.filename == '':
            return "No selected file", 400

        # 2. Ensure upload directory exists
        upload_folder = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)  # Critical fix

        # 3. Handle photo upload
        photo_path = None
        if 'photo' in request.files:
            photo = request.files['photo']
            if photo.filename != '' and allowed_photo(photo.filename):
                photo_filename = secure_filename(photo.filename)
                photo_path = os.path.join(upload_folder, photo_filename)
                try:
                    photo.save(photo_path)
                except Exception as e:
                    return f"Photo save error: {str(e)}", 500

        # 4. Process resume file
        selected_template = request.form.get("template", "professional")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            ext = filename.rsplit('.', 1)[1].lower()
            path = os.path.join(upload_folder, filename)

            try:
                # Save resume first
                file.save(path)

                # Extract and process content
                content = extract_text(path, ext)
                professional_resume = generate_professional_resume(content)

                # 5. Database operations
                enhanced_resume = Resume(
                    user_id=current_user.id,
                    type="ai_enhanced",
                    original_name=f"Enhanced_{filename}",
                    text_content=professional_resume,
                    ai_feedback="Automatically enhanced to professional standards",
                    photo_path=photo_path,
                    template_name=selected_template,
                    created_at=datetime.now()
                )

                original_resume = Resume(
                    user_id=current_user.id,
                    type="uploaded_original",
                    original_name=filename,
                    text_content=content,
                    created_at=datetime.now()
                )

                db.session.add(original_resume)
                db.session.add(enhanced_resume)
                db.session.commit()

                return redirect(url_for('view_enhanced_resume', id=enhanced_resume.id))

            except Exception as e:
                # 6. Comprehensive error cleanup
                if photo_path and os.path.exists(photo_path):
                    os.remove(photo_path)
                if path and os.path.exists(path):
                    os.remove(path)
                return f"Error processing file: {str(e)}", 500

        return "Invalid file type. Supported formats: PDF, DOCX, TXT, RTF", 400

    return render_template("upload.html")


# Helper function for photo validation
def allowed_photo(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


@app.route("/resume/<int:id>")
def view_resume(id):
    resume = Resume.query.get_or_404(id)
    return render_template("resume_view.html", resume=resume)

@app.route("/resumes")
@login_required
def resume_list():
    resumes = Resume.query.filter_by(user_id=current_user.id).order_by(Resume.created_at.desc()).all()
    return render_template("resumes_list.html", resumes=resumes)

@app.route("/resume/new")
def new_resume_form():
    return render_template("generate_resume.html")


def generate_professional_resume(original_content):
    """Generate a world-class professional resume from uploaded content"""
    try:
        # Comprehensive analysis and enhancement prompt
        prompt = f"""
        You are a world-class resume writer and career coach. Analyze the following resume content and create a professional, ATS-optimized resume that meets international standards.

        ORIGINAL RESUME CONTENT:
        {original_content}

        REQUIREMENTS:
        1. Create a modern, professional resume structure with these sections:
           - Contact Information (clean, professional format)
           - Professional Summary (2-3 powerful sentences)
           - Core Skills (ATS-friendly keyword optimization)
           - Professional Experience (with quantified achievements)
           - Education
           - Certifications (if applicable)

        2. ENHANCEMENT GUIDELINES:
           - Use strong action verbs and quantifiable achievements
           - Optimize for Applicant Tracking Systems (ATS)
           - Include industry-relevant keywords naturally
           - Ensure consistent formatting and professional language
           - Focus on results and impact rather than just responsibilities
           - Use proper resume formatting with clear section headers
           - Keep bullet points concise and impactful
           - Maintain chronological order for work experience

        3. PROFESSIONAL STANDARDS:
           - Use formal business language
           - Ensure error-free grammar and spelling
           - Apply consistent date formatting (MM/YYYY)
           - Include measurable results where possible (percentages, numbers, achievements)
           - Tailor language to be professional yet engaging
           - Remove any unprofessional elements or excessive personal information

        4. ATS OPTIMIZATION:
           - Use standard section headings
           - Include relevant keywords from common job descriptions in the field
           - Avoid graphics, tables, or unusual formatting
           - Use simple, clean bullet points
           - Ensure proper spacing and readability

        Generate a complete, professional resume that would pass both ATS systems and impress hiring managers. Format it cleanly with clear section breaks and professional presentation.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are an expert resume writer specializing in creating world-class, ATS-optimized professional resumes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent, professional output
            max_tokens=2000
        )

        return response.choices[0].message['content'].strip()

    except Exception as e:
        return f"⚠️ AI Enhancement Error: {str(e)}"


@app.route("/resume/enhanced/<int:id>")
@login_required
def view_enhanced_resume(id):
    resume = Resume.query.get_or_404(id)
    if resume.user_id != current_user.id:
        abort(403)
    return render_template("enhanced_resume_view.html", resume=resume)


@app.route("/resume/compare/<int:original_id>/<int:enhanced_id>")
@login_required
def compare_resumes(original_id, enhanced_id):
    original = Resume.query.get_or_404(original_id)
    enhanced = Resume.query.get_or_404(enhanced_id)

    if original.user_id != current_user.id or enhanced.user_id != current_user.id:
        abort(403)

    return render_template("compare_resumes.html", original=original, enhanced=enhanced)


# Delete route with explicit endpoint
@app.route("/resume/<int:id>/delete", methods=["POST"], endpoint='delete_resume_endpoint')
@login_required
def delete_resume(id):
    resume = Resume.query.get_or_404(id)
    if resume.user_id != current_user.id:
        abort(403)
    db.session.delete(resume)
    db.session.commit()
    return redirect(url_for('my_resumes'))

# View route with explicit endpoint
@app.route("/resume/<int:id>", endpoint='view_resume_endpoint')
@login_required
def view_resume(id):
    resume = Resume.query.get_or_404(id)
    if resume.user_id != current_user.id:
        abort(403)
    return render_template("resume_view.html", resume=resume)

# Registration part for the user
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if password != confirm_password:
            return render_template("register.html", error="Passwords do not match")

        hashed_password = generate_password_hash(password)
        user = User(name=username, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(email=request.form['email']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('home'))
    return render_template("login.html")


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))  # Changed to redirect

@app.route('/career-advice', methods=['GET', 'POST'])
def career_advice():
    advice = None
    if request.method == 'POST':
        question = request.form['question']

        if not question:
            return "Please ask a valid question.", 400

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert career advisor."},
                    {"role": "user", "content": question}
                ],
                max_tokens=10000
            )
            advice = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"An error occurred: {str(e)}"

    return render_template("career_advice.html", advice=advice)


@app.route("/cover-letter", methods=["GET", "POST"])
@login_required
def cover_letter():
    if request.method == "POST":
        resume_text = request.form.get("resume_text", "").strip()
        job_info = request.form.get("job_info", "").strip()

        if not resume_text or not job_info:
            flash("Please provide both resume text and job information", "error")
            return render_template("cover_letter.html")

        try:
            prompt = f"""
            Write a professional cover letter with these sections:
            1. Header: Applicant contact info
            2. Date and hiring manager details
            3. Salutation
            4. Opening paragraph: Mention position and enthusiasm
            5. Body: 2-3 paragraphs matching skills to job requirements
            6. Closing paragraph: Call to action
            7. Professional closing

            Job Details:
            {job_info}

            Applicant Resume Summary:
            {resume_text}

            Important:
            - Use formal business letter format
            - Keep under 400 words
            - Include quantifiable achievements
            - Address to "Hiring Manager" if no name available
            """

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional resume writer"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1200
            )
            letter = response.choices[0].message["content"].strip()

            return render_template("cover_letter.html", letter=letter)

        except Exception as e:
            flash(f"Error generating cover letter: {str(e)}", "error")
            return render_template("cover_letter.html")

    # Return the form on GET requests
    return render_template("cover_letter.html")


@app.route("/interview-tips", methods=["GET", "POST"])
def interview_tips():
    tips = None
    if request.method == "POST":
        job_desc = request.form["job_desc"]
        prompt = f"""
        I'm preparing for an interview. Can you give me a list of 10 specific and relevant interview questions (with answers if possible) for a position in: {job_desc}?
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        tips = response.choices[0].message["content"]
    return render_template("interview_tips.html", tips=tips)

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.route("/my_resumes")
@login_required
def my_resumes():
    resumes = Resume.query.filter_by(user_id=current_user.id).all()
    return render_template("resumes_list.html", resumes=resumes)



@app.route('/download-template-resume', methods=['POST'])
@login_required
def download_template_resume():
    data = request.get_json()
    template_type = data['template']
    color_option = data['colorOption']
    content = data['content']
    pass



@app.context_processor
def inject_now():
    return {'now': datetime.now}

@app.errorhandler(429)
def too_many_requests(e):
    return render_template("429.html"), 429

if __name__ == "__main__":
    app.run(debug=True, port=5001)
