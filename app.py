from flask import Flask, render_template, request, redirect, url_for, session, abort, flash
from flask_migrate import Migrate
from flask_login import LoginManager, login_required, login_user, current_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from dotenv import load_dotenv
import os
import openai
import docx
from PyPDF2 import PdfReader
from striprtf.striprtf import rtf_to_text
from models import db
from flask import make_response, render_template_string
from weasyprint import HTML
import markdown
import re

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL").strip("'").strip('"')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Create uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize extensions
from models import db
db.init_app(app)
migrate = Migrate(app, db)

# Initialize LoginManager (SINGLE initialization)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# Import models after db initialization
from models import User, Resume, ResumeInput, UsageLog, CoverLetter

# User loader function (SINGLE implementation)
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


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



@app.route("/resumes")
@login_required
def resume_list():
    resumes = Resume.query.filter_by(user_id=current_user.id).order_by(Resume.created_at.desc()).all()
    return render_template("resumes_list.html", resumes=resumes)

def generate_professional_resume(original_content):
    """Enhance and professionally rewrite the uploaded resume content only—do not add, comment, or explain."""
    try:
        prompt = f"""
                  Here's my resune , please enhance it 
         {original_content}
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": """You are an expert resume writer. 
                            Rewrite and enhance the following resume content to be more clear, impactful, and professional. 
                            Do NOT add new sections, do NOT add any comments or instructions, and do NOT include any explanations or extra text. 
                            Return ONLY the improved resume content, preserving the user's structure and sections as much as possible."""
                 },

                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
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
    html_content = markdown.markdown(resume.text_content)
    return render_template(
        "enhanced_resume_view.html",
        resume=resume,
        html_content=html_content,
        resume_markdown=resume.text_content
    )

@app.route("/resume/compare/<int:original_id>/<int:enhanced_id>")
@login_required
def compare_resumes(original_id, enhanced_id):
    original = Resume.query.get_or_404(original_id)
    enhanced = Resume.query.get_or_404(enhanced_id)

    if original.user_id != current_user.id or enhanced.user_id != current_user.id:
        abort(403)

    return render_template("compare_resumes.html", original=original, enhanced=enhanced)


# Deleting the route with explicit endpoint by the user
@app.route("/resume/<int:id>/delete", methods=["POST"], endpoint='delete_resume_endpoint')
@login_required
def delete_resume(id):
    resume = Resume.query.get_or_404(id)
    if resume.user_id != current_user.id:
        abort(403)
    db.session.delete(resume)
    db.session.commit()
    return redirect(url_for('my_resumes'))

# Viewing the  route with explicit endpoint
@app.route("/resume/<int:id>", endpoint='view_resume_endpoint')
@login_required
def view_resume(id):
    resume = Resume.query.get_or_404(id)
    if resume.user_id != current_user.id:
        abort(403)
    return render_template("resume_view.html", resume=resume)




# Update Resume - edit resume text
@app.route("/resume/<int:id>/edit", methods=["GET", "POST"])
@login_required
def edit_resume(id):
    resume = Resume.query.get_or_404(id)
    if resume.user_id != current_user.id:
        abort(403)

    if request.method == "POST":
        new_text = request.form.get("text_content", "").strip()
        new_template = request.form.get("template_name", "").strip()

        if not new_text:
            flash("Resume content cannot be empty", "error")
            return render_template("edit_resume.html", resume=resume)

        resume.text_content = new_text
        if new_template:
            resume.template_name = new_template

        db.session.commit()
        flash("Resume updated successfully", "success")
        return redirect(url_for('view_resume_endpoint', id=resume.id))

    return render_template("edit_resume.html", resume=resume)


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
    if 'chat_history' not in session:
        session['chat_history'] = []

    advice = None
    if request.method == 'POST':
        question = request.form['question'].strip()
        if not question:
            return "Please ask a valid question.", 400

        # System prompt tells the AI to act as if it remembers the conversation
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert career advisor. "
                    "You always remember the user's previous questions and your previous answers in this conversation, "
                    "and you refer back to them naturally when helpful. "
                    "Do not mention memory limitations. "
                    "If the user asks about remembering, respond as if you remember everything in this chat."
                )
            }
        ]
        for turn in session['chat_history']:
            messages.append({"role": "user", "content": turn['question']})
            messages.append({"role": "assistant", "content": turn['answer']})
        messages.append({"role": "user", "content": question})

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1200
            )
            advice = response['choices'][0]['message']['content'].strip()
            session['chat_history'].append({'question': question, 'answer': advice})
            session.modified = True
        except Exception as e:
            return f"An error occurred: {str(e)}"

    # Do NOT pass chat_history to the template
    return render_template("career_advice.html", advice=advice)



def strip_markdown_code_fences(text):
    # Remove all triple-backtick code blocks (with or without language label)
    return re.sub(r"``````", "", text)

@app.route("/cover-letter", methods=["GET", "POST"])
@login_required
def cover_letter():
    if request.method == "POST":
        resume_text = request.form.get("resume_text", "").strip()
        job_info = request.form.get("job_info", "").strip()

        if not resume_text or not job_info:
            flash("Please provide both resume text and job information", "error")
            return render_template("cover_letter.html", letter="", letter_markdown="")

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
            - Format the entire letter using Markdown syntax (headers, bold, lists, etc.)
            - Do NOT use code blocks or triple backticks anywhere in your output.
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
            letter_markdown = response.choices[0].message["content"].strip()
            print("AI raw output:", repr(letter_markdown))

            letter_markdown_clean = strip_markdown_code_fences(letter_markdown)
            print("Cleaned markdown:", repr(letter_markdown_clean))

            letter_html = markdown.markdown(letter_markdown_clean)

            return render_template(
                "cover_letter.html",
                letter=letter_html,
                letter_markdown=letter_markdown_clean
            )

        except Exception as e:
            print("AI error:", e)
            flash(f"Error generating cover letter: {str(e)}", "error")
            return render_template("cover_letter.html", letter="", letter_markdown="")

    return render_template("cover_letter.html", letter="", letter_markdown="")


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.route("/my_resumes")
@login_required
def my_resumes():
    resumes = Resume.query.filter_by(user_id=current_user.id).all()
    return render_template("resumes_list.html", resumes=resumes)




@app.route('/download-template-pdf', methods=['POST'])
@login_required
def download_template_pdf():
    data = request.get_json()
    template_type = data['template']
    color_option = data['colorOption']
    content = data['content']
    doc_type = data.get('docType', 'resume')
    user_info = data.get('userInfo', {})

    # Convert Markdown to HTML if needed
    import markdown
    if data.get('isMarkdown', False):
        content_html = markdown.markdown(content)
    else:
        content_html = content

    # Select template HTML based on template_type, color_option, and doc_type
    if doc_type == "cover_letter":
        # Use cover letter template styles
        template_html = """
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2, h3 { color: #d97706; }
                .header { margin-bottom: 32px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ name }}</h1>
                <div>{{ email }}</div>
            </div>
            <div class="letter">
                {{ content_html | safe }}
            </div>
        </body>
        </html>
        """
    else:
        # Use resume template styles as before, with template_type/color_option logic
        if template_type == "professional":
            template_html = """
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h1 { color: #2563eb; }
                    .section { margin-bottom: 24px; }
                    .header { background: #2563eb; color: white; padding: 12px; border-radius: 8px; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{{ name }}</h1>
                    <div>{{ email }}</div>
                </div>
                <div class="section">
                    {{ content_html | safe }}
                </div>
            </body>
            </html>
            """
        elif template_type == "modern":
            template_html = """
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    body { font-family: 'Segoe UI', sans-serif; margin: 40px; background: #e0f2fe; }
                    h1 { color: #059669; }
                    .section { margin-bottom: 24px; }
                    .header { background: #059669; color: white; padding: 12px; border-radius: 8px; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{{ name }}</h1>
                    <div>{{ email }}</div>
                </div>
                <div class="section">
                    {{ content_html | safe }}
                </div>
            </body>
            </html>
            """
        else:  # creative or fallback
            template_html = """
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    body { font-family: 'Georgia', serif; margin: 40px; background: #f3e8ff; }
                    h1 { color: #a21caf; }
                    .section { margin-bottom: 24px; }
                    .header { background: #a21caf; color: white; padding: 12px; border-radius: 8px; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{{ name }}</h1>
                    <div>{{ email }}</div>
                </div>
                <div class="section">
                    {{ content_html | safe }}
                </div>
            </body>
            </html>
            """

    rendered_html = render_template_string(
        template_html,
        name=user_info.get('name', 'Your Name'),
        email=user_info.get('email', 'your@email.com'),
        content_html=content_html
    )

    from weasyprint import HTML
    pdf = HTML(string=rendered_html).write_pdf()
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={doc_type.capitalize()}_{template_type}.pdf'
    return response


@app.route('/download-template-resume', methods=['POST'])
@login_required
def download_template_resume():
    data = request.get_json()
    template_type = data.get('template', 'professional')
    color_option = data.get('colorOption', 'color')
    content = data.get('content', '')
    user_info = data.get('userInfo', {})
    is_markdown = data.get('isMarkdown', False)

    # Convert Markdown to HTML if needed
    content_html = markdown.markdown(content) if is_markdown else content

    # Select template HTML based on template_type and color_option
    if template_type == "professional":
        template_html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2563eb; }
                .section { margin-bottom: 24px; }
                .header { background: #2563eb; color: white; padding: 12px; border-radius: 8px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ name }}</h1>
                <div>{{ email }}</div>
            </div>
            <div class="section">
                {{ content_html | safe }}
            </div>
        </body>
        </html>
        """
    elif template_type == "modern":
        template_html = """
        <html>
        <head>
            <style>
                body { font-family: 'Segoe UI', sans-serif; margin: 40px; background: #e0f2fe; }
                h1 { color: #059669; }
                .section { margin-bottom: 24px; }
                .header { background: #059669; color: white; padding: 12px; border-radius: 8px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ name }}</h1>
                <div>{{ email }}</div>
            </div>
            <div class="section">
                {{ content_html | safe }}
            </div>
        </body>
        </html>
        """
    else:  # creative or fallback
        template_html = """
        <html>
        <head>
            <style>
                body { font-family: 'Georgia', serif; margin: 40px; background: #f3e8ff; }
                h1 { color: #a21caf; }
                .section { margin-bottom: 24px; }
                .header { background: #a21caf; color: white; padding: 12px; border-radius: 8px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ name }}</h1>
                <div>{{ email }}</div>
            </div>
            <div class="section">
                {{ content_html | safe }}
            </div>
        </body>
        </html>
        """

    rendered_html = render_template_string(
        template_html,
        name=user_info.get('name', 'Your Name'),
        email=user_info.get('email', 'your@email.com'),
        content_html=content_html
    )

    pdf = HTML(string=rendered_html).write_pdf()
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=Resume_{template_type}.pdf'
    return response


@app.context_processor
def inject_now():
    return {'now': datetime.now}

@app.errorhandler(429)
def too_many_requests(e):
    return render_template("429.html"), 429

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5001)