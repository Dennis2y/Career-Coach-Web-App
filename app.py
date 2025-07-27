from flask import Flask, render_template, session, request, redirect, url_for, abort, flash, make_response, render_template_string, send_file,get_flashed_messages
from flask_migrate import Migrate
from flask_login import LoginManager, login_required, login_user, current_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from dotenv import load_dotenv
import os
from openai import OpenAI
import pdfkit
from PyPDF2 import PdfReader
from striprtf.striprtf import rtf_to_text
from models import db, Conversation
import markdown
import re
from bs4 import BeautifulSoup

from markupsafe import Markup
from weasyprint import HTML
import html
import PyPDF2
import io
import docx
from flask import make_response, render_template
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx2pdf import convert
from markupsafe import Markup

import random


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
db.init_app(app)
migrate = Migrate(app, db)

# Initialize LoginManager (SINGLE initialization)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def allowed_photo(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def make_all_links_clickable(text):
    # Convert Markdown links [Label](http(s)://...) including http and https
    text = re.sub(
        r'\[([^\]]+)\]\(((?:https?://)[^\s)]+)\)',
        r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
        text
    )

    # Convert raw http/https URLs not already in anchors
    text = re.sub(
        r'(?<![">=])((https?://[^\s<>"\'\)]+))',
        r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>',
        text
    )
    return text


def add_hyperlink(paragraph, text, url):
    """
    Adds a clickable hyperlink to a paragraph.
    `text` is the clickable text shown; `url` is the actual link.
    """
    part = paragraph.part
    r_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")

    # Style the hyperlink - blue, underlined
    rStyle = OxmlElement("w:rStyle")
    rStyle.set(qn("w:val"), "Hyperlink")
    rPr.append(rStyle)

    new_run.append(rPr)

    text_elem = OxmlElement("w:t")
    text_elem.text = text
    new_run.append(text_elem)
    hyperlink.append(new_run)

    paragraph._p.append(hyperlink)
    return hyperlink


# Links dictionary
links = {
    "GitHub": "https://github.com/Dennis2y",
    "LinkedIn": "https://linkedin.com/in/dennis-charles53/",
    "Portfolio": "https://dennischarles.dev",
    "Resume": "https://example.com/dennis_resume.pdf"
}

doc = Document()
doc.add_heading("Credentials", level=1)

for label, url in links.items():
    para = doc.add_paragraph(f"{label}: ")
    add_hyperlink(para, label, url)  # Display label as clickable text

docx_filename = "enhanced_resume.docx"
pdf_filename = "enhanced_resume.pdf"

doc.save(docx_filename)

# Convert to PDF (requires MS Word installed)
convert(docx_filename, pdf_filename)



def extract_text(file_path, ext):
    if ext == 'pdf':
        text = ""
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif ext == 'docx':
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    elif ext == 'rtf':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = rtf_to_text(f.read())
    else:
        raise ValueError("Unsupported file type")

    clean_text = text.replace('\x00', '')

    return make_all_links_clickable(clean_text)


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_resume():
    if request.method == "POST":
        if 'resume' not in request.files:
            return "No file part", 400
        file = request.files['resume']
        if file.filename == '':
            return "No selected file", 400
        upload_folder = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)
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
        selected_template = request.form.get("template", "professional")
        selected_model = request.form.get("model", "gpt-4o-mini")
        if selected_model not in ['gpt-4o-mini', 'gpt-4.1-mini']:
            selected_model = "gpt-4o-mini"  # fallback default
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            ext = filename.rsplit('.', 1)[1].lower()
            path = os.path.join(upload_folder, filename)
            try:
                file.save(path)
                content = extract_text(path, ext)
                professional_resume = generate_professional_resume(content, model=selected_model)
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
                if photo_path and os.path.exists(photo_path):
                    os.remove(photo_path)
                if path and os.path.exists(path):
                    os.remove(path)
                return f"Error processing file: {str(e)}", 500
        return "Invalid file type. Supported formats: PDF, DOCX, TXT, RTF", 400
    return render_template("upload.html")

def clean_letter_markdown(letter_markdown, user_email):
    lines = letter_markdown.split('\n')
    cleaned = []
    email_seen = False
    for line in lines:
        if line.strip() == user_email:
            if not email_seen:
                cleaned.append(line)
                email_seen = True
            else:
                continue
        else:
            cleaned.append(line)
    return '\n'.join(cleaned)

@app.route("/resumes")
@login_required
def resume_list():
    resumes = Resume.query.filter_by(user_id=current_user.id).order_by(Resume.created_at.desc()).all()
    return render_template("resumes_list.html", resumes=resumes)

def generate_professional_resume(original_content, model="gpt-4o-mini"):
    try:
        prompt = f"""
                  Here's my resune , please enhance it 
         {original_content}
        """
        response = client.chat.completions.create(
            model=model,
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
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ AI Enhancement Error: {str(e)}"

@app.route("/resume/enhanced/<int:id>")
@login_required
def view_enhanced_resume(id):
    resume = Resume.query.get_or_404(id)
    if resume.user_id != current_user.id:
        abort(403)
    html_content = make_all_links_clickable(resume.text_content)
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



@app.route("/resume/<int:id>/delete", methods=["POST"], endpoint='delete_resume_endpoint')
@login_required
def delete_resume(id):
    resume = Resume.query.get_or_404(id)
    if resume.user_id != current_user.id:
        abort(403)
    db.session.delete(resume)
    db.session.commit()
    return redirect(url_for('my_resumes'))

@app.route("/resume/<int:id>", endpoint='view_resume_endpoint')
@login_required
def view_resume(id):
    resume = Resume.query.get_or_404(id)
    if resume.user_id != current_user.id:
        abort(403)
    return render_template("resume_view.html", resume=resume)

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

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        if password != confirm_password:
            return render_template("register.html", error="Passwords do not match")
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template("register.html", error="Email is already registered")
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
    return redirect(url_for('home'))


def ensure_period(text: str) -> str:
    return text.strip() + ("" if text.endswith((".", "!", "?")) else ".")

    def is_meaningful(content):
        content = content.strip().lower()
        return (
            content and
            not content.startswith(("warning", "info", "thread(")) and
            content not in {"you're welcome", "thanks", "thank you", "ok", "okay", "sure"}
        )

    recent_turns = [t for t in previous_turns if is_meaningful(t.content)]
    recent_turns = recent_turns[-max_turns:]

    if not recent_turns:
        no_meaningful_msgs = {
            "English": "You haven’t discussed anything meaningful yet.",
            "German": "Sie haben noch keine wichtigen Themen besprochen.",
        }
        return no_meaningful_msgs.get(language, no_meaningful_msgs["English"])

    try:
        system_content = (
            f"You are a professional assistant. Summarize the following conversation in 1–2 well-written sentences in {language}. "
            "Avoid repeating phrases like 'Yes' or 'I remember'. Focus only on the key points discussed. Use a helpful, friendly tone."
        )
        if markdown:
            system_content += " Format the summary using Markdown."

        summarization_prompt = [{"role": "system", "content": system_content}] + [
            {"role": turn.role, "content": turn.content}
            for turn in recent_turns
        ]

        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=summarization_prompt,
            max_tokens=150
        )

        summary = result.choices[0].message.content.strip()

        # Cleaning and punctuation enforcement, language-dependent
        if language.lower() == "english":
            summary = re.sub(r"^(yes[,!.\s]*)?", "", summary, flags=re.IGNORECASE).strip()
            summary = re.sub(r"^(i remember[,!.\s]*)?", "", summary, flags=re.IGNORECASE).strip()
            if summary:
                summary = summary[0].upper() + summary[1:]
                if summary[-1] not in {".", "!", "?"}:
                    summary += "."
        else:
            # For other languages, optionally implement punctuation enforcement
            if summary and summary[-1] not in {".", "!", "?"}:
                summary += "."

        return summary

    except Exception as e:
        print("Summarization error:", e)
        fallback_msgs = {
            "English": "Sorry, I couldn't summarize your past discussions.",
            "German": "Entschuldigung, ich konnte Ihre bisherigen Gespräche nicht zusammenfassen.",
        }
        return fallback_msgs.get(language, fallback_msgs["English"])


@app.route('/career-advice', methods=['GET', 'POST'])
@login_required
def career_advice():
    if request.method == 'POST':
        question = request.form['question'].strip()
        selected_model = request.form.get("model", "gpt-4o-mini")

        if selected_model not in ['gpt-4o-mini', 'gpt-4.1-mini']:
            flash("Invalid model selected.", "error")
            return redirect(url_for('career_advice'))

        if not question:
            flash("Please ask a valid question.", "error")
            return redirect(url_for('career_advice'))

        # Handle uploaded file if any
        file_text = ""
        attachment = request.files.get('attachment')
        if attachment and attachment.filename != "":
            filename = attachment.filename.lower()
            try:
                content_bytes = attachment.read()

                if filename.endswith('.txt'):
                    # Simple plain text file
                    file_text = content_bytes.decode('utf-8', errors='ignore')

                elif filename.endswith('.pdf'):
                    # Extract text from PDF using PyPDF2
                    reader = PyPDF2.PdfReader(io.BytesIO(content_bytes))
                    texts = []
                    for page in reader.pages:
                        texts.append(page.extract_text() or "")
                    file_text = "\n".join(texts).strip()

                elif filename.endswith('.docx'):
                    # Extract text from DOCX using python-docx
                    doc = docx.Document(io.BytesIO(content_bytes))
                    texts = [para.text for para in doc.paragraphs]
                    file_text = "\n".join(texts).strip()

                else:
                    flash("Unsupported file type. Please upload TXT, PDF, or DOCX files.", "warning")

            except Exception as e:
                flash(f"Failed to process attachment: {str(e)}", "error")
                return redirect(url_for('career_advice'))

        # Build messages with conversation history
        history = Conversation.query.filter_by(user_id=current_user.id).order_by(Conversation.timestamp).all()

        messages = [{
            "role": "system",
            "content": (
                "You are an expert career advisor. "
                "You always remember the user's previous questions and your previous answers in this conversation, "
                "and you refer back to them naturally when helpful. "
                "Do not mention memory limitations. "
                "If the user asks about remembering, respond as if you remember everything in this chat."
            )
        }]

        last_messages = history[-20:]
        for turn in last_messages:
            messages.append({"role": turn.role, "content": turn.content})

        # Append the user question + optional file content
        user_content = question
        if file_text:
            user_content += "\n\nAdditional context from the uploaded file:\n" + file_text

        messages.append({"role": "user", "content": user_content})

        try:
            response = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                max_tokens=1200
            )
            advice = response.choices[0].message.content.strip()

            db.session.add(Conversation(user_id=current_user.id, role='user', content=user_content, timestamp=datetime.now()))
            db.session.add(Conversation(user_id=current_user.id, role='assistant', content=advice, timestamp=datetime.now()))
            db.session.commit()

        except Exception as e:
            flash(f"An error occurred: {str(e)}", "error")
            return redirect(url_for('career_advice'))

        # Show response directly in the same request
        history = Conversation.query.filter_by(user_id=current_user.id).order_by(Conversation.timestamp).all()

        return render_template("career_advice.html",
                               advice=advice,
                               history=history,
                               selected_model=selected_model,
                               question="")  # Clear input

    # GET: first page load
    return render_template("career_advice.html",
                           advice=None,
                           history=Conversation.query.filter_by(user_id=current_user.id).order_by(Conversation.timestamp).all(),
                           selected_model="gpt-4o-mini",
                           question="")



def strip_markdown_code_fences(text):
    return re.sub(r"``````", "", text)

@app.route("/cover-letter", methods=["GET", "POST"])
@login_required
def cover_letter():
    selected_model = "gpt-4o-mini"  # default

    if request.method == "POST":
        selected_model = request.form.get("model", "gpt-4o-mini")
        if selected_model not in ['gpt-4o-mini', 'gpt-4.1-mini']:
            selected_model = "gpt-4o-mini"

        address = request.form.get("address", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        resume_text = request.form.get("resume_text", "").strip()
        job_info = request.form.get("job_info", "").strip()
        date_today = datetime.now().strftime("%d %B %Y")
        if not address or not email or not resume_text or not job_info:
            flash("Please provide your address, email, resume text, and job information", "error")
            return render_template("cover_letter.html", letter="", letter_markdown="", address=address, email=email, phone=phone, selected_model=selected_model)
        try:
            header_block = f"""{address}
Email: {email}
Phone: {phone}
Date: {date_today}
"""
            prompt = f"""
Write a professional cover letter with these sections:
1. Header: Use this block exactly as provided:
{header_block}
2. Date and hiring manager details (if not already in header)
3. Salutation
4. Opening paragraph: Mention position and enthusiasm
5. Body: 2-3 paragraphs matching skills to job requirements
6. Closing paragraph: Call to action
7. Professional closing (sign off with the applicant's name at the end only)

Job Details:
{job_info}

Applicant Resume Summary:
{resume_text}

Important:
- Do NOT include the applicant's name anywhere except as the signature at the end.
- Use formal business letter format
- Keep under 400 words
- Include quantifiable achievements
- Address to "Hiring Manager" if no name available
- Format the entire letter using Markdown syntax (headers, bold, lists, etc.)
- Do NOT use code blocks or triple backticks anywhere in your output.
- The applicant's name should only appear as the signature at the end of the letter, not at the top (except as part of the address block).
"""
            response = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": "You are a professional resume writer"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1200
            )
            letter_markdown = response.choices[0].message.content.strip()
            letter_markdown_clean = strip_markdown_code_fences(letter_markdown)
            letter_markdown_clean = clean_letter_markdown(letter_markdown_clean, email)
            letter_html = markdown.markdown(letter_markdown_clean)
            return render_template(
                "cover_letter.html",
                letter=letter_html,
                letter_markdown=letter_markdown_clean,
                address=address,
                email=email,
                phone=phone,
                selected_model=selected_model
            )
        except Exception as e:
            flash(f"Error generating cover letter: {str(e)}", "error")
            return render_template("cover_letter.html", letter="", letter_markdown="", address=address, email=email, phone=phone, selected_model=selected_model)
    # On GET, render empty form with selected_model default
    return render_template("cover_letter.html", letter="", letter_markdown="", address="", email=current_user.email, phone="", selected_model=selected_model)


def remove_name_from_header(markdown_text, user_name):
    lines = markdown_text.strip().splitlines()
    # Remove leading name if it matches and is not in the closing
    if lines and user_name.lower() in lines[0].lower():
        lines = lines[1:]
    return "\n".join(lines)


def clean_letter_markdown(letter_markdown, user_email):
    email_pattern = re.compile(re.escape(user_email.strip()), re.IGNORECASE)
    lines = letter_markdown.split('\n')
    cleaned = []
    email_seen = False
    for line in lines:
        if email_pattern.search(line.strip()):
            if not email_seen:
                cleaned.append(line)
                email_seen = True
        else:
            cleaned.append(line)
    return '\n'.join(cleaned)

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.route("/my_resumes")
@login_required
def my_resumes():
    resumes = Resume.query.filter_by(user_id=current_user.id).all()
    return render_template("resumes_list.html", resumes=resumes)


def clean_letter_markdown(content, email):
    """
    Clean the letter markdown content as needed before rendering or PDF generation.
    For example, this function can:
      - Remove duplicate email lines
      - Sanitize inputs
      - Strip unwanted characters or whitespace
      - Normalize formatting
    
    Currently, this is a pass-through function returning content unchanged.
    """

    # Example placeholder code:
    # lines = content.split('\n')
    # cleaned_lines = []
    # email_seen = False
    # for line in lines:
    #     if line.strip() == email:
    #         if not email_seen:
    #             cleaned_lines.append(line)
    #             email_seen = True
    #     else:
    #         cleaned_lines.append(line)
    # return '\n'.join(cleaned_lines)

    return content



# --- Utility functions ---

def replace_placeholder_links(text, link_mapping):
    for label, url in link_mapping.items():
        pattern = rf'\[{re.escape(label)}\]\((#|)\)'
        replacement = f'[{label}]({url})'
        text = re.sub(pattern, replacement, text)
    return text

def rewrite_markdown_links_to_matching_text(md_text):
    pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
    return re.sub(pattern, lambda m: f"[{m.group(2)}]({m.group(2)})", md_text)

def make_all_links_clickable(text):
    # Markdown links → HTML anchors
    text = re.sub(
        r'\[([^\]]+)\]\(((?:https?://)[^\s)]+)\)',
        r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
        text
    )
    # Raw URLs → HTML anchors
    text = re.sub(
        r'(?<![">=])((https?://[^\s<>"\'\)]+))',
        r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>',
        text
    )
    return text




@app.route('/download-template-pdf', methods=['POST'])
@login_required
def download_template_pdf():
    data = request.get_json()

    template_type = data.get('template', 'professional')
    doc_type = data.get('docType', 'resume')
    content = data.get('content', '')
    is_markdown = data.get('isMarkdown', False)

    # Defining the  real URLs for labels
    link_map = {
        "LinkedIn": "https://www.linkedin.com/in/dennis-charles53/",
        "GitHub": "https://github.com/Dennis2y",
        "Movie Project": "https://github.com/Dennis2y/Dennis2y-Movie-Project-SQL-HTML-API",
        "Best-Buy-2.0": "https://github.com/Dennis2y/Best-Buy-2.0",
        "Credentials": "https://drive.google.com/file/d/1Hav3gSC4TMhOz091ZosUlcVvXJOTwRRi/view?pli=1",
        "Portfolio": "https://dennischarles.dev",
        "Resume": "https://example.com/dennis_resume.pdf",
        "View Credentials": "https://drive.google.com/file/d/1Hav3gSC4TMhOz091ZosUlcVvXJOTwRRi/view?pli=1",
        "Link": "https://github.com/Dennis2y/Dennis2y-Movie-Project-SQL-HTML-API"
    }

    # Replace [Label](#) in markdown content with actual URLs from link_map
    def replace_placeholder_links(md_text, links):
        # regex matches markdown links like [Label](#)
        pattern = re.compile(r'\[([^\]]+)\]\(#\)')
        def replacer(match):
            label = match.group(1)
            url = links.get(label, '#')
            return f'[{label}]({url})'
        return pattern.sub(replacer, md_text)

    def rewrite_markdown_links_to_matching_text(md_text):
        # no special rewrite needed here, keep it as-is
        return md_text

    # Step 1: Replace placeholder links
    content = replace_placeholder_links(content, link_map)

    # Step 2: Markdown to HTML + clickable link conversion
    if is_markdown:
        content = rewrite_markdown_links_to_matching_text(content)
        content_html = markdown.markdown(content, extensions=['extra', 'sane_lists'])
    else:
        content_html = content

    # If your make_all_links_clickable does other fixes, keep it; else no-op:
    def make_all_links_clickable(html):
        return html
    content_html = make_all_links_clickable(content_html)

    # Add target="_blank" to all <a> tags so PDF viewers recognize links
    def add_target_blank_to_links(html):
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all('a', href=True):
            a['target'] = '_blank'
        return str(soup)

    content_html = add_target_blank_to_links(content_html)

    # Step 3: Choose styling template
    anchor_css = "a { color: #2563eb; text-decoration: underline; }"

    base_styles = {
        "cover_letter": "body { font-family: Arial, sans-serif; margin: 40px; } h1,h2,h3 { color: #d97706; }",
        "professional": "body { font-family: Arial, sans-serif; margin: 40px; } h1 { color: #2563eb; }",
        "modern": "body { font-family: 'Segoe UI', sans-serif; margin: 40px; background: #e0f2fe; } h1 { color: #059669; }",
        "creative": "body { font-family: 'Georgia', serif; margin: 40px; background: #f3e8ff; } h1 { color: #a21caf; }"
    }

    template_style = base_styles.get(template_type, base_styles["professional"])
    content_class = "letter" if doc_type == "cover_letter" else "section"

    template_html = f"""
    <html><head><meta charset="utf-8"><style>
    {template_style}
    {anchor_css}
    </style></head><body>
    <div class="{content_class}">{{{{ content_html | safe }}}}</div>
    </body></html>
    """

    # Step 4: Render and generate PDF
    rendered_html = render_template_string(template_html, content_html=Markup(content_html))
    pdf = HTML(string=rendered_html).write_pdf()

    # Step 5: Return PDF response
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={doc_type.capitalize()}_{template_type}.pdf'
    return response




    def clean_letter_markdown(letter_markdown, user_email):
        lines = letter_markdown.split('\n')
        cleaned_lines = []
        email_seen = False
        for line in lines:
            if line.strip() == user_email:
               if not email_seen:
                    cleaned_lines.append(line)
                    email_seen = True
            else:
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    content = clean_letter_markdown(content, user_email)

    def rewrite_markdown_links_to_matching_text(md_text):
        pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
        def replacer(match):
            url = match.group(2)
            return f'[{url}]({url})'
        return re.sub(pattern, replacer, md_text)

    if is_markdown:
        content = rewrite_markdown_links_to_matching_text(content)
        content_html = markdown.markdown(content)
        content_html = make_all_links_clickable(content_html)
    else:
        content_html = make_all_links_clickable(content)

    anchor_css = """
        a {
            color: #2563eb;
            text-decoration: underline;
        }
    """

    if doc_type == "cover_letter":
        template_html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #d97706; }}
                .header {{ margin-bottom: 32px; }}
                {anchor_css}
            </style>
        </head>
        <body>
            <div class="header"></div>
            <div class="letter">
                {{{{ content_html | safe }}}}
            </div>
        </body>
        </html>
        """
    elif template_type == "professional":
        template_html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2563eb; }}
                .section {{ margin-bottom: 24px; }}
                .header {{ background: #2563eb; color: white; padding: 12px; border-radius: 8px; }}
                {anchor_css}
            </style>
        </head>
        <body>
            <div class="header"></div>
            <div class="section">
                {{{{ content_html | safe }}}}
            </div>
        </body>
        </html>
        """
    elif template_type == "modern":
        template_html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; margin: 40px; background: #e0f2fe; }}
                h1 {{ color: #059669; }}
                .section {{ margin-bottom: 24px; }}
                .header {{ background: #059669; color: white; padding: 12px; border-radius: 8px; }}
                {anchor_css}
            </style>
        </head>
        <body>
            <div class="header"></div>
            <div class="section">
                {{{{ content_html | safe }}}}
            </div>
        </body>
        </html>
        """
    else:
        template_html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Georgia', serif; margin: 40px; background: #f3e8ff; }}
                h1 {{ color: #a21caf; }}
                .section {{ margin-bottom: 24px; }}
                .header {{ background: #a21caf; color: white; padding: 12px; border-radius: 8px; }}
                {anchor_css}
            </style>
        </head>
        <body>
            <div class="header"></div>
            <div class="section">
                {{{{ content_html | safe }}}}
            </div>
        </body>
        </html>
        """

    rendered_html = render_template_string(template_html, content_html=Markup(content_html))

    pdf = HTML(string=rendered_html, base_url=None).write_pdf()

    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={doc_type.capitalize()}_{template_type}.pdf'

    return response



@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # handle form submission here, e.g. save data, send email, etc.
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        flash('Thank you for contacting us!', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/privacy-policy')
def privacy_policy():
    return render_template('privacy_policy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')


def remove_header_emails(text):
    email_pattern = re.compile(r'^[\s>]*[\w\.-]+@[\w\.-]+\.\w+[\s<]*$', re.IGNORECASE)
    lines = text.splitlines()
    new_lines = []
    email_found = False
    for line in lines:
        if not new_lines and email_pattern.match(line.strip()):
            continue
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


@app.route('/download_template_pdf_alt', methods=['POST'], endpoint='download_pdf_two')
@login_required
def download_template_pdf_alt():
    data = request.json
    content = data['content']  # Markdown or plain text

    # Use make_all_links_clickable instead of linkify_text:
    html_ready = make_all_links_clickable(content)

    html_rendered = render_template('resume_pdf_template.html', content=html_ready)
    pdf = pdfkit.from_string(html_rendered, False, options={'enable-local-file-access': None})

    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=resume.pdf'
    return response


@app.route('/download-resume', methods=['POST'])
def download_resume():
    return send_file(
        "resume.pdf",
        as_attachment=True,
        download_name="resume.pdf",
        mimetype="application/pdf"
    )


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
