# Career Coach Web App

**AI-powered Career Coach Web Application for resume feedback, cover letter generation, and career advice.**

---

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Description

Career Coach is a modern web application that leverages AI to help users improve their resumes, generate professional cover letters, and receive personalized career advice. It supports resume uploads, AI-driven enhancements, and provides a user-friendly interface for managing career documents.

---

## Features

- **AI-powered resume analysis and enhancement**
- **Professional cover letter generation with AI**
- **Multiple resume templates with photo integration**
- **Live typing animation for AI-generated content**
- **User authentication and dashboard**
- **Resume upload and management**
- **Career advice and interview tips**

---

## Installation

1. **Clone the repository: https://github.com/Dennis2y/Career-Coach-Web-App.git


2. **Create and activate a virtual environment:**


3. **Install dependencies:**



4. **Set up environment variables:**
- Create a `.env` file in the project root.
- Add your OpenAI API key and other configurations.

5. **Run database migrations:**


6. **Start the Flask development server:**



7. **Open your browser and navigate to `http://localhost:5001`.**

---

## Usage

- **Register or log in to your account.**
- **Upload your resume for AI enhancement.**
- **Generate cover letters tailored to job descriptions.**
- **Choose from multiple resume templates.**
- **Access career advice and interview tips.**

---

## Project Structure

Career-Coach-Web-App

├── migrations
├── static/
│ └── images/
│ ├── robot.png
│ └── logo.png
├── templates/
│ ├── 404.html
│ ├── 429.html
│ ├── admin_dashboard.html
│ ├── base.html
│ ├── career_advice.html
│ ├── cover_letter.html
│ ├── dashboard.html
│ ├── enhanced_resume_view.html
│ ├── home.html
│ ├── interview_tips.html
│ ├── login.html
│ ├── register.html
│ ├── resume_view.html
│ ├── resumes_list.html
│ ├── upload.html
│ └── uploads/ # Uploaded files
├── .env 
├── render.yaml 
├── app.py 
├── models.py 
├── requirements.txt 
└── README.md 


## Technologies

- **Python 3.10+**
- **Flask**
- **Flask-Login**
- **SQLAlchemy**
- **OpenAI API**
- **Tailwind CSS**


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

