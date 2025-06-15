from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, login_required, logout_user, current_user
from app import db
from app.models import User, Resume
from app.forms import RegisterForm, LoginForm, ResumeForm
from werkzeug.security import generate_password_hash, check_password_hash
from app.ai_engine import generate_resume_updates

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('home.html')

@main.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('main.login'))
    return render_template('register.html', form=form)

@main.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('main.dashboard'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html', form=form)

@main.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.home'))

@main.route('/dashboard')
@login_required
def dashboard():
    resume = Resume.query.filter_by(user_id=current_user.id).first()
    return render_template('dashboard.html', resume=resume)

@main.route('/resume', methods=['GET', 'POST'])
@login_required
def resume():
    resume = Resume.query.filter_by(user_id=current_user.id).first()
    form = ResumeForm(obj=resume)
    if form.validate_on_submit():
        if resume:
            resume.content = form.content.data
        else:
            resume = Resume(content=form.content.data, user_id=current_user.id)
            db.session.add(resume)
        db.session.commit()
        flash('Resume saved!', 'success')
        return redirect(url_for('main.dashboard'))
    return render_template('resume.html', form=form)

@main.route('/resume/ai-update')
@login_required
def ai_update_resume():
    resume = Resume.query.filter_by(user_id=current_user.id).first()
    if not resume:
        flash("No resume found!", "warning")
        return redirect(url_for('main.resume'))

    updated_content = generate_resume_updates(resume.content)
    resume.content = updated_content
    db.session.commit()
    flash("Resume updated using AI!", "success")
    return redirect(url_for('main.resume'))
