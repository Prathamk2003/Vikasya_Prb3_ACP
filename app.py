# app.py - Main Flask Application

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask_migrate import Migrate

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///academic_platform.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

migrate = Migrate(app, db)
# After this, use `flask db init`, `flask db migrate`, and `flask db upgrade` for migrations

# Database Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    user_type = db.Column(db.String(20), nullable=False)  # 'student' or 'educator'
    institution = db.Column(db.String(120))
    bio = db.Column(db.Text)
    profile_pic = db.Column(db.String(120), default='default_profile.png')
    date_joined = db.Column(db.DateTime, default=datetime.utcnow)
    domain = db.Column(db.String(120), nullable=True)  # For educators
    skills = db.Column(db.String(255), nullable=True)  # For educators, comma-separated
    designation = db.Column(db.String(120), nullable=True)  # For educators
    
    projects = db.relationship('Project', backref='author', lazy=True)
    queries = db.relationship('Query', backref='author', lazy=True)
    resources = db.relationship('Resource', backref='uploader', lazy=True)
    comments = db.relationship('Comment', backref='author', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    content = db.Column(db.Text)
    project_type = db.Column(db.String(50))
    subject = db.Column(db.String(50))
    date_posted = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    collaborators = db.Column(db.Text)  # Stored as JSON string of user IDs
    files = db.relationship('ProjectFile', backref='project', lazy=True)
    comments = db.relationship('Comment', backref='project', lazy=True)
    views = db.Column(db.Integer, default=0)
    likes = db.Column(db.Integer, default=0)
    scope = db.Column(db.String(10), default='public')  # 'public' or 'private'

class ProjectFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    path = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)

class Query(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    subject = db.Column(db.String(50))
    date_posted = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    answers = db.relationship('Answer', backref='query', lazy=True)
    views = db.Column(db.Integer, default=0)
    solved = db.Column(db.Boolean, default=False)

class Answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    date_posted = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query_id = db.Column(db.Integer, db.ForeignKey('query.id'), nullable=False)
    votes = db.Column(db.Integer, default=0)
    marked_correct = db.Column(db.Boolean, default=False)
    
    author = db.relationship('User', backref='answers')

class Resource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    resource_type = db.Column(db.String(50))  # Book, Paper, Video, etc.
    subject = db.Column(db.String(50))
    date_posted = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(120))
    path = db.Column(db.String(255))
    external_link = db.Column(db.String(255))
    downloads = db.Column(db.Integer, default=0)
    ratings = db.relationship('ResourceRating', backref='resource', lazy=True)

class ResourceRating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    rating = db.Column(db.Integer, nullable=False)  # 1-5 stars
    review = db.Column(db.Text)
    date_posted = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    resource_id = db.Column(db.Integer, db.ForeignKey('resource.id'), nullable=False)
    
    author = db.relationship('User', backref='ratings')

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    date_posted = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'))

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(50))  # comment, answer, collaboration, etc.
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    read = db.Column(db.Boolean, default=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    user = db.relationship('User', backref='notifications')

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_type = db.Column(db.String(50))  # project, query, resource
    content_id = db.Column(db.Integer, nullable=False)
    interaction_type = db.Column(db.String(50))  # view, download, like, comment
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class JoinRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, accepted, declined
    date_requested = db.Column(db.DateTime, default=datetime.utcnow)

# Flask-Login manager
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# AI Helper Functions
def preprocess_text(text):
    """Preprocess text for ML algorithms"""
    if not text:
        return ""
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
    return " ".join(filtered_tokens)

def find_similar_projects(project_description, top_n=5):
    """Find similar projects based on description using TF-IDF and cosine similarity"""
    projects = Project.query.all()
    if len(projects) < 2:
        return []
    
    # Prepare corpus
    corpus = [preprocess_text(p.description) for p in projects]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Get the TF-IDF for the query description
    query_tfidf = vectorizer.transform([preprocess_text(project_description)])
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_tfidf, tfidf_matrix)[0]
    
    # Get top_n similar projects (excluding the query project itself)
    similar_indices = similarities.argsort()[:-top_n-1:-1]
    similar_projects = [projects[i] for i in similar_indices if similarities[i] > 0.2]
    
    return similar_projects

def recommend_resources(user_id):
    """Recommend resources based on user's interaction history"""
    # Get user's viewed projects and queries
    user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
    
    # Extract subjects of interest
    subjects_of_interest = []
    for interaction in user_interactions:
        if interaction.content_type == 'project':
            project = Project.query.get(interaction.content_id)
            if project and project.subject:
                subjects_of_interest.append(project.subject)
        elif interaction.content_type == 'query':
            query = Query.query.get(interaction.content_id)
            if query and query.subject:
                subjects_of_interest.append(query.subject)
    
    # Count frequency of each subject
    subject_counts = {}
    for subject in subjects_of_interest:
        if subject in subject_counts:
            subject_counts[subject] += 1
        else:
            subject_counts[subject] = 1
    
    # Get top subjects
    top_subjects = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    top_subjects = [s[0] for s in top_subjects]
    
    # Recommend resources based on top subjects
    recommended_resources = []
    for subject in top_subjects:
        resources = Resource.query.filter_by(subject=subject).limit(3).all()
        recommended_resources.extend(resources)
    
    return recommended_resources[:5]  # Return top 5 recommended resources

def analyze_query(query_text):
    """Analyze a query to provide AI-assisted suggestions"""
    # This is a placeholder for more sophisticated analysis
    # In a real implementation, you might use NLP to categorize the query
    # and suggest potential answers or resources
    
    # Simple keyword matching for demo purposes
    keywords = {
        "python": ["programming", "coding", "algorithm"],
        "math": ["calculus", "algebra", "statistics"],
        "physics": ["mechanics", "thermodynamics", "quantum"],
        "chemistry": ["organic", "inorganic", "biochemistry"],
        "biology": ["genetics", "ecology", "anatomy"]
    }
    
    query_text = query_text.lower()
    suggestions = []
    
    for subject, related_terms in keywords.items():
        if subject in query_text or any(term in query_text for term in related_terms):
            suggestions.append(f"This appears to be related to {subject.title()}")
            # Find related resources
            resources = Resource.query.filter_by(subject=subject).limit(2).all()
            if resources:
                suggestions.append("You might find these resources helpful:")
                for resource in resources:
                    suggestions.append(f"- {resource.title}")
    
    return suggestions if suggestions else ["No specific suggestions available for this query."]

def generate_ai_insights(user_projects, user_queries, num_solved, industrial_people, subject_counts, type_counts):
    insights = []
    if len(user_projects) == 0:
        insights.append("Start by creating your first project to build your portfolio.")
    else:
        insights.append(f"Great job! You've created {len(user_projects)} project(s). Keep building your portfolio.")
    if user_queries:
        unsolved = sum(1 for q in user_queries if not q.solved)
        if unsolved > 0:
            insights.append(f"You have {unsolved} unsolved question(s). Consider revisiting them or seeking help from the community.")
    insights.append("Try commenting on projects or queries to increase your engagement.")
    if industrial_people:
        insights.append("Consider collaborating with industrial professionals for real world exposure.")
    if subject_counts:
        top_subject = max(subject_counts, key=subject_counts.get)
        insights.append(f"Explore more resources or projects in your top interest: {top_subject}.")
    if type_counts:
        top_type = max(type_counts, key=type_counts.get)
        insights.append(f"Create more projects to showcase your skills and interests in {top_type}.")
    return insights

# Routes
@app.route('/')
def index():
    """Home page showing recent projects, queries, and resources"""
    if current_user.is_authenticated:
        recent_projects = Project.query.order_by(Project.date_posted.desc()).limit(5).all()
        recent_queries = Query.query.order_by(Query.date_posted.desc()).limit(5).all()
        popular_resources = Resource.query.order_by(Resource.downloads.desc()).limit(5).all()
    else:
        recent_projects = []
        recent_queries = []
        popular_resources = []

    return render_template('index.html', 
                          projects=recent_projects, 
                          queries=recent_queries, 
                          resources=popular_resources)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        user_type = request.form.get('user_type')
        institution = request.form.get('institution')
        
        # Check if username or email already exists
        user_exists = User.query.filter((User.username == username) | (User.email == email)).first()
        if user_exists:
            flash('Username or email already exists.')
            return redirect(url_for('register'))
        
        # Create new user
        new_user = User(username=username, email=email, user_type=user_type, institution=institution)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid username or password.')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard showing personalized content"""
    user_projects = Project.query.filter_by(user_id=current_user.id).all()
    user_queries = Query.query.filter_by(user_id=current_user.id).all()
    user_resources = Resource.query.filter_by(user_id=current_user.id).all()
    
    # Get recommended resources using AI
    recommended_resources = recommend_resources(current_user.id)
    
    # Get notifications
    notifications = Notification.query.filter_by(user_id=current_user.id, read=False).all()
    
    # Get public projects by other users
    public_projects = Project.query.filter(Project.scope=='public', Project.user_id!=current_user.id).all()
    
    my_project_ids = [p.id for p in user_projects]
    pending_join_requests = JoinRequest.query.filter(JoinRequest.project_id.in_(my_project_ids), JoinRequest.status=='pending').all()
    # For each project, count collaborators
    project_collaborator_counts = {p.id: len(json.loads(p.collaborators or '[]')) for p in user_projects}
    public_project_collaborator_counts = {p.id: len(json.loads(p.collaborators or '[]')) for p in public_projects}
    
    return render_template('dashboard.html',
                          projects=user_projects,
                          queries=user_queries,
                          resources=user_resources,
                          recommended_resources=recommended_resources,
                          notifications=notifications,
                          public_projects=public_projects,
                          pending_join_requests=pending_join_requests,
                          project_collaborator_counts=project_collaborator_counts,
                          public_project_collaborator_counts=public_project_collaborator_counts)

@app.route('/profile/<username>')
def profile(username):
    """View user profile"""
    user = User.query.filter_by(username=username).first_or_404()
    projects = Project.query.filter_by(user_id=user.id).all()
    queries = Query.query.filter_by(user_id=user.id).all()
    resources = Resource.query.filter_by(user_id=user.id).all()

    students_grouped_by_subject = {}
    students_collaborated = []
    students_list = []
    filter_subject = None
    filter_institution = None
    filter_name = None
    analytics = {}
    chart_data = {}
    if user.user_type == 'educator':
        # Filtering options from query params
        filter_subject = request.args.get('filter_subject')
        filter_institution = request.args.get('filter_institution')
        filter_name = request.args.get('filter_name')

        # Group students by subject for subjects the educator has projects in
        educator_subjects = {p.subject for p in projects if p.subject}
        for subject in educator_subjects:
            students = User.query.filter_by(user_type='student').join(Project, Project.user_id == User.id).filter(Project.subject == subject)
            if filter_institution:
                students = students.filter(User.institution.ilike(f"%{filter_institution}%"))
            if filter_name:
                students = students.filter(User.username.ilike(f"%{filter_name}%"))
            students = students.all()
            students_grouped_by_subject[subject] = students
        # Students who have taken this educator as collaborator
        all_projects = Project.query.all()
        for project in all_projects:
            if project.collaborators:
                try:
                    collaborators = json.loads(project.collaborators)
                except Exception:
                    collaborators = []
                if user.id in collaborators:
                    student = User.query.get(project.user_id)
                    if student and student.user_type == 'student' and student not in students_collaborated:
                        if (not filter_subject or project.subject == filter_subject) and \
                           (not filter_institution or (student.institution and filter_institution.lower() in student.institution.lower())) and \
                           (not filter_name or (student.username and filter_name.lower() in student.username.lower())):
                            students_collaborated.append(student)
        # All students (filtered)
        students_query = User.query.filter_by(user_type='student')
        if filter_institution:
            students_query = students_query.filter(User.institution.ilike(f"%{filter_institution}%"))
        if filter_name:
            students_query = students_query.filter(User.username.ilike(f"%{filter_name}%"))
        students_list = students_query.all()

        # Analytics
        # 1. Number of unique students collaborated with (overall and per subject)
        unique_students = set(s.id for s in students_collaborated)
        analytics['unique_students_collaborated'] = len(unique_students)
        subject_collab_counts = {}
        for subject in educator_subjects:
            subject_collab_counts[subject] = len([s for s in students_collaborated if any(p.subject == subject for p in s.projects)])
        analytics['subject_collab_counts'] = subject_collab_counts
        # 2. Top students by number of collaborations
        student_collab_counts = {}
        for project in all_projects:
            if project.collaborators:
                try:
                    collaborators = json.loads(project.collaborators)
                except Exception:
                    collaborators = []
                if user.id in collaborators:
                    student = User.query.get(project.user_id)
                    if student and student.user_type == 'student':
                        student_collab_counts[student.id] = student_collab_counts.get(student.id, 0) + 1
        top_students = sorted([(User.query.get(sid), count) for sid, count in student_collab_counts.items()], key=lambda x: x[1], reverse=True)
        analytics['top_students'] = top_students[:5]
        # 3. Most popular subjects among collaborators
        subject_popularity = {}
        for student in students_collaborated:
            for p in student.projects:
                if p.subject:
                    subject_popularity[p.subject] = subject_popularity.get(p.subject, 0) + 1
        analytics['subject_popularity'] = sorted(subject_popularity.items(), key=lambda x: x[1], reverse=True)
        # 4. Student activity stats
        student_activity = []
        for student in students_collaborated:
            student_activity.append({
                'username': student.username,
                'projects': len(student.projects),
                'queries': len(student.queries),
                'resources': len(student.resources)
            })
        analytics['student_activity'] = student_activity
        # 5. Chart data
        chart_data['collab_per_subject'] = [{'subject': s, 'count': c} for s, c in subject_collab_counts.items()]
        chart_data['student_activity'] = student_activity

    return render_template('profile.html', 
                          user=user, 
                          projects=projects, 
                          queries=queries, 
                          resources=resources,
                          students_grouped_by_subject=students_grouped_by_subject,
                          students_collaborated=students_collaborated,
                          students_list=students_list,
                          filter_subject=filter_subject,
                          filter_institution=filter_institution,
                          filter_name=filter_name,
                          analytics=analytics,
                          chart_data=chart_data)

@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    """Edit user profile"""
    if request.method == 'POST':
        current_user.username = request.form.get('username')
        current_user.email = request.form.get('email')
        current_user.institution = request.form.get('institution')
        current_user.bio = request.form.get('bio')
        if current_user.user_type == 'educator':
            current_user.domain = request.form.get('domain')
            current_user.skills = request.form.get('skills')
            current_user.designation = request.form.get('designation')
        # Handle profile picture upload
        if 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'profile_pics', filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                file.save(file_path)
                current_user.profile_pic = filename
        db.session.commit()
        flash('Profile updated successfully.')
        return redirect(url_for('profile', username=current_user.username))
    
    return render_template('edit_profile.html', user=current_user)

@app.route('/projects')
def projects():
    """List all projects with filter options"""
    subject = request.args.get('subject')
    project_type = request.args.get('project_type')
    
    query = Project.query
    
    if subject:
        query = query.filter_by(subject=subject)
    if project_type:
        query = query.filter_by(project_type=project_type)
        
    projects = query.order_by(Project.date_posted.desc()).all()
    visible_projects = []
    for project in projects:
        if project.scope == 'public':
            visible_projects.append(project)
        elif current_user.is_authenticated and (project.user_id == current_user.id or (project.collaborators and current_user.id in json.loads(project.collaborators))):
            visible_projects.append(project)
    
    subjects = db.session.query(Project.subject).distinct().all()
    project_types = db.session.query(Project.project_type).distinct().all()
    
    return render_template('projects.html', 
                          projects=visible_projects, 
                          subjects=subjects, 
                          project_types=project_types)

@app.route('/project/<int:project_id>')
def view_project(project_id):
    """View a specific project"""
    project = Project.query.get_or_404(project_id)
    comments = Comment.query.filter_by(project_id=project_id).order_by(Comment.date_posted).all()
    
    # Record this view
    if current_user.is_authenticated:
        # Only record view if not the author
        if project.user_id != current_user.id:
            # Log the interaction
            interaction = UserInteraction(
                user_id=current_user.id,
                content_type='project',
                content_id=project_id,
                interaction_type='view'
            )
            db.session.add(interaction)
            
            # Increment view count
            project.views += 1
            db.session.commit()
    
    # Find similar projects
    similar_projects = find_similar_projects(project.description)
    # Filter out the current project if it's in the list
    similar_projects = [p for p in similar_projects if p.id != project_id]
    
    return render_template('view_project.html', 
                          project=project, 
                          comments=comments, 
                          similar_projects=similar_projects)

@app.route('/create_project', methods=['GET', 'POST'])
@login_required
def create_project():
    """Create a new project"""
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        content = request.form.get('content')
        project_type = request.form.get('project_type')
        subject = request.form.get('subject')
        scope = request.form.get('scope')
        collaborators = request.form.getlist('collaborators') if scope == 'private' else []
        
        new_project = Project(
            title=title,
            description=description,
            content=content,
            project_type=project_type,
            subject=subject,
            user_id=current_user.id,
            scope=scope,
            collaborators=json.dumps([int(uid) for uid in collaborators]) if collaborators else None
        )
        
        db.session.add(new_project)
        db.session.commit()
        
        # Handle file uploads
        if 'files[]' in request.files:
            files = request.files.getlist('files[]')
            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'project_files', str(new_project.id), filename)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    file.save(file_path)
                    
                    project_file = ProjectFile(
                        filename=filename,
                        path=file_path,
                        project_id=new_project.id
                    )
                    db.session.add(project_file)
        
        db.session.commit()
        flash('Project created successfully!')
        return redirect(url_for('view_project', project_id=new_project.id))
    
    return render_template('create_project.html', users=User.query.all())

@app.route('/edit_project/<int:project_id>', methods=['GET', 'POST'])
@login_required
def edit_project(project_id):
    """Edit an existing project"""
    project = Project.query.get_or_404(project_id)
    
    # Check if user is the project author
    if project.user_id != current_user.id:
        flash('You can only edit your own projects.')
        return redirect(url_for('view_project', project_id=project_id))
    
    if request.method == 'POST':
        project.title = request.form.get('title')
        project.description = request.form.get('description')
        project.content = request.form.get('content')
        project.project_type = request.form.get('project_type')
        project.subject = request.form.get('subject')
        
        # Handle file uploads
        if 'files[]' in request.files:
            files = request.files.getlist('files[]')
            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'project_files', str(project.id), filename)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    file.save(file_path)
                    
                    project_file = ProjectFile(
                        filename=filename,
                        path=file_path,
                        project_id=project.id
                    )
                    db.session.add(project_file)
        
        db.session.commit()
        flash('Project updated successfully!')
        return redirect(url_for('view_project', project_id=project_id))
    
    return render_template('edit_project.html', project=project, users=User.query.all())

@app.route('/add_collaborator/<int:project_id>', methods=['POST'])
@login_required
def add_collaborator(project_id):
    """Add a collaborator to a project"""
    project = Project.query.get_or_404(project_id)
    
    # Check if user is the project author
    if project.user_id != current_user.id:
        flash('Only the project owner can add collaborators.')
        return redirect(url_for('view_project', project_id=project_id))
    
    username = request.form.get('username')
    user = User.query.filter_by(username=username).first()
    
    if not user:
        flash(f'User {username} not found.')
        return redirect(url_for('view_project', project_id=project_id))
    
    # Check if already a collaborator
    collaborators = json.loads(project.collaborators or '[]')
    if user.id in collaborators:
        flash(f'{username} is already a collaborator.')
        return redirect(url_for('view_project', project_id=project_id))
    
    # Add collaborator
    collaborators.append(user.id)
    project.collaborators = json.dumps(collaborators)
    
    # Create notification for the collaborator
    notification = Notification(
        content=f"You have been added as a collaborator to project: {project.title}",
        type="collaboration",
        user_id=user.id
    )
    
    db.session.add(notification)
    db.session.commit()
    
    flash(f'{username} has been added as a collaborator.')
    return redirect(url_for('view_project', project_id=project_id))

@app.route('/queries')
def queries():
    """List all queries with filter options"""
    subject = request.args.get('subject')
    solved = request.args.get('solved')
    
    query = Query.query
    
    if subject:
        query = query.filter_by(subject=subject)
    if solved is not None:
        query = query.filter_by(solved=(solved == 'true'))
        
    queries = query.order_by(Query.date_posted.desc()).all()
    
    subjects = db.session.query(Query.subject).distinct().all()
    
    return render_template('queries.html', 
                          queries=queries, 
                          subjects=subjects)

@app.route('/query/<int:query_id>')
def view_query(query_id):
    """View a specific query and its answers"""
    query = Query.query.get_or_404(query_id)
    answers = db.session.query(Answer).filter_by(query_id=query_id).order_by(Answer.votes.desc()).all()
    
    # Record this view
    if current_user.is_authenticated:
        # Only record view if not the author
        if query.user_id != current_user.id:
            # Log the interaction
            interaction = UserInteraction(
                user_id=current_user.id,
                content_type='query',
                content_id=query_id,
                interaction_type='view'
            )
            db.session.add(interaction)
            
            # Increment view count
            query.views += 1
            db.session.commit()
    
    # Get AI-assisted suggestions
    ai_suggestions = analyze_query(query.content)
    
    return render_template('view_query.html', 
                          query=query, 
                          answers=answers, 
                          ai_suggestions=ai_suggestions)

@app.route('/create_query', methods=['GET', 'POST'])
@login_required
def create_query():
    """Create a new query"""
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        subject = request.form.get('subject')
        
        new_query = Query(
            title=title,
            content=content,
            subject=subject,
            user_id=current_user.id
        )
        
        db.session.add(new_query)
        db.session.commit()
        
        flash('Query posted successfully!')
        return redirect(url_for('view_query', query_id=new_query.id))
    
    return render_template('create_query.html')

@app.route('/answer_query/<int:query_id>', methods=['POST'])
@login_required
def answer_query(query_id):
    """Post an answer to a query"""
    query = Query.query.get_or_404(query_id)
    content = request.form.get('content')
    
    new_answer = Answer(
        content=content,
        query_id=query_id,
        user_id=current_user.id
    )
    
    db.session.add(new_answer)
    
    # Create notification for query author
    if query.user_id != current_user.id:
        notification = Notification(
            content=f"Your query '{query.title}' has a new answer.",
            type="answer",
            user_id=query.user_id
        )
        db.session.add(notification)
    
    db.session.commit()
    
    flash('Answer posted successfully!')
    return redirect(url_for('view_query', query_id=query_id))

@app.route('/mark_answer/<int:answer_id>', methods=['POST'])
@login_required
def mark_answer(answer_id):
    """Mark an answer as correct"""
    answer = Answer.query.get_or_404(answer_id)
    query = Query.query.get_or_404(answer.query_id)
    
    # Check if user is the query author
    if query.user_id != current_user.id:
        flash('Only the query author can mark answers as correct.')
        return redirect(url_for('view_query', query_id=query.id))
    
    # Unmark all other answers
    for a in query.answers:
        a.marked_correct = False
    
    # Mark this answer as correct
    answer.marked_correct = True
    query.solved = True
    
    # Create notification for answer author
    if answer.user_id != current_user.id:
        notification = Notification(
            content=f"Your answer to '{query.title}' was marked as correct!",
            type="correct_answer",
            user_id=answer.user_id
        )
        db.session.add(notification)
    
    db.session.commit()
    
    flash('Answer marked as correct!')
    return redirect(url_for('view_query', query_id=query.id))

@app.route('/vote_answer/<int:answer_id>/<int:vote>', methods=['POST'])
@login_required
def vote_answer(answer_id, vote):
    """Vote on an answer (1 for upvote, -1 for downvote)"""
    answer = Answer.query.get_or_404(answer_id)
    
    # Check if user is not voting on their own answer
    if answer.user_id == current_user.id:
        flash('You cannot vote on your own answer.')
        return redirect(url_for('view_query', query_id=answer.query_id))
    
    # Update vote count
    answer.votes += vote
    
    db.session.commit()
    
    return redirect(url_for('view_query', query_id=answer.query_id))

@app.route('/resources')
def resources():
    """List all resources with filter options"""
    subject = request.args.get('subject')
    resource_type = request.args.get('resource_type')
    
    query = Resource.query
    
    if subject:
        query = query.filter_by(subject=subject)
    if resource_type:
        query = query.filter_by(resource_type=resource_type)
        
    resources = query.order_by(Resource.date_posted.desc()).all()
    
    subjects = db.session.query(Resource.subject).distinct().all()
    resource_types = db.session.query(Resource.resource_type).distinct().all()
    
    return render_template('resources.html', 
                          resources=resources, 
                          subjects=subjects, 
                          resource_types=resource_types)

@app.route('/resource/<int:resource_id>')
def view_resource(resource_id):
    """View a specific resource"""
    resource = Resource.query.get_or_404(resource_id)
    ratings = ResourceRating.query.filter_by(resource_id=resource_id).all()
    
    # Calculate average rating
    avg_rating = 0
    if ratings:
        avg_rating = sum(rating.rating for rating in ratings) / len(ratings)
    
    # Record this view/download
    if current_user.is_authenticated:
        # Only record if not the uploader
        if resource.user_id != current_user.id:
            # Log the interaction
            interaction = UserInteraction(
                user_id=current_user.id,
                content_type='resource',
                content_id=resource_id,
                interaction_type='view'
            )
            db.session.add(interaction)
            db.session.commit()
    
    return render_template('view_resource.html', 
                          resource=resource, 
                          ratings=ratings, 
                          avg_rating=avg_rating)

@app.route('/download_resource/<int:resource_id>')
def download_resource(resource_id):
    """Download a resource file"""
    resource = Resource.query.get_or_404(resource_id)
    
    # Record the download
    if current_user.is_authenticated:
        # Log the interaction
        interaction = UserInteraction(
            user_id=current_user.id,
            content_type='resource',
            content_id=resource_id,
            interaction_type='download'
        )
        db.session.add(interaction)
        
    # Increment download count
    resource.downloads += 1
    db.session.commit()
    
    if resource.path:
        return send_file(resource.path, as_attachment=True, download_name=resource.filename)
    elif resource.external_link:
        return redirect(resource.external_link)
    else:
        flash('No downloadable content available for this resource.')
        return redirect(url_for('view_resource', resource_id=resource_id))

@app.route('/create_resource', methods=['GET', 'POST'])
@login_required
def create_resource():
    """Create a new resource"""
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        resource_type = request.form.get('resource_type')
        subject = request.form.get('subject')
        external_link = request.form.get('external_link')
        
        new_resource = Resource(
            title=title,
            description=description,
            resource_type=resource_type,
            subject=subject,
            external_link=external_link,
            user_id=current_user.id
        )
        
        # Handle file upload
        if 'resource_file' in request.files:
            file = request.files['resource_file']
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resources', filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                file.save(file_path)
                
                new_resource.filename = filename
                new_resource.path = file_path
        
        db.session.add(new_resource)
        db.session.commit()
        
        flash('Resource created successfully!')
        return redirect(url_for('view_resource', resource_id=new_resource.id))
    
    return render_template('create_resource.html')

@app.route('/rate_resource/<int:resource_id>', methods=['POST'])
@login_required
def rate_resource(resource_id):
    """Rate a resource"""
    resource = Resource.query.get_or_404(resource_id)
    
    # Check if user has already rated this resource
    existing_rating = ResourceRating.query.filter_by(
        user_id=current_user.id,
        resource_id=resource_id
    ).first()
    
    rating = int(request.form.get('rating'))
    review = request.form.get('review')
    
    if existing_rating:
        # Update existing rating
        existing_rating.rating = rating
        existing_rating.review = review
    else:
        # Create new rating
        new_rating = ResourceRating(
            rating=rating,
            review=review,
            user_id=current_user.id,
            resource_id=resource_id
        )
        db.session.add(new_rating)
        
        # Create notification for resource uploader
        if resource.user_id != current_user.id:
            notification = Notification(
                content=f"Your resource '{resource.title}' has a new rating.",
                type="rating",
                user_id=resource.user_id
            )
            db.session.add(notification)
    
    db.session.commit()
    
    flash('Rating submitted successfully!')
    return redirect(url_for('view_resource', resource_id=resource_id))

@app.route('/comment_project/<int:project_id>', methods=['POST'])
@login_required
def comment_project(project_id):
    """Add a comment to a project"""
    project = Project.query.get_or_404(project_id)
    content = request.form.get('content')
    
    new_comment = Comment(
        content=content,
        project_id=project_id,
        user_id=current_user.id
    )
    
    db.session.add(new_comment)
    
    # Create notification for project author
    if project.user_id != current_user.id:
        notification = Notification(
            content=f"Your project '{project.title}' has a new comment.",
            type="comment",
            user_id=project.user_id
        )
        db.session.add(notification)
    
    db.session.commit()
    
    flash('Comment added successfully!')
    return redirect(url_for('view_project', project_id=project_id))

@app.route('/like_project/<int:project_id>', methods=['POST'])
@login_required
def like_project(project_id):
    """Like a project"""
    project = Project.query.get_or_404(project_id)
    
    # Check if user is not liking their own project
    if project.user_id == current_user.id:
        flash('You cannot like your own project.')
        return redirect(url_for('view_project', project_id=project_id))
    
    # Log the interaction
    interaction = UserInteraction(
        user_id=current_user.id,
        content_type='project',
        content_id=project_id,
        interaction_type='like'
    )
    db.session.add(interaction)
    
    # Increment like count
    project.likes += 1
    
    # Create notification for project author
    notification = Notification(
        content=f"Your project '{project.title}' received a like!",
        type="like",
        user_id=project.user_id
    )
    db.session.add(notification)
    
    db.session.commit()
    
    return redirect(url_for('view_project', project_id=project_id))

@app.route('/notifications')
@login_required
def notifications():
    """View all notifications"""
    notifications = Notification.query.filter_by(user_id=current_user.id).order_by(Notification.date_created.desc()).all()
    
    # Mark all as read
    for notification in notifications:
        notification.read = True
    
    db.session.commit()
    
    return render_template('notifications.html', notifications=notifications)

@app.route('/search')
def search():
    """Search functionality"""
    query = request.args.get('q', '')
    search_type = request.args.get('type', 'all')
    
    if not query:
        return render_template('search.html', results=None)
    
    results = {
        'projects': [],
        'queries': [],
        'resources': [],
        'users': []
    }
    
    # Search projects
    if search_type in ['all', 'projects']:
        projects = Project.query.filter(
            (Project.title.contains(query)) | 
            (Project.description.contains(query)) |
            (Project.content.contains(query))
        ).all()
        results['projects'] = projects
    
    # Search queries
    if search_type in ['all', 'queries']:
        queries = Query.query.filter(
            (Query.title.contains(query)) | 
            (Query.content.contains(query))
        ).all()
        results['queries'] = queries
    
    # Search resources
    if search_type in ['all', 'resources']:
        resources = Resource.query.filter(
            (Resource.title.contains(query)) | 
            (Resource.description.contains(query))
        ).all()
        results['resources'] = resources
    
    # Search users
    if search_type in ['all', 'users']:
        users = User.query.filter(
            (User.username.contains(query)) | 
            (User.bio.contains(query))
        ).all()
        results['users'] = users
    
    return render_template('search.html', results=results, query=query, search_type=search_type)

@app.route('/analytics')
@login_required
def analytics():
    user_projects = Project.query.filter_by(user_id=current_user.id).all()
    user_queries = Query.query.filter_by(user_id=current_user.id).all()
    user_resources = Resource.query.filter_by(user_id=current_user.id).all()
    user_answers = current_user.answers

    # Project stats
    project_views = sum(p.views for p in user_projects)
    project_likes = sum(p.likes for p in user_projects)
    project_data = []
    subject_counts = {}
    type_counts = {}
    for project in user_projects:
        subject_counts[project.subject] = subject_counts.get(project.subject, 0) + 1
        type_counts[project.project_type] = type_counts.get(project.project_type, 0) + 1
        project_data.append({
            'title': project.title,
            'views': project.views,
            'likes': project.likes
        })

    # Q&A stats
    query_views = sum(q.views for q in user_queries)
    resource_downloads = sum(r.downloads for r in user_resources)
    num_questions = len(user_queries)
    num_answers = len(user_answers)
    num_solved = sum(1 for q in user_queries if q.solved)

    # Industrial/mentor/guide connections
    industrial_people = set()
    for project in user_projects:
        if project.collaborators:
            for uid in json.loads(project.collaborators):
                user = User.query.get(uid)
                if user and user.user_type in ['educator', 'guide', 'admin']:
                    industrial_people.add(user.username)
        for comment in project.comments:
            user = User.query.get(comment.user_id)
            if user and user.user_type in ['educator', 'guide', 'admin']:
                industrial_people.add(user.username)

    ai_insights = generate_ai_insights(user_projects, user_queries, num_solved, industrial_people, subject_counts, type_counts)
    return render_template('analytics.html',
        project_views=project_views,
        project_likes=project_likes,
        query_views=query_views,
        resource_downloads=resource_downloads,
        project_data=project_data,
        subject_counts=subject_counts,
        type_counts=type_counts,
        num_questions=num_questions,
        num_answers=num_answers,
        num_solved=num_solved,
        industrial_people=list(industrial_people),
        ai_insights=ai_insights
    )

@app.route('/api/recommend_projects')
@login_required
def api_recommend_projects():
    """API endpoint to get project recommendations"""
    # Get user's viewed projects
    interactions = UserInteraction.query.filter_by(
        user_id=current_user.id,
        content_type='project',
        interaction_type='view'
    ).all()
    
    viewed_project_ids = [interaction.content_id for interaction in interactions]
    
    # Get interests based on viewed projects
    interests = []
    for project_id in viewed_project_ids:
        project = Project.query.get(project_id)
        if project and project.subject:
            interests.append(project.subject)
    
    # Count frequency of each interest
    interest_counts = {}
    for interest in interests:
        if interest in interest_counts:
            interest_counts[interest] += 1
        else:
            interest_counts[interest] = 1
    
    # Get top interests
    top_interests = sorted(interest_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    top_interests = [i[0] for i in top_interests]
    
    # Get recommended projects
    recommended_projects = []
    for interest in top_interests:
        projects = Project.query.filter_by(subject=interest).limit(3).all()
        for project in projects:
            if project.id not in viewed_project_ids and project.user_id != current_user.id:
                recommended_projects.append({
                    'id': project.id,
                    'title': project.title,
                    'description': project.description,
                    'author': User.query.get(project.user_id).username,
                    'subject': project.subject
                })
    
    return jsonify(recommended_projects[:5])

@app.route('/api/mark_notification_read/<int:notification_id>', methods=['POST'])
@login_required
def api_mark_notification_read(notification_id):
    """API endpoint to mark a notification as read"""
    notification = Notification.query.get_or_404(notification_id)
    
    if notification.user_id != current_user.id:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    notification.read = True
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/project_activity/<int:project_id>')
@login_required
def api_project_activity(project_id):
    """API endpoint to get activity data for a project"""
    project = Project.query.get_or_404(project_id)
    
    # Check if user has access to this project
    if project.user_id != current_user.id and current_user.id not in json.loads(project.collaborators or '[]'):
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    # Get all interactions with this project
    interactions = UserInteraction.query.filter_by(
        content_type='project',
        content_id=project_id
    ).order_by(UserInteraction.timestamp).all()
    
    # Format activity data
    activity_data = []
    for interaction in interactions:
        user = User.query.get(interaction.user_id)
        activity_data.append({
            'user': user.username,
            'type': interaction.interaction_type,
            'timestamp': interaction.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Get comments
    comments = Comment.query.filter_by(project_id=project_id).order_by(Comment.date_posted).all()
    for comment in comments:
        user = User.query.get(comment.user_id)
        activity_data.append({
            'user': user.username,
            'type': 'comment',
            'content': comment.content,
            'timestamp': comment.date_posted.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Sort by timestamp
    activity_data.sort(key=lambda x: x['timestamp'])
    
    return jsonify(activity_data)

@app.route('/api/query_suggestions/<int:query_id>')
@login_required
def api_query_suggestions(query_id):
    """API endpoint to get AI-generated suggestions for a query"""
    query = Query.query.get_or_404(query_id)
    
    # Analyze the query
    suggestions = analyze_query(query.content)
    
    return jsonify(suggestions)

@app.route('/api/recommend_teachers', methods=['POST'])
@login_required
def api_recommend_teachers():
    data = request.get_json()
    project_title = data.get('title', '')
    project_description = data.get('description', '')
    project_subject = data.get('subject', '')
    # Combine title and description for skill matching
    project_text = f"{project_title} {project_description}".lower()
    project_keywords = set(project_text.split())
    educators = User.query.filter_by(user_type='educator').all()
    recommendations = []
    for educator in educators:
        score = 0
        # Domain/subject match
        if educator.domain and project_subject and project_subject.lower() in educator.domain.lower():
            score += 2
        # Skill match
        if educator.skills:
            educator_skills = [s.strip().lower() for s in educator.skills.split(',')]
            skill_matches = [s for s in educator_skills if s in project_keywords]
            score += len(skill_matches)
        # Designation bonus (optional, e.g., if 'professor' or 'mentor')
        if educator.designation and any(word in educator.designation.lower() for word in ['professor', 'mentor', 'guide']):
            score += 1
        recommendations.append({
            'id': educator.id,
            'username': educator.username,
            'domain': educator.domain,
            'skills': educator.skills,
            'designation': educator.designation,
            'institution': educator.institution,
            'score': score
        })
    # Sort by score descending
    recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
    return jsonify(recommendations[:5])

# AI model training and evaluation routes
@app.route('/admin/train_recommendation_model')
@login_required
def train_recommendation_model():
    """Admin route to train the recommendation model"""
    # Check if user is an admin
    if current_user.username != 'admin':
        flash('Access denied.')
        return redirect(url_for('dashboard'))
    
    # Collect training data
    interactions = UserInteraction.query.all()
    
    # Create user-item interaction matrix
    user_project_interactions = {}
    
    for interaction in interactions:
        if interaction.content_type == 'project':
            user_id = interaction.user_id
            project_id = interaction.content_id
            
            if user_id not in user_project_interactions:
                user_project_interactions[user_id] = {}
            
            if project_id not in user_project_interactions[user_id]:
                if interaction.interaction_type == 'view':
                    user_project_interactions[user_id][project_id] = 1
                elif interaction.interaction_type == 'like':
                    user_project_interactions[user_id][project_id] = 5
            else:
                if interaction.interaction_type == 'like':
                    user_project_interactions[user_id][project_id] = 5
    
    # Convert to pandas DataFrame
    data = []
    for user_id, projects in user_project_interactions.items():
        for project_id, rating in projects.items():
            data.append([user_id, project_id, rating])
    
    if data:
        df = pd.DataFrame(data, columns=['user_id', 'project_id', 'rating'])
        
        # Save training data
        df.to_csv('recommendation_data.csv', index=False)
        
        # In a real application, this is where you'd train an ML model
        # For simplicity, we'll just save the interaction data
        
        flash('Recommendation model training data collected!')
    else:
        flash('No interaction data available for training.')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    """Admin dashboard with system statistics"""
    # Check if user is an admin
    if current_user.username != 'admin':
        flash('Access denied.')
        return redirect(url_for('dashboard'))
    
    # Calculate system statistics
    user_count = User.query.count()
    project_count = Project.query.count()
    query_count = Query.query.count()
    resource_count = Resource.query.count()
    comment_count = Comment.query.count()
    
    # Get most viewed projects
    top_projects = Project.query.order_by(Project.views.desc()).limit(5).all()
    
    # Get most downloaded resources
    top_resources = Resource.query.order_by(Resource.downloads.desc()).limit(5).all()
    
    # Get most active users
    users_activity = db.session.query(
        UserInteraction.user_id, 
        db.func.count(UserInteraction.id)
    ).group_by(UserInteraction.user_id).order_by(db.func.count(UserInteraction.id).desc()).limit(5).all()
    
    top_users = []
    for user_id, activity_count in users_activity:
        user = User.query.get(user_id)
        if user:
            top_users.append({
                'username': user.username,
                'activity_count': activity_count
            })
    
    return render_template('admin_dashboard.html',
                          user_count=user_count,
                          project_count=project_count,
                          query_count=query_count,
                          resource_count=resource_count,
                          comment_count=comment_count,
                          top_projects=top_projects,
                          top_resources=top_resources,
                          top_users=top_users)

@app.route('/join_project/<int:project_id>', methods=['POST'])
@login_required
def join_project(project_id):
    project = Project.query.get_or_404(project_id)
    if project.scope != 'public':
        flash('You can only join public projects.')
        return redirect(url_for('dashboard'))
    if project.user_id == current_user.id:
        flash('You cannot join your own project.')
        return redirect(url_for('dashboard'))
    # Check if already requested
    existing_request = JoinRequest.query.filter_by(user_id=current_user.id, project_id=project_id, status='pending').first()
    if existing_request:
        flash('You have already requested to join this project.')
        return redirect(url_for('dashboard'))
    # Create join request
    join_request = JoinRequest(user_id=current_user.id, project_id=project_id)
    db.session.add(join_request)
    # Notify project owner
    notification = Notification(
        content=f"{current_user.username} has requested to join your project '{project.title}' as a collaborator.",
        type="join_request",
        user_id=project.user_id
    )
    db.session.add(notification)
    db.session.commit()
    flash('Join request sent to the project owner!')
    return redirect(url_for('dashboard'))

@app.route('/handle_join_request/<int:request_id>/<action>', methods=['POST'])
@login_required
def handle_join_request(request_id, action):
    join_request = JoinRequest.query.get_or_404(request_id)
    project = Project.query.get_or_404(join_request.project_id)
    if project.user_id != current_user.id:
        flash('Only the project owner can handle join requests.')
        return redirect(url_for('dashboard'))
    if action == 'accept':
        join_request.status = 'accepted'
        collaborators = json.loads(project.collaborators or '[]')
        if join_request.user_id not in collaborators:
            collaborators.append(join_request.user_id)
            project.collaborators = json.dumps(collaborators)
        # Notify user
        notification = Notification(
            content=f"Your request to join '{project.title}' was accepted!",
            type="join_accepted",
            user_id=join_request.user_id
        )
        db.session.add(notification)
    elif action == 'decline':
        join_request.status = 'declined'
        notification = Notification(
            content=f"Your request to join '{project.title}' was declined.",
            type="join_declined",
            user_id=join_request.user_id
        )
        db.session.add(notification)
    db.session.commit()
    return redirect(url_for('dashboard'))

@app.route('/register_educator', methods=['GET', 'POST'])
def register_educator():
    """Educator registration page"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        institution = request.form.get('institution')
        domain = request.form.get('domain')
        skills = request.form.get('skills')
        designation = request.form.get('designation')

        # Check if username or email already exists
        user_exists = User.query.filter((User.username == username) | (User.email == email)).first()
        if user_exists:
            flash('Username or email already exists.')
            return redirect(url_for('register_educator'))

        # Create new educator user
        new_user = User(
            username=username,
            email=email,
            user_type='educator',
            institution=institution,
            domain=domain,
            skills=skills,
            designation=designation
        )
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Educator registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register_educator.html')

@app.template_filter('from_json')
def from_json_filter(s):
    import json
    return json.loads(s) if s else []

# Run the application
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables
    app.run(debug=True)