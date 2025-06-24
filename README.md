# Academic Collaboration Platform

A full-featured academic collaboration platform for students and educators to share projects, resources, queries, and collaborate online.

## Features
- User registration/login for students and educators
- Educator-specific fields (domain, skills, designation)
- Project creation, file uploads, and collaboration
- AI-powered teacher recommendation for students
- Invite students as collaborators (for educators)
- Analytics and dashboards
- Notifications, comments, and more

## Local Development

1. **Clone the repository**
2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. **Run the app locally:**
   ```bash
   python app.py
   ```
   The app will be available at `http://localhost:5000`.

4. **Database migrations:**
   ```bash
   flask db init         # Only once, to initialize migrations
   flask db migrate -m "Message"
   flask db upgrade
   ```

## Deployment on Vercel

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```
2. **Project structure:**
   - `api/index.py` (entry point, imports your Flask app)
   - `app.py` (your main Flask app)
   - `requirements.txt`, `vercel.json`, `static/`, `templates/` (as usual)

3. **vercel.json** is already configured:
   ```json
   {
     "version": 2,
     "builds": [
       { "src": "api/index.py", "use": "@vercel/python" }
     ],
     "routes": [
       { "src": "/(.*)", "dest": "api/index.py" }
     ]
   }
   ```

4. **Deploy:**
   ```bash
   vercel --prod
   ```
   Follow the prompts to complete deployment.

5. **Database:**
   - Vercel serverless functions are stateless. Use a cloud database (e.g., PostgreSQL, MySQL, MongoDB, or managed SQLite).
   - Update your `SQLALCHEMY_DATABASE_URI` in `app.py` to point to your cloud database.
   - Set environment variables in the Vercel dashboard or with `vercel env add`.

## Notes
- Static files are served from the `static/` directory.
- Templates are in the `templates/` directory.
- For production, always use a secure secret key and a production-ready database.

## License
BSD 3-Clause License

Copyright (c) 2025, PrathamK
