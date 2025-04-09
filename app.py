from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
import MySQLdb.cursors
from flask_bcrypt import Bcrypt


app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'  
app.config['MYSQL_PASSWORD'] = ''  # Add your MySQL password if there
app.config['MYSQL_DB'] = 'fashion_db'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)
bcrypt = Bcrypt(app)


# ------------------ Render HTML Pages ------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/recommend')
def recommend_page():
    return render_template('recommend.html')

@app.route('/logout')
def logout():
    if 'loggedin' in session:
        session.clear()  # Clear session only when user explicitly logs out
        # flash("You have been logged out!", "info")
    return redirect(url_for('home'))  # Redirect to home page instead of login

@app.errorhandler(500)
def internal_error(error):
    return "500 Internal Server Error. Please try again later.", 500


# ------------------ User Registration ------------------
@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        full_name = request.form['full_name']
        username = request.form['username']
        email = request.form['email']
        phone_number = request.form['phone_number']
        password = request.form['password']

        # Check if user already exists
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s OR phone_number = %s", 
                       (username, email, phone_number))
        existing_user = cursor.fetchone()

        if existing_user:
            flash("User already exists! Please try logging in.", "danger")
            return redirect(url_for('login_page'))  # Go to Login page
        
        # Hash the password before storing it
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

        # Insert new user into the database
        cursor.execute("""
            INSERT INTO users (full_name, username, email, phone_number, password_hash) 
            VALUES (%s, %s, %s, %s, %s)
        """, (full_name, username, email, phone_number, password_hash))

        mysql.connection.commit()
        cursor.close()

        flash("Registration successful!", "success")
        return redirect(url_for('recommend_page'))  # Redirect to recommend page

    return render_template('register.html')

# ------------------ User Login ------------------
@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT id, username, password_hash FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()

        # Check if user exists
        if not user:
            flash("Username not found! Please register first.", "danger")
            return redirect(url_for('register_page'))  # Go to register page

        # Check if password is correct
        if not bcrypt.check_password_hash(user['password_hash'], password):
            flash("Incorrect password! Please try again.", "danger")
            return redirect(url_for('login_page'))  # Stay on login page

        # Successful login, create session
        session['loggedin'] = True
        session['id'] = user['id']
        session['username'] = user['username']
        # flash("Login successful!", "success")
        return redirect(url_for('recommend_page'))  # Redirect to recommend page

    return render_template('login.html')


# ------------------ Run Flask App ------------------
if __name__ == '__main__':
    app.run(debug=True)
