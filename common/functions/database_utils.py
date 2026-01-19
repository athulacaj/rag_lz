import sqlite3
from sqlite3 import Error
import logging
import json
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_connection(db_file):
    """ 
    Create a database connection to the SQLite database specified by db_file.
    If the database does not exist, it will be created.
    
    :param db_file: database file path
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        logging.info(f"Connected to SQLite database: {db_file}")
        return conn
    except Error as e:
        logging.error(f"Error connecting to database: {e}")
    
    return conn

def get_schema(conn):
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%';
        """)

        schema = {}
        for (table,) in cursor.fetchall():
            cursor.execute(f"PRAGMA table_info({table});")
            schema[table] = [
                {"column": c[1], "type": c[2]} for c in cursor.fetchall()
            ]

        return schema
    except Error as e:
        logging.error(f"Error getting schema: {e}")
        return None

def schema_to_text(schema):
    lines = []
    for table, cols in schema.items():
        col_str = ", ".join(f"{c['column']} ({c['type']})" for c in cols)
        lines.append(f"Table {table}: {col_str}")
    return "\n".join(lines) 

def create_table(conn, create_table_sql):
    """ 
    Create a table from the create_table_sql statement
    
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
        logging.info("Table created successfully")
    except Error as e:
        logging.error(f"Error creating table: {e}")

def create_record(conn, sql, params):
    """
    Create a new record into the table
    
    :param conn: Connection object
    :param sql: INSERT statement
    :param params: tuple of values
    :return: last row id
    """
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        logging.info(f"Record created successfully with ID: {cur.lastrowid}")
        return cur.lastrowid
    except Error as e:
        logging.error(f"Error creating record: {e}")
        return None

def read_records(conn, sql, params=None):
    """
    Query all rows in the table
    
    :param conn: Connection object
    :param sql: SELECT statement
    :param params: tuple of values (optional)
    :return: list of rows
    """
    try:
        cur = conn.cursor()
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        rows = cur.fetchall()
        return rows
    except Error as e:
        logging.error(f"Error reading records: {e}")
        return []

def update_record(conn, sql, params):
    """
    Update a record in the table
    
    :param conn: Connection object
    :param sql: UPDATE statement
    :param params: tuple of values
    :return: number of rows updated
    """
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        logging.info(f"Updated {cur.rowcount} row(s)")
        return cur.rowcount
    except Error as e:
        logging.error(f"Error updating record: {e}")
        return 0

def delete_record(conn, sql, params):
    """
    Delete a record from the table
    
    :param conn: Connection object
    :param sql: DELETE statement
    :param params: tuple of values
    :return: number of rows deleted
    """
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        logging.info(f"Deleted {cur.rowcount} row(s)")
        return cur.rowcount
    except Error as e:
        logging.error(f"Error deleting record: {e}")
        return 0

def close_connection(conn):
    """
    Close the database connection
    
    :param conn: Connection object
    """
    if conn:
        conn.close()
        logging.info("Database connection closed")

@contextmanager
def get_db_connection(db_file):
    """
    Context manager for database connections.
    Automatically closes the connection when the block exits.
    
    Usage:
    with get_db_connection(db_file) as conn:
        # do operations
    """
    conn = None
    try:
        conn = create_connection(db_file)
        if conn is None:
            yield None
        else:
            yield conn
    finally:
        if conn:
            close_connection(conn)

def create_resume_tables(conn):
    """
    Create the users and experience tables with the required keys.
    Users table includes general info and skills.
    Experience table includes experience details linked to users.
    
    :param conn: Connection object
    """
    users_sql = """
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        name TEXT,
        position TEXT,
        skills TEXT
    );
    """
    
    experience_sql = """
    CREATE TABLE IF NOT EXISTS experience (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT NOT NULL,
        company_name TEXT,
        start_date TEXT,
        end_date TEXT,
        position TEXT,
        description TEXT,
        FOREIGN KEY (user_email) REFERENCES users (email)
    );
    """
    
    create_table(conn, users_sql)
    create_table(conn, experience_sql)

def insert_resume_data(conn, resume_data):
    """
    Insert resume data from a dictionary into users and experience tables.
    
    :param conn: Connection object
    :param resume_data: Dictionary containing resume data
    """
    try:
        general = resume_data.get("general", {})
        skills = resume_data.get("skills", [])
        experience = resume_data.get("experience", [])
        
        email = general.get("email")
        if not email:
            logging.error("Email is mandatory for inserting resume data.")
            return

        # Insert user
        # Using INSERT OR REPLACE to update if exists
        user_sql = """
        INSERT OR REPLACE INTO users (email, name, position, skills)
        VALUES (?, ?, ?, ?);
        """
        # Convert skills list to JSON string
        skills_str = json.dumps(skills)
        
        user_params = (email, general.get("name"), general.get("position"), skills_str)
        create_record(conn, user_sql, user_params)
        
        # Insert experience
        # First, delete existing experience for this user
        delete_record(conn, "DELETE FROM experience WHERE user_email = ?", (email,))
        
        # Then insert new entries
        exp_sql = """
        INSERT INTO experience (user_email, company_name, start_date, end_date, position, description)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        
        for exp in experience:
            exp_params = (
                email,
                exp.get("company_name"),
                exp.get("start_date"),
                exp.get("end_date"),
                exp.get("position"),
                exp.get("description")
            )
            create_record(conn, exp_sql, exp_params)
        
        logging.info(f"Inserted resume data for user: {email}")

    except Exception as e:
        logging.error(f"Error inserting resume data: {e}")

def get_data_by_email(conn, email_or_list):
    """
    Get user and experience data by email(s).
    
    :param conn: Connection object
    :param email_or_list: Single email string or list of emails
    :return: List of dictionaries containing user and experience data
    """
    if isinstance(email_or_list, str):
        emails = [email_or_list]
    else:
        emails = email_or_list
        
    results = []
    
    for email in emails:
        user_sql = "SELECT * FROM users WHERE email = ?"
        user_rows = read_records(conn, user_sql, (email,))
        
        if not user_rows:
            continue
            
        user_data = user_rows[0]
        # user_data is a tuple (email, name, position, skills)
        
        user_dict = {
            "email": user_data[0],
            "name": user_data[1],
            "position": user_data[2],
            "skills": user_data[3]
        }
        
        try:
            if user_dict["skills"]:
                user_dict["skills"] = json.loads(user_dict["skills"])
        except:
            pass
            
        exp_sql = "SELECT * FROM experience WHERE user_email = ?"
        exp_rows = read_records(conn, exp_sql, (email,))
        
        exp_list = []
        for row in exp_rows:
            # row is (id, user_email, company_name, start_date, end_date, position, description)
            exp_dict = {
                "company_name": row[2],
                "start_date": row[3],
                "end_date": row[4],
                "position": row[5],
                "description": row[6]
            }
            exp_list.append(exp_dict)
            
        results.append({
            "general": user_dict,
            "experience": exp_list
        })
        
    return results

def get_data_by_name(conn, name_or_list):
    """
    Get user and experience data by name(s).
    
    :param conn: Connection object
    :param name_or_list: Single name string or list of names
    :return: List of dictionaries containing user and experience data
    """
    if isinstance(name_or_list, str):
        names = [name_or_list]
    else:
        names = name_or_list
        
    results = []
    
    for name in names:
        user_sql = "SELECT * FROM users WHERE name LIKE ?"
        user_rows = read_records(conn, user_sql, (f"%{name}%",))
        
        if not user_rows:
            continue
            
        user_data = user_rows[0]
        # user_data is a tuple (email, name, position, skills)
        
        user_dict = {
            "email": user_data[0],
            "name": user_data[1],
            "position": user_data[2],
            "skills": user_data[3]
        }
        
        try:
            if user_dict["skills"]:
                user_dict["skills"] = json.loads(user_dict["skills"])
        except:
            pass
            
        exp_sql = "SELECT * FROM experience WHERE user_email = ?"
        exp_rows = read_records(conn, exp_sql, (user_data[0],))
        
        exp_list = []
        for row in exp_rows:
            # row is (id, user_email, company_name, start_date, end_date, position, description)
            exp_dict = {
                "company_name": row[2],
                "start_date": row[3],
                "end_date": row[4],
                "position": row[5],
                "description": row[6]
            }
            exp_list.append(exp_dict)
            
        results.append({
            "general": user_dict,
            "experience": exp_list
        })
        
    return results

def read_db_by_sql(conn, sql, params=None):
    """
    Execute a read-only SQL query.
    
    :param conn: Connection object
    :param sql: SELECT statement
    :param params: tuple of values (optional)
    :return: list of rows
    """
    if not sql.strip().upper().startswith("SELECT"):
        logging.error("Only SELECT queries are allowed.")
        return None
        
    return read_records(conn, sql, params)
