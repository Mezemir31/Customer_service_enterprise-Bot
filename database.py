import sqlite3
import datetime
import os
import logging
from contextlib import contextmanager
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SQLite datetime handling
def adapt_datetime_iso(val):
    """Convert datetime to ISO 8601 string"""
    return val.isoformat()

def convert_datetime_iso(val):
    """Convert ISO 8601 string to datetime"""
    return datetime.datetime.fromisoformat(val.decode())

# Register the adapter and converter for datetime
sqlite3.register_adapter(datetime.datetime, adapt_datetime_iso)
sqlite3.register_converter("datetime", convert_datetime_iso)

# Also adapt date objects
def adapt_date_iso(val):
    """Convert date to ISO 8601 string"""
    return val.isoformat()

def convert_date_iso(val):
    """Convert ISO 8601 string to date"""
    return datetime.date.fromisoformat(val.decode())

sqlite3.register_adapter(datetime.date, adapt_date_iso)
sqlite3.register_converter("date", convert_date_iso)

@contextmanager
def get_db_connection():
    """Get a database connection with type detection enabled as a context manager"""
    conn = None
    try:
        conn = sqlite3.connect(Config.DB_NAME, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row  # Enable accessing columns by name
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def init_database(reset=False):
    """Initialize the database schema. Set reset=True to drop existing tables."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            if reset:
                logger.warning("[X] Resetting database tables...")
                cursor.execute('DROP TABLE IF EXISTS conversations')
                cursor.execute('DROP TABLE IF EXISTS session_context')
                cursor.execute('DROP TABLE IF EXISTS customer_profiles')
                cursor.execute('DROP TABLE IF EXISTS knowledge_base')
            
            # Main conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    phone_number TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    user_input TEXT,
                    detected_language TEXT,
                    grievance_type TEXT,
                    extracted_phones TEXT,
                    extracted_digits TEXT,
                    extracted_fin_uin TEXT,
                    extracted_fan_fcn TEXT,
                    response TEXT,
                    confidence TEXT,
                    sentiment_score REAL,
                    urgency_level TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Enhanced session context
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS session_context (
                    session_id TEXT PRIMARY KEY,
                    phone_number TEXT,
                    last_interaction TEXT DEFAULT CURRENT_TIMESTAMP,
                    conversation_history TEXT,
                    sms_attempts INTEGER DEFAULT 0,
                    customer_tier TEXT DEFAULT 'standard',
                    preferred_language TEXT,
                    total_interactions INTEGER DEFAULT 1,
                    avg_sentiment REAL DEFAULT 0.5,
                    escalation_count INTEGER DEFAULT 0
                )
            ''')
            
            # Customer profiles
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS customer_profiles (
                    phone_number TEXT PRIMARY KEY,
                    customer_name TEXT,
                    registration_date TEXT,
                    total_tickets INTEGER DEFAULT 0,
                    resolved_tickets INTEGER DEFAULT 0,
                    preferred_language TEXT,
                    customer_satisfaction_score REAL DEFAULT 0.0,
                    last_interaction TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Knowledge base for common issues
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT,
                    subcategory TEXT,
                    question_pattern TEXT,
                    response_template TEXT,
                    language TEXT,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0
                )
            ''')
            
            conn.commit()
            logger.info("Database initialized.")
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def update_database_schema():
    """Update existing database with new columns"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if columns exist in conversations table
            cursor.execute("PRAGMA table_info(conversations)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Define all required columns
            required_columns = {
                'extracted_fin_uin': 'TEXT',
                'extracted_fan_fcn': 'TEXT', 
                'sentiment_score': 'REAL',
                'urgency_level': 'TEXT'
            }
            
            # Add missing columns
            for column_name, column_type in required_columns.items():
                if column_name not in columns:
                    cursor.execute(f'ALTER TABLE conversations ADD COLUMN {column_name} {column_type}')
                    logger.info(f"Added {column_name} column to conversations table")
            
            # Check session_context table
            cursor.execute("PRAGMA table_info(session_context)")
            session_columns = [column[1] for column in cursor.fetchall()]
            
            session_required_columns = {
                'preferred_language': 'TEXT',
                'total_interactions': 'INTEGER DEFAULT 1',
                'avg_sentiment': 'REAL DEFAULT 0.5',
                'escalation_count': 'INTEGER DEFAULT 0'
            }
            
            for column_name, column_type in session_required_columns.items():
                if column_name not in session_columns:
                    cursor.execute(f'ALTER TABLE session_context ADD COLUMN {column_name} {column_type}')
                    logger.info(f"Added {column_name} column to session_context table")
            
            conn.commit()
            logger.info("Database schema updated.")
            
    except Exception as e:
        logger.error(f"Error updating database schema: {e}")
        # We don't automatically reset here to avoid data loss in production

