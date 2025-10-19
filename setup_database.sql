-- ============================================================================
-- Database Setup for Medical Chatbot ANN Version
-- ============================================================================
-- This script creates a new database specifically for the ANN-based chatbot
-- to keep it separate from the old PhoBERT version (triage_db)
--
-- Database Name: medical_chatbot_ann
-- Purpose: Store patient records, conversation history, and diagnosis results
--
-- Usage:
--   mysql -u root -p < setup_database.sql
-- ============================================================================

-- Create database for ANN version
CREATE DATABASE IF NOT EXISTS medical_chatbot_ann
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

-- Use the new database
USE medical_chatbot_ann;

-- ============================================================================
-- Table: patient_records
-- ============================================================================
-- Stores complete patient interaction records with diagnosis results
CREATE TABLE IF NOT EXISTS patient_records (
    record_id VARCHAR(64) PRIMARY KEY COMMENT 'Unique record identifier',
    user_id VARCHAR(64) NOT NULL COMMENT 'User/session identifier',

    -- Patient Information
    patient_name VARCHAR(255) COMMENT 'Patient full name',
    patient_phone VARCHAR(20) COMMENT 'Patient phone number',
    patient_age VARCHAR(10) COMMENT 'Patient age',
    patient_gender VARCHAR(10) COMMENT 'Patient gender',

    -- Medical Information
    symptoms TEXT COMMENT 'JSON string of reported symptoms',
    onset VARCHAR(100) COMMENT 'When symptoms started',
    allergies TEXT COMMENT 'Known allergies',
    current_medications TEXT COMMENT 'Current medications',
    pain_scale VARCHAR(10) COMMENT 'Pain scale (1-10)',

    -- Diagnosis Results (from ANN model)
    predicted_diseases TEXT COMMENT 'JSON string of {disease: probability}',
    top_disease VARCHAR(255) COMMENT 'Most likely disease',
    top_confidence DECIMAL(5,4) COMMENT 'Confidence of top prediction (0.0-1.0)',
    confidence_category VARCHAR(20) COMMENT 'HIGH, MEDIUM, or LOW',
    recommended_department VARCHAR(128) COMMENT 'Recommended hospital department',

    -- Active Questioning Data
    questions_asked INT DEFAULT 0 COMMENT 'Number of questions asked during session',
    questioning_strategy VARCHAR(50) COMMENT 'Strategy used: HYBRID, DIFFERENTIAL, etc.',
    initial_confidence DECIMAL(5,4) COMMENT 'Confidence before questioning',
    final_confidence DECIMAL(5,4) COMMENT 'Confidence after questioning',

    -- System & Metadata
    chat TEXT COMMENT 'JSON string of full conversation history',
    model_version VARCHAR(50) DEFAULT 'ANN_v1.0' COMMENT 'Model version used',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Record creation time',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update time',

    -- Indexes
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at),
    INDEX idx_top_disease (top_disease),
    INDEX idx_department (recommended_department),
    INDEX idx_confidence (confidence_category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Patient records with ANN diagnosis and active questioning data';

-- ============================================================================
-- Table: questioning_sessions (Optional - for detailed analytics)
-- ============================================================================
-- Tracks detailed information about active questioning sessions
CREATE TABLE IF NOT EXISTS questioning_sessions (
    session_id VARCHAR(64) PRIMARY KEY COMMENT 'Questioning session ID',
    record_id VARCHAR(64) COMMENT 'Reference to patient_records',

    -- Session Configuration
    max_questions INT DEFAULT 10 COMMENT 'Maximum questions allowed',
    confidence_threshold DECIMAL(5,4) DEFAULT 0.85 COMMENT 'Target confidence threshold',
    strategy VARCHAR(50) COMMENT 'Questioning strategy used',

    -- Session Results
    questions_asked INT DEFAULT 0 COMMENT 'Actual questions asked',
    initial_top_disease VARCHAR(255) COMMENT 'Initial prediction',
    initial_confidence DECIMAL(5,4) COMMENT 'Initial confidence',
    final_top_disease VARCHAR(255) COMMENT 'Final prediction',
    final_confidence DECIMAL(5,4) COMMENT 'Final confidence',
    confidence_improved DECIMAL(5,4) COMMENT 'Confidence gain (final - initial)',

    -- Question Details (JSON)
    questions_log TEXT COMMENT 'JSON array of {question, answer, symptom, strategy}',
    hypotheses_history TEXT COMMENT 'JSON array of hypothesis evolution',

    -- Session Status
    status VARCHAR(20) DEFAULT 'active' COMMENT 'active, completed, abandoned',
    completion_reason VARCHAR(100) COMMENT 'Why session ended',

    -- Timestamps
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,

    -- Foreign key
    FOREIGN KEY (record_id) REFERENCES patient_records(record_id) ON DELETE CASCADE,

    -- Indexes
    INDEX idx_record_id (record_id),
    INDEX idx_status (status),
    INDEX idx_started_at (started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Detailed tracking of active questioning sessions';

-- ============================================================================
-- Table: model_metrics (Optional - for monitoring)
-- ============================================================================
-- Tracks model performance metrics over time
CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id INT AUTO_INCREMENT PRIMARY KEY,

    -- Metric Data
    date DATE NOT NULL COMMENT 'Date of metrics',
    total_predictions INT DEFAULT 0 COMMENT 'Total predictions made',
    avg_confidence DECIMAL(5,4) COMMENT 'Average confidence',
    high_confidence_count INT DEFAULT 0 COMMENT 'Count of HIGH confidence predictions',
    medium_confidence_count INT DEFAULT 0 COMMENT 'Count of MEDIUM confidence predictions',
    low_confidence_count INT DEFAULT 0 COMMENT 'Count of LOW confidence predictions',

    -- Active Questioning Metrics
    avg_questions_asked DECIMAL(5,2) COMMENT 'Average questions per session',
    avg_confidence_gain DECIMAL(5,4) COMMENT 'Average confidence improvement',

    -- Department Distribution (JSON)
    department_distribution TEXT COMMENT 'JSON of {department: count}',

    -- Top Diseases (JSON)
    top_diseases TEXT COMMENT 'JSON array of most common predictions',

    -- System
    model_version VARCHAR(50) COMMENT 'Model version',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    UNIQUE KEY unique_date_version (date, model_version),
    INDEX idx_date (date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Daily model performance metrics';

-- ============================================================================
-- Create user with appropriate permissions (optional)
-- ============================================================================
-- Uncomment the following lines if you want to create a dedicated user

-- CREATE USER IF NOT EXISTS 'medchat_ann'@'localhost' IDENTIFIED BY 'SecurePassword123!';
-- GRANT ALL PRIVILEGES ON medical_chatbot_ann.* TO 'medchat_ann'@'localhost';
-- FLUSH PRIVILEGES;

-- ============================================================================
-- Initial Data / Sample Records (Optional)
-- ============================================================================
-- You can add sample data here for testing

-- ============================================================================
-- Verification Queries
-- ============================================================================
-- Run these to verify the setup

-- Show all tables
SHOW TABLES;

-- Show table structure
DESCRIBE patient_records;
DESCRIBE questioning_sessions;
DESCRIBE model_metrics;

-- Show database info
SELECT
    TABLE_NAME,
    TABLE_ROWS,
    ROUND(DATA_LENGTH / 1024 / 1024, 2) AS 'Size (MB)',
    TABLE_COMMENT
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = 'medical_chatbot_ann'
ORDER BY TABLE_NAME;

-- ============================================================================
-- Done!
-- ============================================================================
-- Database 'medical_chatbot_ann' is now ready for use.
-- Update your .env file with:
-- DATABASE_URL=mysql+pymysql://root:YOUR_PASSWORD@127.0.0.1:3306/medical_chatbot_ann
-- ============================================================================
