// ========= Frontend Configuration =========
// Configure API endpoints and deployment settings

const CONFIG = {
  // API Base URL - Update this for different deployment environments
  API_BASE: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://127.0.0.1:8004'  // Local development (ANN version port)
    : '/api',                  // Production with proxy or same domain
  
  // Alternative configurations for different environments:
  // API_BASE: 'https://your-backend-domain.com',  // Remote backend
  // API_BASE: 'http://192.168.1.100:8000',        // LAN deployment
  
  // Application settings
  SESSION_STORAGE_KEY: 'medical_session_id',
  HISTORY_STORAGE_PREFIX: 'medical_history_',
  
  // UI Configuration
  AUTO_RESIZE_TEXTAREA: true,
  MAX_TEXTAREA_HEIGHT: 120,
  NOTIFICATION_TIMEOUT: 4000,
  TYPING_DELAY: 500,
  
  // Vietnamese IME handling
  COMPOSITION_DELAY: 10,
  INPUT_RESET_DELAY: 1,
  
  // API timeouts (in milliseconds)
  API_TIMEOUT: 30000,
  
  // Features
  FEATURES: {
    EXPORT_ENABLED: true,
    HISTORY_ENABLED: true,
    OFFLINE_SUPPORT: false
  }
};

// Make config globally available
window.MEDICAL_CHATBOT_CONFIG = CONFIG;

console.log('🏥 Medical Chatbot Configuration loaded:', CONFIG.API_BASE);