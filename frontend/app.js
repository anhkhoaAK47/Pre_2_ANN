// ========= Medical Chatbot Frontend - Hospital Interface =========
// Modern, accessible, hospital-appropriate UI for Vietnamese medical consultation

class MedicalChatbot {
  constructor() {
    // Get configuration from config.js
    const config = window.MEDICAL_CHATBOT_CONFIG || {};
    this.API_BASE = config.API_BASE || 'http://127.0.0.1:8003';
    this.config = config;
    this.sessionId = this.getOrCreateSession();
    this.isTyping = false;
    this.isDuplicate = false;

    // Conversation state tracking
    this.lastBotResponse = null;
    this.currentMissingSlots = [];
    
    // UI Elements
    this.chatBox = document.getElementById('chat-box');
    this.userInput = document.getElementById('user-input');
    this.sendBtn = document.getElementById('send-btn');
    this.resetBtn = document.getElementById('reset-btn');
    this.printRecordBtn = document.getElementById('print-record-btn');
    this.viewProfileBtn = document.getElementById('view-profile-btn');
    this.userIdDisplay = document.getElementById('user-id');
    this.filledSlots = document.getElementById('filled-slots');
    this.missingSlots = document.getElementById('missing-slots');
    this.nextAction = document.getElementById('next-action');
    this.progressFill = document.getElementById('progress-fill');
    this.progressText = document.getElementById('progress-text');

    // Vietnamese slot names mapping
    this.slotNames = {
      'name': '📝 Họ và tên',
      'phone_number': '📞 Số điện thoại',
      'symptoms': '🩺 Triệu chứng',
      'onset': '⏰ Thời gian xuất hiện',
      'age': '👤 Tuổi',
      'gender': '⚥ Giới tính',
      'allergies': '🚫 Dị ứng',
      'current_medications': '💊 Thuốc đang dùng',
      'pain_scale': '📊 Mức độ đau'
    };

    this.initializeUI();
    this.attachEventListeners();
    this.loadChatHistory();
  }

  // ========= Session Management =========
  getOrCreateSession() {
    const storageKey = this.config.SESSION_STORAGE_KEY || 'medical_session_id';
    let sessionId = localStorage.getItem(storageKey);
    if (!sessionId) {
      sessionId = this.generateSessionId();
      localStorage.setItem(storageKey, sessionId);
    }
    return sessionId;
  }

  generateSessionId() {
    const timestamp = Date.now().toString(36);
    const randomStr = Math.random().toString(36).substr(2, 5);
    return `u${timestamp}${randomStr}`;
  }

  resetSession() {
    const storageKey = this.config.SESSION_STORAGE_KEY || 'medical_session_id';
    const historyPrefix = this.config.HISTORY_STORAGE_PREFIX || 'medical_history_';
    localStorage.removeItem(storageKey);
    localStorage.removeItem(`${historyPrefix}${this.sessionId}`);
    this.sessionId = this.generateSessionId();
    localStorage.setItem(storageKey, this.sessionId);
    this.clearChatBox();
    this.updateUI();
    this.showWelcomeMessage();
  }

  // ========= UI Initialization =========
  initializeUI() {
    this.userIdDisplay.textContent = this.sessionId;
    this.updateProgressBar(0, 0);
    this.updateActionStatus('collecting', 'Đang thu thập thông tin bệnh nhân');
    this.updateRecordButtons();
  }

  showWelcomeMessage() {
    const welcomeMsg = {
      assistant_message: "🩺 Chào mừng bạn đến với Hệ thống tư vấn y tế AI\n\nTôi là trợ lý y tế thông minh, sẵn sàng hỗ trợ bạn:\n• Thu thập thông tin bệnh nhân\n• Phân tích triệu chứng\n• Đưa ra gợi ý chẩn đoán ban đầu\n• Định hướng khoa khám phù hợp\n\nVui lòng cho biết họ tên của bạn để bắt đầu...",
      filled_slots: {},
      missing_slots: ['name', 'phone_number', 'symptoms', 'onset', 'age', 'gender', 'allergies', 'current_medications', 'pain_scale'],
      next_action: 'ask_for_missing_slots'
    };
    this.updateUI(welcomeMsg);
    this.updateRecordButtons();
  }

  // ========= Event Listeners =========
  attachEventListeners() {
    // Initialize Vietnamese IME handling
    this.isComposing = false;
    
    // Setup input events with new method
    this.setupInputEvents();
    
    this.sendBtn.addEventListener('click', () => {
      // Also use requestAnimationFrame for button clicks
      requestAnimationFrame(() => {
        this.handleSendMessage();
      });
    });
    
    // Reset session
    this.resetBtn.addEventListener('click', () => this.resetSession());
    
    // Patient record functionality
    this.printRecordBtn.addEventListener('click', () => this.handlePrintRecord());
    this.viewProfileBtn.addEventListener('click', () => this.handleViewProfile());
  }

  autoResizeTextarea() {
    const textarea = this.userInput;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
  }

  async resetInputField() {
    console.log('Resetting input field completely');
    
    const oldInput = this.userInput;
    const container = oldInput.parentNode;
    
    // Create new input element with same attributes
    const newInput = document.createElement('textarea');
    newInput.id = 'user-input';
    newInput.className = oldInput.className;
    newInput.placeholder = oldInput.placeholder;
    newInput.rows = oldInput.rows;
    
    // Copy all attributes
    Array.from(oldInput.attributes).forEach(attr => {
      if (attr.name !== 'value') {
        newInput.setAttribute(attr.name, attr.value);
      }
    });
    
    // Replace old input with new one
    container.replaceChild(newInput, oldInput);
    this.userInput = newInput;
    
    // Re-attach event listeners to new input
    this.setupInputEvents();
    
    // Focus the new input
    await new Promise(resolve => setTimeout(resolve, 1));
    newInput.focus();
    this.autoResizeTextarea();
  }

  setupInputEvents() {
    // Enhanced Vietnamese IME composition events
    this.userInput.addEventListener('compositionstart', () => { 
      this.isComposing = true;
      console.log('Composition started');
    });
    
    this.userInput.addEventListener('compositionend', (e) => { 
      this.isComposing = false;
      console.log('Composition ended:', e.data);
      
      // Small delay to ensure composition is fully processed
      setTimeout(() => {
        this.isComposing = false;
      }, 10);
    });

    // Handle input changes
    this.userInput.addEventListener('input', () => {
      this.autoResizeTextarea();
    });

    // Simplified keydown handler
    this.userInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !this.isComposing) {
        e.preventDefault();
        requestAnimationFrame(() => {
          this.handleSendMessage();
        });
      }
    });
  }

  // ========= Chat Functionality =========
  async handleSendMessage() {
    const message = this.userInput.value.trim();
    console.log('handleSendMessage called, message:', message, 'isComposing:', this.isComposing);
    
    if (!message || this.isTyping || this.isDuplicate || this.isComposing) {
      console.log('Message blocked:', { message: !!message, isTyping: this.isTyping, isDuplicate: this.isDuplicate, isComposing: this.isComposing });
      return;
    }

    // Prevent duplicate sends
    this.isDuplicate = true;
    setTimeout(() => { this.isDuplicate = false; }, 500);

    // Complete Vietnamese IME fix: Replace input element
    await this.resetInputField();

    // Add user message to chat
    this.addMessage('user', message);
    this.saveToHistory('user', message);

    // Show typing indicator
    this.showTypingIndicator();

    try {
      const response = await this.sendToAPI(message);
      this.hideTypingIndicator();
      
      if (response.assistant_message) {
        this.addMessage('assistant', response.assistant_message);
        this.saveToHistory('assistant', response.assistant_message);
        this.updateUI(response);
      }
    } catch (error) {
      this.hideTypingIndicator();
      const errorMsg = `Xin lỗi, có lỗi xảy ra khi gọi API: ${error.message}`;
      this.addMessage('assistant', errorMsg);
      this.saveToHistory('assistant', errorMsg);
      console.error('API Error:', error);
    }
  }

  async sendToAPI(message) {
    const response = await fetch(`${this.API_BASE}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: this.sessionId,
        message: message
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  }

  // ========= UI Updates =========
  updateUI(response) {
    if (!response) return;

    // Update slots display
    this.updateSlotsDisplay(response.filled_slots || {}, response.missing_slots || []);
    
    // Update progress
    const totalSlots = 9;
    const filledCount = Object.keys(response.filled_slots || {}).length;
    this.updateProgressBar(filledCount, totalSlots);
    
    // Update action status
    if (response.next_action === 'call_phobert') {
      this.updateActionStatus('ready', 'Sẵn sàng phân tích - Tất cả thông tin đã được thu thập');
    } else if (response.next_action === 'final_confirmation') {
      this.updateActionStatus('diagnosis', 'Đã hoàn tất chẩn đoán - Có thể hỏi thêm hoặc kết thúc');
    } else if (response.next_action === 'session_complete') {
      this.updateActionStatus('completed', 'Phiên tư vấn đã hoàn tất - Sẵn sàng tạo phiên mới');
    } else {
      this.updateActionStatus('collecting', 'Đang thu thập thông tin bệnh nhân');
    }

    // Store current state for tracking
    this.currentMissingSlots = response.missing_slots || [];
    this.lastBotResponse = response.next_action;
  }

  updateSlotsDisplay(filledSlots, missingSlots) {
    // Update filled slots
    this.filledSlots.innerHTML = '';
    Object.entries(filledSlots).forEach(([key, value]) => {
      if (value && value.trim()) {
        const slotElement = this.createSlotElement(key, value, true);
        this.filledSlots.appendChild(slotElement);
      }
    });

    // Update missing slots
    this.missingSlots.innerHTML = '';
    missingSlots.forEach(slot => {
      const slotElement = this.createSlotElement(slot, null, false);
      this.missingSlots.appendChild(slotElement);
    });

    // Show empty state messages
    if (this.filledSlots.children.length === 0) {
      this.filledSlots.innerHTML = '<div class="slot-item" style="opacity: 0.6; font-style: italic;">Chưa có thông tin nào được thu thập</div>';
    }
    
    if (this.missingSlots.children.length === 0) {
      this.missingSlots.innerHTML = '<div class="slot-item filled" style="opacity: 0.8;">✅ Đã thu thập đủ thông tin</div>';
    }
  }

  createSlotElement(key, value, isFilled) {
    const div = document.createElement('div');
    div.className = `slot-item ${isFilled ? 'filled' : 'missing'}`;
    
    const slotName = this.slotNames[key] || key;
    const icon = isFilled ? 'fas fa-check-circle' : 'fas fa-clock';
    
    if (isFilled && value) {
      const truncatedValue = value.length > 30 ? value.substring(0, 30) + '...' : value;
      div.innerHTML = `
        <i class="${icon}"></i>
        <span><strong>${slotName}:</strong> ${truncatedValue}</span>
      `;
      div.title = `${slotName}: ${value}`;
    } else {
      div.innerHTML = `
        <i class="${icon}"></i>
        <span>${slotName}</span>
      `;
    }
    
    return div;
  }

  updateProgressBar(filled, total) {
    const percentage = total > 0 ? (filled / total) * 100 : 0;
    this.progressFill.style.width = `${percentage}%`;
    this.progressText.textContent = `${filled}/${total} thông tin đã thu thập`;
  }

  updateActionStatus(type, message) {
    this.nextAction.className = `action-status ${type}`;
    
    let icon = 'info-circle';
    if (type === 'ready') icon = 'check-circle';
    else if (type === 'diagnosis') icon = 'stethoscope';
    else if (type === 'completed') icon = 'flag-checkered';
    
    this.nextAction.innerHTML = `
      <i class="fas fa-${icon}"></i>
      ${message}
    `;
  }

  // ========= Message Display =========
  addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    // Format content with line breaks
    const formattedContent = this.formatMessageContent(content);
    messageContent.innerHTML = formattedContent;
    
    // Add timestamp
    const time = new Date().toLocaleTimeString('vi-VN', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = time;
    messageContent.appendChild(timeDiv);
    
    messageDiv.appendChild(messageContent);
    this.chatBox.appendChild(messageDiv);
    this.scrollToBottom();
  }

  formatMessageContent(content) {
    return content
      .replace(/\n/g, '<br>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>');
  }

  showTypingIndicator() {
    this.isTyping = true;
    this.sendBtn.disabled = true;
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant';
    typingDiv.id = 'typing-indicator';
    
    typingDiv.innerHTML = `
      <div class="typing-indicator">
        <span>Trợ lý đang soạn trả lời</span>
        <div class="typing-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    `;
    
    this.chatBox.appendChild(typingDiv);
    this.scrollToBottom();
  }

  hideTypingIndicator() {
    this.isTyping = false;
    this.sendBtn.disabled = false;
    
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
      typingIndicator.remove();
    }
  }

  scrollToBottom() {
    this.chatBox.scrollTop = this.chatBox.scrollHeight;
  }

  clearChatBox() {
    // Keep the initial welcome message
    this.chatBox.innerHTML = `
      <div class="message assistant">
        <div class="message-content">
          <strong>🩺 Chào mừng bạn đến với Hệ thống tư vấn y tế AI</strong><br><br>
          Tôi là trợ lý y tế thông minh, sẵn sàng hỗ trợ bạn:<br>
          • Thu thập thông tin triệu chứng<br>
          • Đưa ra gợi ý chẩn đoán ban đầu<br>
          • Định hướng khoa khám phù hợp<br><br>
          <em>Vui lòng mô tả triệu chứng của bạn để bắt đầu...</em>
        </div>
      </div>
    `;
  }

  // ========= History Management =========
  saveToHistory(role, message) {
    if (!this.config.FEATURES?.HISTORY_ENABLED) return;
    
    const historyPrefix = this.config.HISTORY_STORAGE_PREFIX || 'medical_history_';
    const historyKey = `${historyPrefix}${this.sessionId}`;
    const history = JSON.parse(localStorage.getItem(historyKey) || '[]');
    
    history.push({
      role,
      message,
      timestamp: Date.now()
    });
    
    localStorage.setItem(historyKey, JSON.stringify(history));
  }

  loadChatHistory() {
    if (!this.config.FEATURES?.HISTORY_ENABLED) {
      this.showWelcomeMessage();
      return;
    }
    
    const historyPrefix = this.config.HISTORY_STORAGE_PREFIX || 'medical_history_';
    const historyKey = `${historyPrefix}${this.sessionId}`;
    const history = JSON.parse(localStorage.getItem(historyKey) || '[]');
    
    if (history.length === 0) {
      this.showWelcomeMessage();
      return;
    }
    
    // Clear chat box before loading history
    this.clearChatBox();
    
    // Load history messages
    history.forEach(item => {
      if (item.role && item.message) {
        this.addMessage(item.role, item.message);
      }
    });
  }

  // ========= Export Functionality =========
  async handleExport() {
    try {
      const response = await fetch(`${this.API_BASE}/export_excel`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      // Get filename from response headers or use default
      const contentDisposition = response.headers.get('content-disposition');
      let filename = 'medical_report.csv';
      
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1].replace(/['"]/g, '');
        }
      }
      
      // Download file
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      // Show success message
      this.showNotification('✅ Báo cáo đã được tải xuống thành công!', 'success');
      
    } catch (error) {
      console.error('Export error:', error);
      this.showNotification('❌ Có lỗi xảy ra khi tải báo cáo: ' + error.message, 'error');
    }
  }

  // ========= Patient Record Functions =========
  async handlePrintRecord() {
    try {
      const patientData = await this.getCurrentPatientData();
      if (!patientData) {
        this.showNotification('Không tìm thấy dữ liệu bệnh nhân để in', 'error');
        return;
      }

      const printContent = this.generateA4PatientRecord(patientData);
      
      // Create temporary div for printing
      const printDiv = document.createElement('div');
      printDiv.className = 'print-record';
      printDiv.innerHTML = printContent;
      document.body.appendChild(printDiv);
      
      // Trigger print
      window.print();
      
      // Clean up
      setTimeout(() => {
        if (printDiv.parentNode) {
          document.body.removeChild(printDiv);
        }
      }, 100);
      
    } catch (error) {
      console.error('Print error:', error);
      this.showNotification('Lỗi khi in hồ sơ: ' + error.message, 'error');
    }
  }

  handleViewProfile() {
    try {
      const patientData = this.getCurrentPatientDataFromSlots();
      this.showProfileModal(patientData);
    } catch (error) {
      console.error('View profile error:', error);
      this.showNotification('Lỗi khi hiển thị hồ sơ: ' + error.message, 'error');
    }
  }

  isConsultationComplete() {
    // Check if consultation is complete based on the current state
    const lastResponse = this.getLastBotResponse();
    return lastResponse && (
      lastResponse.includes('chẩn đoán') ||
      lastResponse.includes('khoa khám') ||
      lastResponse.includes('đề xuất') ||
      (lastResponse.includes('🏥') && lastResponse.includes('📋'))
    );
  }

  getLastBotResponse() {
    const messages = this.chatBox.querySelectorAll('.message.assistant .message-content');
    return messages.length > 0 ? messages[messages.length - 1].textContent : '';
  }

  async getCurrentPatientData() {
    // Get current patient data from the chatbot state
    const historyPrefix = this.config.HISTORY_STORAGE_PREFIX || 'medical_history_';
    const history = localStorage.getItem(`${historyPrefix}${this.sessionId}`);
    
    if (!history) {
      return null;
    }

    try {
      const parsedHistory = JSON.parse(history);
      const lastEntry = parsedHistory[parsedHistory.length - 1];
      
      if (!lastEntry || !lastEntry.filled_slots) {
        return null;
      }

      return {
        record_id: this.sessionId,
        user_id: this.sessionId,
        patient_name: lastEntry.filled_slots.name || '',
        patient_phone: lastEntry.filled_slots.phone_number || '',
        patient_age: lastEntry.filled_slots.age || '',
        patient_gender: lastEntry.filled_slots.gender || '',
        symptoms: lastEntry.filled_slots.symptoms || '',
        onset: lastEntry.filled_slots.onset || '',
        allergies: lastEntry.filled_slots.allergies || 'không có',
        current_medications: lastEntry.filled_slots.current_medications || 'không có',
        pain_scale: lastEntry.filled_slots.pain_scale || '',
        predicted_diseases: lastEntry.predicted_diseases || {},
        recommended_department: lastEntry.recommended_department || 'Khám tổng quát',
        created_at: new Date().toISOString()
      };
    } catch (e) {
      console.error('Error parsing patient data:', e);
      return null;
    }
  }

  generateA4PatientRecord(patient) {
    const currentDate = new Date().toLocaleString('vi-VN');
    let content = `
      <div class="a4-document">
        <div class="a4-header">
          <div class="a4-logo">🏥</div>
          <div class="a4-title">HỆ THỐNG TƯ VẤN Y TẾ AI</div>
          <div class="a4-subtitle">HỒ SƠ BỆNH NHÂN</div>
        </div>
        
        <div class="a4-section">
          <div class="a4-section-title">THÔNG TIN HÀNH CHÍNH</div>
          <div class="a4-field">
            <div class="a4-field-label">Mã phiên tư vấn:</div>
            <div class="a4-field-value">${patient.record_id}</div>
          </div>
    `;
    
    if (patient.patient_name) {
      content += `
          <div class="a4-field">
            <div class="a4-field-label">Họ và tên:</div>
            <div class="a4-field-value">${patient.patient_name}</div>
          </div>
      `;
    }
    
    if (patient.patient_phone) {
      content += `
          <div class="a4-field">
            <div class="a4-field-label">Số điện thoại:</div>
            <div class="a4-field-value">${patient.patient_phone}</div>
          </div>
      `;
    }
    
    if (patient.patient_age) {
      content += `
          <div class="a4-field">
            <div class="a4-field-label">Tuổi:</div>
            <div class="a4-field-value">${patient.patient_age}</div>
          </div>
      `;
    }
    
    if (patient.patient_gender) {
      content += `
          <div class="a4-field">
            <div class="a4-field-label">Giới tính:</div>
            <div class="a4-field-value">${patient.patient_gender}</div>
          </div>
      `;
    }
    
    content += `</div>`;
    
    // Medical information
    content += `
        <div class="a4-section">
          <div class="a4-section-title">THÔNG TIN Y TẾ</div>
    `;
    
    if (patient.symptoms) {
      content += `
          <div class="a4-field">
            <div class="a4-field-label">Triệu chứng:</div>
            <div class="a4-field-value">${patient.symptoms}</div>
          </div>
      `;
    }
    
    if (patient.onset) {
      content += `
          <div class="a4-field">
            <div class="a4-field-label">Thời gian xuất hiện:</div>
            <div class="a4-field-value">${patient.onset}</div>
          </div>
      `;
    }
    
    if (patient.allergies && patient.allergies !== 'không có') {
      content += `
          <div class="a4-field">
            <div class="a4-field-label">Dị ứng:</div>
            <div class="a4-field-value">${patient.allergies}</div>
          </div>
      `;
    }
    
    if (patient.current_medications && patient.current_medications !== 'không có') {
      content += `
          <div class="a4-field">
            <div class="a4-field-label">Thuốc đang dùng:</div>
            <div class="a4-field-value">${patient.current_medications}</div>
          </div>
      `;
    }
    
    if (patient.pain_scale) {
      content += `
          <div class="a4-field">
            <div class="a4-field-label">Mức độ đau:</div>
            <div class="a4-field-value">${patient.pain_scale}</div>
          </div>
      `;
    }
    
    content += `</div>`;
    
    // Diagnosis and recommendations
    content += `
        <div class="a4-section">
          <div class="a4-section-title">KẾT QUẢ CHẨN ĐOÁN</div>
    `;
    
    if (patient.predicted_diseases && Object.keys(patient.predicted_diseases).length > 0) {
      content += `
          <div class="a4-field">
            <div class="a4-field-label">Dự đoán bệnh:</div>
            <div class="a4-field-value">
      `;
      
      Object.entries(patient.predicted_diseases).forEach(([disease, confidence]) => {
        content += `• ${disease}: ${(confidence * 100).toFixed(1)}%<br>`;
      });
      
      content += `
            </div>
          </div>
      `;
    }
    
    content += `
          <div class="a4-field">
            <div class="a4-field-label">Khoa khám được gợi ý:</div>
            <div class="a4-field-value"><strong>${patient.recommended_department}</strong></div>
          </div>
        </div>
    `;
    
    // Footer
    content += `
        <div class="a4-footer">
          <p><strong>Lưu ý:</strong> Đây là kết quả hỗ trợ chẩn đoán ban đầu từ hệ thống AI.</p>
          <p>Vui lòng đến bệnh viện để được thăm khám chính xác bởi bác sĩ chuyên khoa.</p>
          <p>Báo cáo được tạo lúc: ${currentDate}</p>
        </div>
      </div>
    `;
    
    return content;
  }

  getA4PrintStyles() {
    return `
      * { margin: 0; padding: 0; box-sizing: border-box; }
      
      .a4-document {
        width: 210mm;
        min-height: 297mm;
        padding: 20mm;
        margin: 0 auto;
        background: white;
        font-family: 'Times New Roman', serif;
        font-size: 12pt;
        line-height: 1.5;
        color: #000;
      }
      
      .a4-header {
        text-align: center;
        border-bottom: 2px solid #0066cc;
        padding-bottom: 15px;
        margin-bottom: 25px;
      }
      
      .a4-logo { font-size: 24pt; color: #0066cc; margin-bottom: 10px; }
      .a4-title { font-size: 18pt; font-weight: bold; color: #333; margin: 10px 0; }
      .a4-subtitle { font-size: 14pt; color: #666; margin-bottom: 20px; }
      
      .a4-section { margin-bottom: 20px; }
      .a4-section-title {
        font-size: 14pt;
        font-weight: bold;
        color: #0066cc;
        border-bottom: 1px solid #ddd;
        padding-bottom: 5px;
        margin-bottom: 10px;
      }
      
      .a4-field {
        margin-bottom: 8px;
        display: flex;
        align-items: flex-start;
      }
      
      .a4-field-label {
        font-weight: bold;
        min-width: 140px;
        color: #333;
      }
      
      .a4-field-value {
        flex: 1;
        padding-left: 10px;
      }
      
      .a4-footer {
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
        text-align: center;
        font-size: 10pt;
        color: #666;
      }
      
      @media print {
        .a4-document {
          box-shadow: none;
          margin: 0;
          width: 100%;
          min-height: auto;
        }
      }
    `;
  }

  updateRecordButtons() {
    // Enable print button always (removed consultation completion requirement)
    this.printRecordBtn.disabled = false;
    this.printRecordBtn.style.opacity = '1';
  }

  // ========= Profile Viewing Functions =========
  async handleViewProfile() {
    try {
      const patientData = await this.getCurrentPatientDataFromSlots();
      this.showProfileModal(patientData);
    } catch (error) {
      console.error('View profile error:', error);
      this.showNotification('Lỗi khi hiển thị hồ sơ: ' + error.message, 'error');
    }
  }

  async getCurrentPatientDataFromSlots() {
    // First try to get data from filled slots display
    const filledSlotsContainer = document.getElementById('filled-slots');
    const patientData = {
      record_id: this.sessionId,
      patient_name: '',
      patient_phone: '',
      patient_age: '',
      patient_gender: '',
      symptoms: '',
      onset: '',
      allergies: '',
      current_medications: '',
      pain_scale: '',
      predicted_diseases: {},
      recommended_department: ''
    };

    // Extract data from filled slots display
    if (filledSlotsContainer) {
      const slotItems = filledSlotsContainer.querySelectorAll('.slot-item');
      slotItems.forEach(item => {
        const label = item.querySelector('.slot-label')?.textContent || '';
        const value = item.querySelector('.slot-value')?.textContent || '';

        // Map Vietnamese labels to data fields
        if (label.includes('Họ và tên')) patientData.patient_name = value;
        else if (label.includes('Số điện thoại')) patientData.patient_phone = value;
        else if (label.includes('Tuổi')) patientData.patient_age = value;
        else if (label.includes('Giới tính')) patientData.patient_gender = value;
        else if (label.includes('Triệu chứng')) patientData.symptoms = value;
        else if (label.includes('Thời gian xuất hiện')) patientData.onset = value;
        else if (label.includes('Dị ứng')) patientData.allergies = value;
        else if (label.includes('Thuốc đang dùng')) patientData.current_medications = value;
        else if (label.includes('Mức độ đau')) patientData.pain_scale = value;
      });
    }

    // If no data from slots, try localStorage
    if (!patientData.patient_name && !patientData.symptoms) {
      const historyPrefix = this.config.HISTORY_STORAGE_PREFIX || 'medical_history_';
      const history = localStorage.getItem(`${historyPrefix}${this.sessionId}`);

      if (history) {
        try {
          const parsedHistory = JSON.parse(history);
          const lastEntry = parsedHistory[parsedHistory.length - 1];

          if (lastEntry && lastEntry.filled_slots) {
            patientData.patient_name = lastEntry.filled_slots.name || '';
            patientData.patient_phone = lastEntry.filled_slots.phone_number || '';
            patientData.patient_age = lastEntry.filled_slots.age || '';
            patientData.patient_gender = lastEntry.filled_slots.gender || '';
            patientData.symptoms = lastEntry.filled_slots.symptoms || '';
            patientData.onset = lastEntry.filled_slots.onset || '';
            patientData.allergies = lastEntry.filled_slots.allergies || '';
            patientData.current_medications = lastEntry.filled_slots.current_medications || '';
            patientData.pain_scale = lastEntry.filled_slots.pain_scale || '';
          }
        } catch (e) {
          console.error('Error parsing history:', e);
        }
      }
    }

    // Try to fetch completed consultation data from backend
    try {
      const response = await fetch(`${this.API_BASE}/patients`);
      if (response.ok) {
        const patients = await response.json();
        // Find current session's completed record
        const currentRecord = patients.find(p => p.user_id === this.sessionId);
        if (currentRecord) {
          // Merge with backend data (backend data takes priority for completed consultations)
          patientData.patient_name = currentRecord.patient_name || patientData.patient_name;
          patientData.patient_phone = currentRecord.patient_phone || patientData.patient_phone;
          patientData.patient_age = currentRecord.patient_age || patientData.patient_age;
          patientData.patient_gender = currentRecord.patient_gender || patientData.patient_gender;
          patientData.symptoms = currentRecord.symptoms || patientData.symptoms;
          patientData.onset = currentRecord.onset || patientData.onset;
          patientData.allergies = currentRecord.allergies || patientData.allergies;
          patientData.current_medications = currentRecord.current_medications || patientData.current_medications;
          patientData.pain_scale = currentRecord.pain_scale || patientData.pain_scale;
          patientData.predicted_diseases = currentRecord.predicted_diseases || {};
          patientData.recommended_department = currentRecord.recommended_department || '';
        }
      }
    } catch (error) {
      console.log('Could not fetch backend data, using frontend data only:', error);
    }

    return patientData;
  }

  showProfileModal(patientData) {
    const modal = document.getElementById('profile-modal');
    const modalBody = document.getElementById('profile-modal-body');
    
    if (!modal || !modalBody) {
      this.showNotification('Không tìm thấy modal hiển thị hồ sơ', 'error');
      return;
    }

    // Generate profile content
    let content = '<div class="profile-section">';
    
    // Administrative information
    content += '<div class="profile-section-title">Thông tin hành chính</div>';
    content += `<div class="profile-field"><span class="profile-label">Mã phiên:</span> <span class="profile-value">${patientData.record_id}</span></div>`;
    
    if (patientData.patient_name) content += `<div class="profile-field"><span class="profile-label">Họ và tên:</span> <span class="profile-value">${patientData.patient_name}</span></div>`;
    if (patientData.patient_phone) content += `<div class="profile-field"><span class="profile-label">Số điện thoại:</span> <span class="profile-value">${patientData.patient_phone}</span></div>`;
    if (patientData.patient_age) content += `<div class="profile-field"><span class="profile-label">Tuổi:</span> <span class="profile-value">${patientData.patient_age}</span></div>`;
    if (patientData.patient_gender) content += `<div class="profile-field"><span class="profile-label">Giới tính:</span> <span class="profile-value">${patientData.patient_gender}</span></div>`;
    
    content += '</div>';
    
    // Medical information
    content += '<div class="profile-section">';
    content += '<div class="profile-section-title">Thông tin y tế</div>';
    
    if (patientData.symptoms) content += `<div class="profile-field"><span class="profile-label">Triệu chứng:</span> <span class="profile-value">${patientData.symptoms}</span></div>`;
    if (patientData.onset) content += `<div class="profile-field"><span class="profile-label">Thời gian xuất hiện:</span> <span class="profile-value">${patientData.onset}</span></div>`;
    if (patientData.allergies && patientData.allergies !== 'không có') content += `<div class="profile-field"><span class="profile-label">Dị ứng:</span> <span class="profile-value">${patientData.allergies}</span></div>`;
    if (patientData.current_medications && patientData.current_medications !== 'không có') content += `<div class="profile-field"><span class="profile-label">Thuốc đang dùng:</span> <span class="profile-value">${patientData.current_medications}</span></div>`;
    if (patientData.pain_scale) content += `<div class="profile-field"><span class="profile-label">Mức độ đau:</span> <span class="profile-value">${patientData.pain_scale}</span></div>`;
    
    content += '</div>';
    
    // Diagnosis results (if available)
    if (patientData.predicted_diseases && Object.keys(patientData.predicted_diseases).length > 0) {
      content += '<div class="profile-section">';
      content += '<div class="profile-section-title">Kết quả chẩn đoán</div>';
      
      content += '<div class="profile-field"><span class="profile-label">Dự đoán bệnh:</span> <div class="profile-value">';
      Object.entries(patientData.predicted_diseases).forEach(([disease, confidence]) => {
        content += `• ${disease}: ${(confidence * 100).toFixed(1)}%<br>`;
      });
      content += '</div></div>';
      
      if (patientData.recommended_department) {
        content += `<div class="profile-field"><span class="profile-label">Khoa khám gợi ý:</span> <span class="profile-value"><strong>${patientData.recommended_department}</strong></span></div>`;
      }
      
      content += '</div>';
    }
    
    // Show empty state if no data
    if (!patientData.patient_name && !patientData.symptoms) {
      content = `<div class="profile-empty">
        <h3>Chưa có thông tin hồ sơ</h3>
        <p>Vui lòng nhập thông tin cơ bản trước khi xem hồ sơ.</p>
        <p><strong>Mã phiên:</strong> ${patientData.record_id}</p>
        <p><em>Hệ thống đang thu thập thông tin của bạn. Vui lòng trả lời các câu hỏi để hoàn tất hồ sơ.</em></p>
      </div>`;
    }
    
    modalBody.innerHTML = content;
    modal.style.display = 'block';
    
    // Setup modal close functionality
    this.setupProfileModalEvents(modal);
  }

  setupProfileModalEvents(modal) {
    // Close button
    const closeBtn = modal.querySelector('.profile-close');
    if (closeBtn) {
      closeBtn.onclick = () => {
        modal.style.display = 'none';
      };
    }
    
    // Click outside to close
    modal.onclick = (e) => {
      if (e.target === modal) {
        modal.style.display = 'none';
      }
    };
    
    // Escape key to close
    const escHandler = (e) => {
      if (e.key === 'Escape') {
        modal.style.display = 'none';
        document.removeEventListener('keydown', escHandler);
      }
    };
    document.addEventListener('keydown', escHandler);
  }

  // ========= Print Record Function =========
  async handlePrintRecord() {
    try {
      const patientData = await this.getCurrentPatientDataFromSlots();
      
      // Check if we have any patient data
      if (!patientData.patient_name && !patientData.symptoms) {
        this.showNotification('Chưa có đủ thông tin để in hồ sơ', 'error');
        return;
      }
      
      // Generate print-friendly content
      const printContent = this.generateA4PatientRecord(patientData);
      const printStyles = this.getA4PrintStyles();
      
      // Create print window
      const printWindow = window.open('', '_blank');
      printWindow.document.write(`
        <!DOCTYPE html>
        <html>
          <head>
            <title>Hồ sơ bệnh nhân - ${patientData.patient_name || patientData.record_id}</title>
            <style>${printStyles}</style>
          </head>
          <body>
            ${printContent}
          </body>
        </html>
      `);
      
      printWindow.document.close();
      printWindow.focus();
      
      // Print after a short delay to ensure content is loaded
      setTimeout(() => {
        printWindow.print();
        printWindow.close();
      }, 500);
      
      this.showNotification('✅ Hồ sơ đã được chuẩn bị để in', 'success');
      
    } catch (error) {
      console.error('Print error:', error);
      this.showNotification('Lỗi khi in hồ sơ: ' + error.message, 'error');
    }
  }

  // ========= Notifications =========
  showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
      color: white;
      padding: 12px 20px;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      z-index: 1000;
      max-width: 400px;
      font-size: 14px;
      line-height: 1.4;
      animation: slideInRight 0.3s ease-out;
    `;
    notification.textContent = message;
    
    // Add animation styles
    if (!document.getElementById('notification-styles')) {
      const style = document.createElement('style');
      style.id = 'notification-styles';
      style.textContent = `
        @keyframes slideInRight {
          from { transform: translateX(100%); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOutRight {
          from { transform: translateX(0); opacity: 1; }
          to { transform: translateX(100%); opacity: 0; }
        }
      `;
      document.head.appendChild(style);
    }
    
    document.body.appendChild(notification);
    
    // Auto-remove after 4 seconds
    setTimeout(() => {
      notification.style.animation = 'slideOutRight 0.3s ease-out forwards';
      setTimeout(() => {
        if (notification.parentNode) {
          notification.parentNode.removeChild(notification);
        }
      }, 300);
    }, 4000);
  }

}

// ========= Initialize Application =========
document.addEventListener('DOMContentLoaded', () => {
  const chatbot = new MedicalChatbot();
  
  // Global error handler
  window.addEventListener('error', (e) => {
    console.error('Application error:', e.error);
    chatbot.showNotification('Có lỗi xảy ra trong ứng dụng. Vui lòng thử lại.', 'error');
  });
  
  // Handle online/offline status
  window.addEventListener('online', () => {
    chatbot.showNotification('✅ Đã khôi phục kết nối internet', 'success');
  });
  
  window.addEventListener('offline', () => {
    chatbot.showNotification('⚠️ Mất kết nối internet', 'error');
  });
  
  console.log('🏥 Medical Chatbot System initialized successfully');
});