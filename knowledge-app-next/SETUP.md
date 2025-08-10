# 🚀 Knowledge App Next.js Setup Guide

## **Quick Start (Automated)**

### **Option 1: Use the Automated Startup Script**
```bash
# Windows
start-full-app.bat

# This will automatically:
# 1. Start the Python bridge server (port 8000)
# 2. Start the Next.js development server (port 3000)
```

### **Option 2: Manual Setup**

#### **Step 1: Install Python Dependencies**
```bash
# Install Python bridge requirements
npm run bridge:install

# Or manually:
pip install -r api-server/requirements.txt
```

#### **Step 2: Install Node.js Dependencies**
```bash
# Install Next.js dependencies
npm install
```

#### **Step 3: Start Both Servers**

**Terminal 1 - Python Bridge Server:**
```bash
# Windows
start-bridge.bat

# Or manually:
python api-server/bridge-server.py
```

**Terminal 2 - Next.js Development Server:**
```bash
npm run dev:next
```

## **What's Running**

- **Next.js Frontend**: http://localhost:3000
- **Python Bridge Server**: http://localhost:8000
- **WebSocket Connection**: Real-time communication between frontend and backend

## **Connection Status**

The app will show a connection status indicator in the top-right corner:
- 🟢 **Connected**: Everything working properly
- 🟡 **Bridge Issues**: Connected but backend has problems
- 🔴 **Disconnected**: Cannot connect to Python bridge

## **Troubleshooting**

### **"Cannot connect to Python backend" Error**
1. Make sure the Python bridge server is running on port 8000
2. Check that all Python dependencies are installed
3. Verify your existing Knowledge App Python modules are accessible

### **Python Import Errors**
1. Make sure you're running from the project root directory
2. Check that your existing `src/knowledge_app/` modules are present
3. Install any missing Python dependencies

### **Port Conflicts**
```bash
# Clear any stuck ports
npm run clear-ports
```

## **Development Workflow**

1. **Start the servers** using `start-full-app.bat` or manually
2. **Open http://localhost:3000** in your browser
3. **Check connection status** in the top-right corner
4. **Generate quizzes** - they will now use your real Python backend!
5. **View question history** - loads from your actual question database

## **Features Now Working**

✅ **Quiz Generation**: Uses your existing MCQ generation system
✅ **Question History**: Loads from your question database
✅ **Real-time Updates**: WebSocket communication for live features
✅ **File Uploads**: For training your models
✅ **Settings Management**: Connects to your backend configuration
✅ **Connection Monitoring**: Shows backend status in real-time

## **Next Steps**

The Next.js frontend is now fully connected to your existing Python backend. You can:

1. **Generate real quizzes** using your trained models
2. **View actual question history** from your database
3. **Upload files for training** through the web interface
4. **Manage settings** that persist to your backend

Your Qt WebEngine → Next.js migration is now complete and functional! 🎉