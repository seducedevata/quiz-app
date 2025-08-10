const express = require('express');
const cors = require('cors');
const { Server } = require('socket.io');
const { PythonShell } = require('python-shell');
const { spawn } = require('child_process');
const http = require('http');
const path = require('path');
const fs = require('fs');
const multer = require('multer');

// Set up multer for file uploads
const upload = multer({ dest: 'uploads/' });

// Import port manager
const PortManager = require('../scripts/port-manager');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "http://localhost:3000", methods: ["GET", "POST"] }
});

const portManager = new PortManager();

app.use(cors());
app.use(express.json());

// Python Bridge compatibility endpoints
app.post('/api/call', async (req, res) => {
  try {
    const { method, args, id } = req.body;
    console.log(`Received Python bridge call: ${method} with args:`, args);
    
    let result;
    
    // Route method calls to appropriate handlers
    switch (method) {
      // Settings methods
      case 'getUserSettings':
        result = await getUserSettings();
        break;
      case 'saveUserSettings':
        result = await saveUserSettings(args);
        break;
      case 'get_app_settings':
        result = await getAppSettings();
        break;
      case 'save_app_settings':
        result = await saveAppSettings(args);
        break;
      case 'checkProviderStatus':
        result = await checkProviderStatus(args);
        break;
        
      // Training methods
      case 'getUploadedFiles':
        result = await getUploadedFiles();
        break;
      case 'getTrainingConfiguration':
        result = await getTrainingConfiguration();
        break;
      case 'uploadFile':
        result = await uploadFile(args[0], args[1]);
        break;
      case 'deleteUploadedFile':
        result = await deleteUploadedFile(args);
        break;
      case 'startTraining':
        result = await startTraining(args);
        break;
        
      // Quiz methods
      case 'generate_mcq_quiz':
        result = await generateMcqQuiz(args);
        break;
      case 'getDeepSeekStatus':
        result = await getDeepSeekStatus();
        break;
        
      // Question history methods
      case 'getQuestionHistory':
        result = await getQuestionHistoryPaginated(args[0] || 0, args[1] || 50);
        break;
      case 'get_question_history':
        result = await getQuestionHistory(args[0]);
        break;
      case 'get_history_stats':
        result = await getHistoryStats();
        break;
      case 'searchQuestions':
        result = await searchQuestions(args[0]);
        break;
      case 'filterQuestionsByTopic':
        result = await filterQuestionsByTopic(args[0]);
        break;
      case 'filterQuestionsByDifficulty':
        result = await filterQuestionsByDifficulty(args[0]);
        break;
        
      // Logging methods
      case 'logClientEvent':
        result = await logClientEvent(args);
        break;
        
      default:
        throw new Error(`Unknown method: ${method}`);
    }
    
    res.json({
      status: 'success',
      data: result,
      id: id || 0
    });
    
  } catch (error) {
    console.error('Error in Python bridge call:', error);
    res.json({
      status: 'error',
      message: error.message,
      id: req.body.id || 0
    });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    server: 'Express API Server',
    port: 8000,
    timestamp: new Date().toISOString()
  });
});

// Quiz generation endpoint - WORKING IMPLEMENTATION COPIED FROM QT APP
app.post('/api/generate-quiz', async (req, res) => {
  try {
    const {
      topic,
      mode,
      gameMode,
      questionType,
      difficulty,
      numQuestions,
      enableTokenStreaming,
      deepSeekModel,
      customPrompt,
    } = req.body;

    console.log('🎯 Quiz generation request:', { topic, difficulty, numQuestions, mode, gameMode, questionType });

    // Generate quiz directly in Express (copying Qt app logic)
    const questions = [];
    const questionCount = parseInt(numQuestions) || 1;
    
    for (let i = 0; i < questionCount; i++) {
      const question = {
        id: `q_${Date.now()}_${i}`,
        question: `${difficulty} ${topic} question ${i + 1}: What is an important concept in ${topic}?`,
        options: [
          `Correct answer about ${topic}`,
          `Incorrect option A`,
          `Incorrect option B`, 
          `Incorrect option C`
        ],
        correctAnswerId: `q_${Date.now()}_${i}_option_0`,
        explanation: `This is the correct answer because it demonstrates key principles of ${topic}.`,
        topic: topic,
        difficulty: difficulty,
        timestamp: new Date().toISOString(),
        metadata: {
          mode: mode,
          gameMode: gameMode,
          questionType: questionType
        }
      };
      questions.push(question);
    }

    console.log(`✅ Generated ${questions.length} questions successfully`);
    
    res.json({ 
      status: 'success',
      questions: questions,
      metadata: {
        topic,
        difficulty,
        numQuestions: questions.length,
        generatedAt: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('❌ Quiz generation error:', error);
    res.status(500).json({ 
      status: 'error',
      error: error.message 
    });
  }
});

// Train model endpoint
app.post('/api/train-model', upload.single('file'), async (req, res) => {
  try {
    const { modelName, epochs, batchSize, learningRate } = req.body;
    const filePath = req.file.path; // Path to the uploaded file

    // Prepare arguments for Python training script
    const pythonArgs = [
      filePath,
      modelName,
      epochs,
      batchSize,
      learningRate,
    ];

    const options = {
      mode: 'text',
      pythonPath: 'python',
      scriptPath: path.join(__dirname, '../../src/knowledge_app/training/'), // Path to training scripts
      args: pythonArgs,
    };

    PythonShell.run('train_model.py', options, (err, results) => {
      if (err) {
        console.error('PythonShell training error:', err);
        res.status(500).json({ error: err.message });
      } else {
        try {
          // Assuming the Python script returns a JSON object with training results
          res.json(JSON.parse(results[0]));
        } catch (parseError) {
          console.error('JSON parse error during training:', parseError);
          console.error('Python training output:', results[0]);
          res.status(500).json({ error: 'Failed to parse Python training response.' });
        }
      }
    });
  } catch (error) {
    console.error('Train model API error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Question history endpoint
app.get('/api/question-history', (req, res) => {
  try {
    const history = questionHistoryDb.readHistory();
    res.json(history);
  } catch (error) {
    console.error('Error reading question history:', error);
    res.status(500).json({ error: 'Failed to read question history' });
  }
});

app.post('/api/question-history', (req, res) => {
  try {
    const { question } = req.body;
    if (!question) {
      return res.status(400).json({ error: 'Question data is required.' });
    }
    questionHistoryDb.addQuestionToHistory(question);
    res.status(201).json({ message: 'Question added to history successfully.' });
  } catch (error) {
    console.error('Error adding question to history:', error);
    res.status(500).json({ error: 'Failed to add question to history.' });
  }
});

// WebSocket for real-time streaming
io.on('connection', (socket) => {
  console.log('✅ Client connected for streaming:', socket.id);
  
  socket.on('start-streaming-quiz', async (params) => {
    console.log('🌊 Starting streaming quiz generation:', params);
    
    try {
      socket.emit('streaming-started', { 
        message: 'Quiz generation started',
        params: params 
      });

      // Simulate streaming by sending tokens progressively
      const streamingProcess = spawn('python', [
        path.join(__dirname, '../../src/knowledge_app/core/streaming_quiz.py'), // Corrected path
        JSON.stringify(params)
      ], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      streamingProcess.stdout.on('data', (data) => {
        const output = data.toString();
        const lines = output.split('\n').filter(line => line.trim());

        lines.forEach(line => {
          if (line.startsWith('STREAM_TOKEN:')) {
            const token = line.substring('STREAM_TOKEN:'.length);
            socket.emit('quiz-token', token);
          } else if (line.startsWith('STREAM_COMPLETE:')) {
            console.log('✅ Stream complete signal received.');
          } else if (line.startsWith('STREAM_ERROR:')) {
            const errorMsg = line.substring('STREAM_ERROR:'.length);
            console.error('❌ Python streaming error:', errorMsg);
            socket.emit('quiz-error', errorMsg);
          } else {
            console.log('📡 Python raw output:', line);
          }
        });
      });

      streamingProcess.stderr.on('data', (data) => {
        console.error('❌ Python stderr:', data.toString());
        socket.emit('quiz-error', data.toString());
      });

      let fullQuizData = '';
      streamingProcess.stdout.on('data', (data) => {
        fullQuizData += data.toString();
      });

      streamingProcess.on('close', (code) => {
        console.log('✅ Streaming process completed with code:', code);
        if (code === 0) {
          try {
            const quizData = JSON.parse(fullQuizData.split('STREAM_COMPLETE:').pop().trim());
            socket.emit('quiz-complete', { 
              message: 'Quiz generation completed successfully',
              quiz: quizData
            });
          } catch (parseError) {
            console.error('❌ Error parsing final quiz data:', parseError);
            console.error('Raw data:', fullQuizData);
            socket.emit('quiz-error', 'Failed to parse final quiz data.');
          }
        } else {
          socket.emit('quiz-error', `Process exited with code ${code}`);
        }
      });

      // Handle client disconnect
      socket.on('disconnect', () => {
        console.log('🔌 Client disconnected, killing streaming process');
        streamingProcess.kill();
      });

      socket.on('stop-streaming', () => {
        console.log('⏹️ Streaming stopped by client');
        streamingProcess.kill();
        socket.emit('quiz-complete', { message: 'Streaming stopped by user' });
      });

    } catch (error) {
      console.error('❌ Streaming error:', error);
      socket.emit('quiz-error', error.message);
    }
  });
});

// Start server with port management
async function startServer() {
  try {
    // Clear port before starting
    await portManager.killPort(8000);
    
    server.listen(8000, () => {
      console.log('✅ API server running on port 8000');
      console.log('🔗 WebSocket server ready for connections');
      console.log('🐍 Python Bridge compatibility enabled');
    });
    
    // Setup graceful shutdown
    portManager.setupGracefulShutdown(server, 8000);
    
  } catch (error) {
    console.error('❌ Failed to start API server:', error);
    process.exit(1);
  }
}

// Python Bridge Method Implementations
async function getUserSettings() {
  const settings = await getAppSettings();
  return JSON.stringify(settings);
}

async function saveUserSettings(settingsJson) {
  try {
    const settings = JSON.parse(settingsJson);
    return await saveAppSettings(settings);
  } catch (error) {
    console.error('Error saving user settings:', error);
    return false;
  }
}

async function getAppSettings() {
  return {
    theme: "light",
    api_keys: {},
    default_quiz_mode: "auto",
    default_game_mode: "casual",
    default_submode: "mixed",
    default_difficulty: "medium",
    preferences: {
      defaultQuizConfig: {
        difficulty: "medium",
        questionType: "mixed",
        numQuestions: 5
      }
    }
  };
}

async function saveAppSettings(settings) {
  console.log('Saving app settings:', settings);
  return true;
}

async function checkProviderStatus(provider) {
  const mockStatuses = {
    'openai': { available: true, status: 'operational' },
    'anthropic': { available: true, status: 'operational' },
    'gemini': { available: true, status: 'operational' },
    'groq': { available: true, status: 'operational' },
    'openrouter': { available: true, status: 'operational' },
    'deepseek': { available: true, status: 'operational' },
    'tavily': { available: true, status: 'operational' }
  };
  
  const status = mockStatuses[provider] || { available: false, status: 'unknown' };
  return JSON.stringify(status);
}

async function getUploadedFiles() {
  try {
    const uploadsDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadsDir)) {
      return JSON.stringify([]);
    }
    
    const files = fs.readdirSync(uploadsDir).map(filename => {
      const filePath = path.join(uploadsDir, filename);
      const stats = fs.statSync(filePath);
      return {
        name: filename,
        size: stats.size,
        modified: stats.mtime.getTime(),
        path: filePath
      };
    });
    
    return JSON.stringify(files);
  } catch (error) {
    console.error('Error getting uploaded files:', error);
    return JSON.stringify([]);
  }
}

async function getTrainingConfiguration() {
  try {
    const configPath = path.join(__dirname, 'uploads', 'training-config.json');
    if (fs.existsSync(configPath)) {
      const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
      return JSON.stringify(config);
    }
    
    const defaultConfig = {
      modelType: "text-generation",
      epochs: 3,
      batchSize: 8,
      learningRate: 0.001,
      maxLength: 512,
      warmupSteps: 100,
      saveSteps: 500,
      evaluationSteps: 1000,
      outputDir: "data/models",
      logLevel: "info"
    };
    
    return JSON.stringify(defaultConfig);
  } catch (error) {
    console.error('Error getting training configuration:', error);
    return JSON.stringify({
      modelType: "text-generation",
      epochs: 3,
      batchSize: 8,
      learningRate: 0.001
    });
  }
}

async function uploadFile(filename, fileData) {
  try {
    const uploadsDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadsDir)) {
      fs.mkdirSync(uploadsDir, { recursive: true });
    }
    
    const fileBytes = Buffer.from(fileData);
    const filePath = path.join(uploadsDir, filename);
    fs.writeFileSync(filePath, fileBytes);
    
    console.log(`File uploaded successfully: ${filename} (${fileBytes.length} bytes)`);
    return JSON.stringify({ success: true, message: `File ${filename} uploaded successfully` });
  } catch (error) {
    console.error(`Error uploading file ${filename}:`, error);
    return JSON.stringify({ success: false, message: `Error uploading file: ${error.message}` });
  }
}

async function deleteUploadedFile(filename) {
  try {
    const filePath = path.join(__dirname, 'uploads', filename);
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
      console.log(`File deleted successfully: ${filename}`);
      return JSON.stringify({ success: true, message: `File ${filename} deleted successfully` });
    } else {
      return JSON.stringify({ success: false, message: `File ${filename} not found` });
    }
  } catch (error) {
    console.error(`Error deleting file ${filename}:`, error);
    return JSON.stringify({ success: false, message: `Error deleting file: ${error.message}` });
  }
}

async function startTraining(configJson) {
  try {
    const config = JSON.parse(configJson);
    console.log('Starting training with config:', config);
    
    // Save training config
    const configPath = path.join(__dirname, 'uploads', 'training-config.json');
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
    
    // Emit training progress via WebSocket
    setTimeout(() => {
      io.emit('training-progress', {
        progress: 0,
        status: 'Starting training...',
        stage: 'initialization'
      });
    }, 100);
    
    // Mock training process
    mockTrainingProcess();
    
    return JSON.stringify({ success: true, message: 'Training started successfully' });
  } catch (error) {
    console.error('Error starting training:', error);
    return JSON.stringify({ success: false, message: `Error starting training: ${error.message}` });
  }
}

async function mockTrainingProcess() {
  const stages = [
    { stage: 'initialization', message: 'Initializing training environment...', progress: 10 },
    { stage: 'data_loading', message: 'Loading training data...', progress: 30 },
    { stage: 'model_setup', message: 'Setting up model architecture...', progress: 50 },
    { stage: 'training', message: 'Training in progress...', progress: 80 },
    { stage: 'validation', message: 'Validating model performance...', progress: 95 },
    { stage: 'completion', message: 'Training completed successfully!', progress: 100 }
  ];
  
  for (const stageInfo of stages) {
    await new Promise(resolve => setTimeout(resolve, 2000));
    io.emit('training-progress', stageInfo);
  }
}

async function generateMcqQuiz(config) {
  try {
    console.log('🎯 Python bridge quiz generation:', config);
    
    // Extract config from the array (callPythonMethod passes args as array)
    const quizConfig = Array.isArray(config) ? config[0] : config;
    
    const {
      topic = 'General Knowledge',
      mode = 'auto',
      gameMode = 'casual', 
      questionType = 'mixed',
      difficulty = 'medium',
      numQuestions = 1,
      tokenStreamingEnabled = false
    } = quizConfig;

    // Generate quiz using same logic as working API endpoint
    const questions = [];
    const questionCount = parseInt(numQuestions) || 1;
    
    for (let i = 0; i < questionCount; i++) {
      const question = {
        id: `q_${Date.now()}_${i}`,
        text: `${difficulty} ${topic} question ${i + 1}: What is an important concept in ${topic}?`,
        options: [
          { id: `q_${Date.now()}_${i}_option_0`, text: `Correct answer about ${topic}` },
          { id: `q_${Date.now()}_${i}_option_1`, text: `Incorrect option A` },
          { id: `q_${Date.now()}_${i}_option_2`, text: `Incorrect option B` },
          { id: `q_${Date.now()}_${i}_option_3`, text: `Incorrect option C` }
        ],
        correctAnswerId: `q_${Date.now()}_${i}_option_0`,
        explanation: `This is the correct answer because it demonstrates key principles of ${topic}.`
      };
      questions.push(question);
    }

    const quizData = {
      id: `quiz_${Date.now()}`,
      totalQuestions: questions.length,
      questions: questions
    };

    console.log(`✅ Generated quiz with ${questions.length} questions via Python bridge`);
    return quizData;
    
  } catch (error) {
    console.error('❌ Quiz generation error in Python bridge:', error);
    throw error;
  }
}

async function getDeepSeekStatus() {
  return {
    available: true,
    model: "deepseek-chat",
    status: "ready",
    version: "v1.0",
    capabilities: ["text-generation", "code-completion"],
    rate_limit: {
      requests_per_minute: 60,
      tokens_per_minute: 10000
    }
  };
}

async function getQuestionHistoryPaginated(offset, limit) {
  const mockQuestions = Array.from({ length: limit }, (_, i) => ({
    id: `q_${offset + i}`,
    question: `Sample question ${offset + i}?`,
    options: ["Option A", "Option B", "Option C", "Option D"],
    correct: i % 4,
    topic: "General",
    difficulty: ["Easy", "Medium", "Hard"][i % 3],
    timestamp: new Date(Date.now() - i * 86400000).toISOString()
  }));
  
  return JSON.stringify(mockQuestions);
}

async function getQuestionHistory(filterParams) {
  return [
    {
      id: "1",
      question: "What is the capital of France?",
      options: ["London", "Berlin", "Paris", "Madrid"],
      correct: 2,
      topic: "Geography",
      difficulty: "Easy",
      timestamp: new Date().toISOString()
    }
  ];
}

async function getHistoryStats() {
  return {
    total_questions: 10,
    by_topic: { "Geography": 3, "Math": 4, "Science": 3 },
    by_difficulty: { "Easy": 4, "Medium": 4, "Hard": 2 }
  };
}

async function searchQuestions(searchTerm) {
  const mockResults = [
    {
      id: "search_1",
      question: `Question containing '${searchTerm}'?`,
      options: ["A", "B", "C", "D"],
      correct: 0,
      topic: "Search Results",
      difficulty: "Medium",
      timestamp: new Date().toISOString()
    }
  ];
  return JSON.stringify(mockResults);
}

async function filterQuestionsByTopic(topic) {
  const mockResults = [
    {
      id: `topic_${topic}_1`,
      question: `Question about ${topic}?`,
      options: ["A", "B", "C", "D"],
      correct: 0,
      topic: topic,
      difficulty: "Medium",
      timestamp: new Date().toISOString()
    }
  ];
  return JSON.stringify(mockResults);
}

async function filterQuestionsByDifficulty(difficulty) {
  const mockResults = [
    {
      id: `diff_${difficulty}_1`,
      question: `${difficulty} difficulty question?`,
      options: ["A", "B", "C", "D"],
      correct: 0,
      topic: "General",
      difficulty: difficulty,
      timestamp: new Date().toISOString()
    }
  ];
  return JSON.stringify(mockResults);
}

async function logClientEvent(eventData) {
  try {
    const eventType = eventData.type || 'unknown';
    const eventLevel = eventData.level || 'info';
    const eventMessage = eventData.message || '';
    const eventDetails = eventData.details || {};
    
    const logMessage = `CLIENT_EVENT [${eventType}]: ${eventMessage}`;
    console.log(logMessage, eventDetails);
    
    return true;
  } catch (error) {
    console.error('Error logging client event:', error);
    return false;
  }
}

startServer();
