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

// Quiz generation endpoint
app.post('/api/generate-quiz', async (req, res) => {
  try {
    const {
      topic,
      mode,
      gameMode, // maps to submode in Python
      questionType,
      difficulty,
      numQuestions, // maps to num_questions in Python
      enableTokenStreaming,
      deepSeekModel,
      customPrompt,
    } = req.body;

    // Prepare arguments for Python script
    const pythonArgs = [
      topic,
      difficulty,
      numQuestions,
      mode,
      gameMode, // submode
      questionType,
      enableTokenStreaming,
      deepSeekModel || '', // Pass empty string if undefined
      customPrompt || '', // Pass empty string if undefined
    ];

    const options = {
      mode: 'text',
      pythonPath: 'python',
      scriptPath: path.join(__dirname, '../../src/knowledge_app/core/'),
      args: pythonArgs,
    };

    PythonShell.run('quiz_api.py', options, (err, results) => {
      if (err) {
        console.error('PythonShell error:', err);
        res.status(500).json({ error: err.message });
      } else {
        try {
          const parsedResult = JSON.parse(results[0]);
          if (parsedResult.error) {
            res.status(500).json({ error: parsedResult.error });
          } else {
            res.json({ questions: parsedResult });
          }
        } catch (parseError) {
          console.error('JSON parse error:', parseError);
          console.error('Python output:', results[0]);
          res.status(500).json({ error: 'Failed to parse Python response.' });
        }
      }
    });
  } catch (error) {
    console.error('API error:', error);
    res.status(500).json({ error: error.message });
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
    });
    
    // Setup graceful shutdown
    portManager.setupGracefulShutdown(server, 8000);
    
  } catch (error) {
    console.error('❌ Failed to start API server:', error);
    process.exit(1);
  }
}

startServer();