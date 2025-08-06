const fs = require('fs');
const path = require('path');

const HISTORY_FILE = path.join(__dirname, 'question-history.json');

const readHistory = () => {
  if (!fs.existsSync(HISTORY_FILE)) {
    fs.writeFileSync(HISTORY_FILE, '[]', 'utf8');
  }
  const data = fs.readFileSync(HISTORY_FILE, 'utf8');
  return JSON.parse(data);
};

const writeHistory = (history) => {
  fs.writeFileSync(HISTORY_FILE, JSON.stringify(history, null, 2), 'utf8');
};

const addQuestionToHistory = (question) => {
  const history = readHistory();
  history.unshift(question); // Add to the beginning
  writeHistory(history);
};

module.exports = {
  readHistory,
  addQuestionToHistory,
};