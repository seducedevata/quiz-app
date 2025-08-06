# Qt WebEngine UI Documentation

This document outlines the UI structure, styling, and functionality of the Qt WebEngine application, based on the analysis of `src/knowledge_app/web/app.js`.

## 1. Screen System

The application uses a single-page interface with multiple "screens" that are shown or hidden as needed.

- **Screen Switching:** The `showScreen(screenName, navElement)` function manages screen visibility.
- **Mechanism:** It hides all elements with the `.screen` class and then displays the one with the ID `${screenName}-screen` by setting its `display` to `block` and adding an `active` class.
- **Screen IDs:**
    - `home-screen`
    - `quiz-screen`
    - `review-screen`
    - `train-screen`
    - `settings-screen`

## 2. Navigation System

- **Function:** `setupNavigationButtons()` attaches click handlers to navigation elements.
- **Structure:** Navigation is built around elements with the `.nav-item` class.
- **Styling:**
    - **Default State:** `.nav-item`
    - **Active State:** The `.nav-item.active` class highlights the currently selected screen.
    - **Hover Effects:** CSS transitions are likely used for hover effects (details to be confirmed from CSS).

## 3. Screen Layouts

### 3.1. Home Screen (`home-screen`)

- **Welcome Card:** A central welcome message or card is the main feature.
- **Statistics Grid:** A 3-column grid displays user statistics. The `updateStats()` function populates elements with `data-stat` attributes (`quizzes`, `score`, `questions`).

### 3.2. Quiz Setup Screen (`quiz-screen` -> `quiz-setup`)

This screen contains a form for configuring a new quiz.

- **Form Structure:** A `div` with `id="quiz-setup"` contains the form elements.
- **Form Fields:**
    - `quiz-topic` (text input)
    - `quiz-mode` (select dropdown)
    - `quiz-game-mode` (select dropdown)
    - `quiz-submode` (select dropdown)
    - `quiz-difficulty` (select dropdown)
    - `quiz-questions` (number input)
- **Expert Mode:** This is an option within the `quiz-difficulty` dropdown. Selecting "expert" triggers a special quiz generation path.
- **Buttons:**
    - `start-quiz-button`: Starts a custom quiz.
    - `quick-quiz-button`: Initiates a quick quiz.

## 4. Styling System

### 4.1. CSS Variables (from `app.js` inline styles and inferred usage)

While `app.js` primarily uses direct style manipulation and class assignments, the presence of a theme toggle and `data-theme` attribute implies the use of CSS variables for theming. Based on common patterns and the `globals.css` file, the following variables are expected or inferred:

- `--bg-primary`: Primary background color.
- `--bg-secondary`: Secondary background color (e.g., for cards, sidebars).
- `--text-primary`: Primary text color.
- `--text-secondary`: Secondary/lighter text color.
- `--border-color`: Color for borders.
- `--primary-color`: Main accent color (e.g., for buttons, highlights).
- `--primary-hover`: Hover state color for primary elements.
- `--success-color`: Color for success indicators.
- `--error-color`: Color for error indicators.
- `--warning-color`: Color for warning indicators.
- `--info-color`: Color for informational messages.

### 4.2. CSS Classes Inventory (from `app.js`)

This is a comprehensive list of CSS classes used or referenced in `app.js`, along with their inferred purposes:

- `.screen`: Base class for all main content screens. Used to show/hide screens.
- `.screen.active`: Applied to the currently visible screen.
- `.nav-item`: Applied to navigation buttons/links in the sidebar.
- `.nav-item.active`: Applied to the active navigation item.
- `.status-message`: Container for status messages (info, success, error, etc.).
- `.status-message.info`, `.status-message.success`, `.status-message.error`, `.status-message.warning`, `.status-message.loading`, `.status-message.gpu`, `.status-message.api`, `.status-message.deepseek`: Type-specific styling for status messages.
- `.status-icon`: Icon within a status message.
- `.status-text`: Text content within a status message.
- `.status-spinner`: Spinner animation within a status message (for loading states).
- `.error-container`: Container for error messages.
- `.error-message`: Individual error message display.
- `.error-icon`: Icon within an error message.
- `.error-text`: Text content within an error message.
- `.error-close`: Close button for error messages.
- `.loading-container`: Container for loading indicators.
- `.loading-message`: Message within a loading indicator.
- `.loading-spinner`: Spinner animation within a loading indicator.
- `.loading-text`: Text content within a loading indicator.
- `.option-button`: Buttons representing answer choices in the quiz.
- `.option-button.selected`: Applied to the currently selected answer option.
- `.option-button.correct-answer`: Applied to the correct answer option (e.g., in review mode).
- `.option-button.incorrect-answer`: Applied to the user's incorrect answer option (e.g., in review mode).
- `.option-letter`: Span for the letter (A, B, C, D) of an option.
- `.option-content`: Span for the actual text content of an option.
- `.timer-warning`: Applied to the timer display when time is low.
- `.timer-normal`: Applied to the timer display when time is normal.
- `.feedback-message`: Container for answer feedback (correct/incorrect).
- `.feedback-message.correct`, `.feedback-message.incorrect`: Type-specific styling for feedback messages.
- `.feedback-icon`: Icon within a feedback message.
- `.feedback-text`: Text content within a feedback message.
- `.explanation`: Container for the explanation text within feedback.
- `.quiz-results`: Container for quiz completion results.
- `.question-card`: Individual card displaying a question in the review history.
- `.question-topic`: Badge/label for the topic of a question in review history.
- `.question-difficulty`: Badge/label for the difficulty of a question in review history.
- `.question-difficulty.easy`, `.question-difficulty.medium`, `.question-difficulty.hard`, `.question-difficulty.expert`: Type-specific styling for difficulty badges.
- `.question-text-review`: Text of the question in review history.
- `.question-options-review`: Container for options in review history.
- `.option-item`: Individual option item in review history.
- `.option-item.correct`: Applied to the correct option in review history.
- `.question-meta`: Metadata (ID, timestamp) for a question in review history.
- `.modal`: Overlay for modal dialogs.
- `.modal-content`: Content area of a modal dialog.
- `.modal-close`: Close button for modal dialogs.
- `.latex-display`: Class for block-level LaTeX expressions.
- `.latex-inline`: Class for inline LaTeX expressions.
- `.fraction`: Container for fraction rendering.
- `.numerator`: Numerator part of a fraction.
- `.denominator`: Denominator part of a fraction.
- `.sqrt`: Container for square root rendering.
- `.radicand`: Radicand part of a square root.
- `.status-indicator`: General status indicator (e.g., for API keys).
- `.status-indicator.ready`: Green status indicator.
- `.status-indicator.error`: Red status indicator.
- `.training-status`: Displays the current training status.
- `.training-status.running`, `.training-status.completed`, `.training-status.error`: Type-specific styling for training status.
- `.progress-bar-container`: Container for a progress bar.
- `.progress-bar`: The actual progress bar element.
- `.metric`: Individual metric display in training section.
- `.file-item`: Individual item in the uploaded files list.
- `.file-icon`: Icon for a file item.
- `.file-name`: Name of a file item.
- `.file-size`: Size of a file item.
- `.delete-file-btn`: Delete button for a file item.
- `.no-files`: Message displayed when no files are uploaded.
- `.topic-disabled`: Applied to question type options that are not suitable for the current topic.
- `.topic-recommended`: Applied to recommended question type options.
- `.topic-optimal`: Applied to the best-matching question type option.
- `.stream-token`: Individual token in the token streaming display.
- `.streaming-complete`: Applied to the token stream container when streaming is complete.
- `.provider-status`: General status for API providers.
- `.provider-status.available`, `.provider-status.unavailable`: Type-specific styling for provider status.
- `.mode-info`, `.game-mode-info`, `.submode-info`, `.difficulty-info`: Information displays for quiz settings.

### 4.3. JavaScript Functions Inventory (UI-Affecting)

This section lists JavaScript functions in `app.js` that directly interact with or affect the UI, along with a brief description of their purpose.

- `AppLogger` (object): Comprehensive logging system that logs various application events, including UI interactions, errors, and performance metrics. It also handles storing critical logs and sending them to the Python backend.
    - `log(level, category, message, data)`: Core logging function.
    - `debug`, `info`, `warn`, `error`, `success`, `action`, `performance`, `user`, `system`: Convenience methods for different log levels.
    - `userAction(elementType, action, details)`: Logs user interactions with UI elements.
    - `navigation(from, to, method)`: Logs screen navigation events.
    - `trackNavigation(from, to, method)`: Tracks navigation with comprehensive details.
    - `trackUserAction(action, context)`: Tracks user actions with context.
    - `performanceMetric(operation, duration, success, metadata)`: Logs performance metrics.
    - `getSessionSummary()`: Returns a summary of the current session.
    - `displayLogSummary()`: Displays a comprehensive logging summary in the console.
    - `clearCriticalLogs()`: Clears stored critical logs.
    - `trackQuizAction(action, details)`: Tracks quiz-related actions.
    - `trackError(category, error, context)`: Tracks errors with context.

- `window.debugApp` (object): Global object providing convenience functions for debugging the application from the browser console.
    - `logs()`: Calls `AppLogger.displayLogSummary()`.
    - `session()`: Calls `AppLogger.getSessionSummary()`.
    - `critical()`: Returns critical logs from `localStorage`.
    - `clearLogs()`: Calls `AppLogger.clearCriticalLogs()`.
    - `testLog()`: Sends a test log message.
    - `bridge()`: Returns Python bridge status and available methods.
    - `testButtons()`: Checks status and clickability of navigation buttons and screens.
    - `fixButtons()`: Force re-attaches button handlers.
    - `testNavigation(screenName)`: Tests navigation to a specified screen.
    - `clickNav(screenName)`: Force clicks a navigation button.
    - `help()`: Displays debug commands help.

- `handleStartQuizClick(event)`: Global click handler for the "Start Quiz" button, includes bridge testing and quiz initiation logic.
- `showStatusDisplay(message, type)`: Displays a temporary status message at the top right of the screen.
- `hideStatusDisplay()`: Hides the status message.
- `getStatusIcon(type)`: Returns an emoji icon based on status type.
- `updateNavigationButtons(activeButton)`: Updates the `active` class on navigation buttons to highlight the current screen.
- `initializeScreen(screenName)`: Initializes screen-specific functionality when a screen is shown (e.g., `updateStats()` for home, `resetQuizState()` for quiz).
- `resetQuizState()`: Resets quiz-related UI elements and state variables.
- `loadSettings()`: Loads user settings from Python bridge or `localStorage` and applies them to the UI.
- `createLoadingContainer()`: Creates and appends a loading container element to the DOM.
- `showQuizLoading(message)`: Displays a quiz loading indicator.
- `hideQuizLoading()`: Hides the quiz loading indicator.
- `showQuizError(message)`: Displays an error message related to the quiz.
- `initializeApp()`: Main application initialization function, sets up event listeners, loads settings, and initializes DeepSeek integration.
- `initializeDeepSeek()`: Initializes DeepSeek integration by checking its status via the Python bridge and updating the UI.
- `updateDeepSeekUI(status)`: Updates the DeepSeek status indicator and text in the UI.
- `generateDeepSeekQuestion()`: Initiates DeepSeek question generation based on user inputs.
- `displayDeepSeekQuestion(question)`: Displays a DeepSeek generated question on the quiz screen.
- `setupTopicAnalysis()`: Sets up real-time topic analysis for the quiz topic input, adapting question type options.
- `handleTopicProfileUpdated(profile)`: Handles topic analysis results, enabling/disabling question type options and auto-selecting optimal types.
- `updateOptionState(option, enabled)`: Updates the visual state and `disabled` property of a select option.
- `updateTopicRecommendationIndicators(profile)`: Adds visual indicators (e.g., "⭐ BEST MATCH") to recommended question types.
- `showScreen(screenName, navElement)`: Core function to switch between different application screens.
- `toggleTheme()`: Toggles between 'dark' and 'light' themes, updates `data-theme` attribute on `<body>`, and saves preference.
- `hideQuizElements()`: Hides various quiz feedback and navigation elements.
- `startQuickQuiz()`: Initiates a quick quiz (navigates to quiz setup).
- `startCustomQuiz()`: Starts a custom quiz based on user-selected parameters, handles streaming, and calls Python bridge.
- `startStatusMonitoring(mode)`: Displays animated status messages during quiz generation (GPU or API related).
- `showQuizGame()`: Displays the main quiz game interface and hides setup/results.
- `handleQuestionReceived(data)`: Processes and displays a new quiz question, including LaTeX rendering and option button creation.
- `selectAnswer(index, event)`: Handles user selecting an answer option, updates UI, and prevents propagation.
- `handleAnswerFeedback(feedbackData)`: Receives and displays feedback on the user's answer.
- `showAnswerFeedback(feedback)`: Updates UI to show correct/incorrect answers and explanations.
- `handleQuizCompleted(completionData)`: Handles quiz completion, displays results.
- `handleError(errorData)`: Displays errors received from the Python bridge.
- `updateStatus(statusData)`: Updates the general status display with messages from the Python bridge.
- `handleStreamingStarted(streamData)`: Initializes UI for token streaming.
- `handleTokenReceived(tokenData)`: Appends received tokens to the streaming display.
- `handleStreamingCompleted(completionData)`: Finalizes the token streaming UI.
- `initializeTokenStreamUI()`: Creates or clears the token streaming display container.
- `handleTrainingProgressStructured(progressData)`: Updates training progress UI.
- `handleTrainingStatusChanged(statusData)`: Updates training status text and styling.
- `handleTrainingMetricsUpdate(metricsData)`: Updates training metrics display.
- `handleTrainingConfigSaved(configData)`: Shows status message when training config is saved.
- `updateTrainingProgressUI(progressData)`: Helper to update the progress bar and text.
- `updateTrainingMetricsUI(metricsData)`: Helper to update the display of training metrics.
- `saveSettings()`: Collects settings from UI and saves them to `localStorage` and Python bridge.
- `applySettingsToUI(settings)`: Applies loaded settings to various UI input elements.
- `collectSettingsFromUI()`: Gathers current settings from UI input elements.
- `loadExistingFiles()`: Loads previously uploaded files via Python bridge and updates the file list UI.
- `setupAutoSave()`: Configures auto-saving of settings on UI changes and before page unload.
- `ensureApiKeyPersistence()`: Ensures API keys are persisted, including periodic session storage backup.
- `saveApiKeysToSessionStorage()`: Saves API keys to session storage.
- `loadApiKeysFromSessionStorage()`: Loads API keys from session storage.
- `updateApiKeyStatusIndicators()`: Updates visual indicators for API key presence.
- `updateProviderStatuses()`: Updates status indicators for all API providers.
- `updateProviderStatusUI(provider, status)`: Helper to update individual provider status in UI.
- `updateFilesList()`: Renders the list of uploaded files in the UI.
- `formatFileSize(bytes)`: Formats file size for display.
- `deleteFile(filename)`: Deletes an uploaded file via Python bridge and updates UI.
- `updateModeInfo()`, `updateGameModeInfo()`, `updateSubmodeInfo()`, `updateDifficultyInfo()`: Update descriptive text for quiz mode, game mode, submode, and difficulty settings.
- `loadTrainingConfiguration()`: Loads training configuration from Python bridge and applies to UI.
- `applyTrainingConfigToUI(config)`: Applies loaded training configuration to UI elements.
- `processLatexText(text)`: Processes text to convert LaTeX syntax into HTML spans for rendering.
- `renderLatex(container)`: Triggers MathJax rendering for LaTeX elements within a container.
- `startTimer()`: Starts a countdown timer for quiz questions.
- `submitAnswer()`: Submits the selected answer to the Python bridge.
- `showNavigationButtons()`: Displays navigation buttons (Previous, Next, Finish Quiz) in the quiz interface.
- `navigateQuestion(direction)`: Navigates to the previous or next question via Python bridge.
- `finishQuiz()`: Requests to finish the current quiz via Python bridge.
- `loadQuestionHistory(offset, limit)`: Loads question history from the backend and displays it.
- `displayQuestionHistory(questions)`: Renders the list of question cards in the review history UI.
- `createQuestionCard(question, index)`: Creates an HTML element for a single question card in the review history.
- `updateTopicFilter(questions)`: Populates the topic filter dropdown with available topics from question history.
- `searchQuestions()`: Searches question history based on a search term.
- `filterQuestionsByTopic()`: Filters question history by selected topic.
- `filterQuestionsByDifficulty()`: Filters question history by selected difficulty.
- `showQuestionStats()`: Loads and displays question statistics.
- `displayQuestionStats(data)`: Renders question statistics in the UI.
- `showQuestionModal(question)`: Displays a modal with detailed information about a selected question.
- `closeQuestionModal()`: Closes the question detail modal.
- `nuclearLoadQuestions()`: Debug function to force load questions from the Python bridge.
- `nuclearTestDisplay()`: Debug function to display mock question data.