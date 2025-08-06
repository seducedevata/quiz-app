# 🎯 **COMPREHENSIVE UI RECREATION PROMPT FOR CURSOR**

Copy and paste this entire prompt into Cursor to recreate your Qt WebEngine Knowledge App UI exactly:

---

## **TASK: Recreate Qt WebEngine Knowledge App UI in React Native Windows**

I need you to recreate the exact UI design and functionality from my original Qt WebEngine Knowledge App. The React Native Windows app is already working with basic navigation, but the UI needs to match the original design precisely.

### **ORIGINAL APP ANALYSIS:**

Based on the original Qt WebEngine app, here are the key UI elements to recreate:

- **Dark theme with modern design** using CSS custom properties
- **Sidebar navigation** with icons and active states
- **Main content area** with cards and panels
- **Professional quiz interface** with progress indicators
- **Real-time token streaming display**
- **Expert mode** with advanced controls
- **Statistics dashboard** with charts
- **Settings panel** with organized sections

### **REQUIRED UI COMPONENTS TO CREATE:**

#### **1. Main Layout Structure**
```
KnowledgeAppRNW/src/components/layout/
├── MainLayout.tsx          // Main app layout with sidebar + content
├── Sidebar.tsx             // Left navigation sidebar
├── TopBar.tsx              // Top header bar
├── ContentArea.tsx         // Main content wrapper
└── StatusBar.tsx           // Bottom status bar
```

#### **2. Navigation Components**
```
KnowledgeAppRNW/src/components/navigation/
├── NavigationItem.tsx      // Individual nav item with icon
├── NavigationGroup.tsx     // Grouped nav items
└── BreadcrumbNav.tsx       // Breadcrumb navigation
```

#### **3. Quiz Interface Components**
```
KnowledgeAppRNW/src/components/quiz/
├── QuizCard.tsx            // Main quiz question card
├── QuestionDisplay.tsx     // Question text with LaTeX support
├── AnswerOptions.tsx       // Multiple choice options
├── QuizProgress.tsx        // Progress bar and indicators
├── TokenStream.tsx         // Real-time token streaming
├── QuizTimer.tsx           // Timer component
├── QuizResults.tsx         // Results display
├── QuizSetup.tsx           // Quiz setup form
└── StatusDisplay.tsx       // Status messages
```

#### **4. Expert Mode Components**
```
KnowledgeAppRNW/src/components/expert/
├── ExpertPanel.tsx         // Main expert mode panel
├── ModelSelector.tsx       // AI model selection
├── AdvancedSettings.tsx    // Expert configuration
├── PipelineVisualization.tsx // Two-model pipeline display
├── ExpertMetrics.tsx       // Advanced metrics
└── DeepSeekSection.tsx     // DeepSeek AI integration
```

#### **5. Dashboard Components**
```
KnowledgeAppRNW/src/components/dashboard/
├── StatsCard.tsx           // Statistics cards
├── PerformanceChart.tsx    // Performance charts
├── RecentActivity.tsx      // Recent quiz activity
├── ProgressOverview.tsx    // Overall progress
└── QuickActions.tsx        // Quick action buttons
```

#### **6. Settings Components**
```
KnowledgeAppRNW/src/components/settings/
├── SettingsPanel.tsx       // Main settings container
├── SettingsGroup.tsx       // Grouped settings
├── SettingItem.tsx         // Individual setting item
├── ThemeSelector.tsx       // Theme selection
├── ApiKeyInput.tsx         // API key configuration
├── PreferenceToggle.tsx    // Toggle switches
└── ApiProviderCard.tsx     // API provider configuration
```

#### **7. Common UI Components**
```
KnowledgeAppRNW/src/components/common/
├── Card.tsx                // Modern card component
├── Button.tsx              // Styled button variants
├── Input.tsx               // Form input components
├── Modal.tsx               // Modal dialogs
├── Dropdown.tsx            // Dropdown selectors
├── ProgressBar.tsx         // Progress indicators
├── LoadingSpinner.tsx      // Loading animations
├── Toast.tsx               // Notification toasts
├── Tooltip.tsx             // Hover tooltips
├── Icon.tsx                // Icon component system
├── FormGroup.tsx           // Form group wrapper
└── StatusMessage.tsx       // Status message component
```

### **DESIGN SPECIFICATIONS:**

#### **Color Scheme (Dark Theme)**
```typescript
const colors = {
  // Light theme colors
  light: {
    bgPrimary: '#ffffff',
    bgSecondary: '#f8f9fa',
    textPrimary: '#212529',
    textSecondary: '#6c757d',
    borderColor: '#dee2e6',
    primaryColor: '#6366f1',
    primaryHover: '#4f46e5',
    successColor: '#10b981',
    dangerColor: '#ef4444',
    warningColor: '#f59e0b',
    shadow: 'rgba(0, 0, 0, 0.1)',
  },
  // Dark theme colors (default)
  dark: {
    bgPrimary: '#0f172a',
    bgSecondary: '#1e293b',
    textPrimary: '#f8fafc',
    textSecondary: '#94a3b8',
    borderColor: '#334155',
    primaryColor: '#6366f1',
    primaryHover: '#818cf8',
    successColor: '#10b981',
    dangerColor: '#ef4444',
    warningColor: '#f59e0b',
    shadow: 'rgba(0, 0, 0, 0.3)',
  }
};
```

#### **Typography System**
```typescript
const typography = {
  h1: { fontSize: 28, fontWeight: '700', lineHeight: 36 },
  h2: { fontSize: 24, fontWeight: '600', lineHeight: 32 },
  h3: { fontSize: 20, fontWeight: '600', lineHeight: 28 },
  h4: { fontSize: 18, fontWeight: '600', lineHeight: 24 },
  body: { fontSize: 16, fontWeight: '400', lineHeight: 24 },
  bodySmall: { fontSize: 14, fontWeight: '400', lineHeight: 20 },
  caption: { fontSize: 12, fontWeight: '400', lineHeight: 16 },
  button: { fontSize: 16, fontWeight: '500', lineHeight: 24 },
};
```

#### **Spacing System**
```typescript
const spacing = {
  xs: 4, sm: 8, md: 16, lg: 24, xl: 32, xxl: 48
};
```

### **SPECIFIC UI REQUIREMENTS:**

#### **1. Header Section**
- **App title**: "Knowledge App" with gradient text effect
- **Subtitle**: "Modern Learning Platform"
- **Theme toggle**: Moon/sun icon button
- **Background**: Secondary background with border bottom

#### **2. Sidebar Navigation**
- **Width**: 250px fixed
- **Dark background** with subtle borders
- **Navigation items**: 
  - 🏠 Home (active by default)
  - 📝 Quiz
  - 📚 Review
  - 🧠 Train Model
  - ⚙️ Settings
- **Active state**: Primary color background with white text
- **Hover effects**: Subtle background change with translateX(4px)
- **Icons**: Use emoji or React Native Vector Icons

#### **3. Main Content Area**
- **Responsive layout** with proper padding (2rem)
- **Card-based design** with shadows and rounded corners
- **Smooth transitions** between screens with fadeIn animation
- **Proper spacing** between elements

#### **4. Home Screen**
- **Welcome card** with title and description
- **Stats grid** with 3 cards:
  - Quizzes Taken: 0
  - Average Score: 0%
  - Questions Answered: 0
- **Dashboard info** section with navigation hints

#### **5. Quiz Setup Screen**
- **Form groups** with labels and inputs:
  - Topic input field
  - Mode selector (Offline/Auto/Online)
  - Game Mode (Casual/Serious)
  - Question Type (Mixed/Numerical/Conceptual)
  - Difficulty (Easy/Medium/Hard/Expert)
  - Number of Questions (1-10)
  - Token Streaming checkbox
- **DeepSeek section** (hidden by default)
- **Start Quiz button** with primary styling

#### **6. Quiz Interface**
- **Quiz header** with question number and timer
- **Question container** with large text
- **Options container** with styled option buttons:
  - Letter circles (A, B, C, D) with primary color
  - Option content with proper spacing
  - Hover effects with translateX(4px)
  - Selected state with primary background
  - Correct/incorrect states with green/red colors
- **Feedback container** for answer feedback
- **Explanation container** with left border accent
- **Navigation buttons** for quiz flow

#### **7. Status Display System**
- **Status messages** with different types:
  - Turbo: Blue gradient background
  - GPU: Green gradient background
  - Success: Green gradient
  - Warning: Yellow gradient
  - Error: Red gradient
  - Info: Blue gradient
  - API: Blue with network pulse animation
- **Status icons** with spin animation
- **Status text** with proper typography

#### **8. Settings Screen**
- **API Provider Cards** in grid layout:
  - Provider header with icon, name, and status
  - API key input field
  - Provider toggle switch
  - Connection status indicators
- **Card states**: Connected (green border), Error (red border), Testing (yellow border)
- **API help section** with collapsible details

### **IMPLEMENTATION REQUIREMENTS:**

#### **1. Theme System**
```typescript
// Create comprehensive theme system
export interface Theme {
  colors: typeof colors.dark;
  typography: typeof typography;
  spacing: typeof spacing;
  borderRadius: { sm: 4, md: 8, lg: 12, xl: 16 };
  shadows: { 
    sm: '0 2px 4px rgba(0, 0, 0, 0.1)', 
    md: '0 4px 6px rgba(0, 0, 0, 0.1)', 
    lg: '0 8px 16px rgba(0, 0, 0, 0.15)' 
  };
}

// Theme context
export const ThemeContext = React.createContext<{
  theme: Theme;
  isDark: boolean;
  toggleTheme: () => void;
}>({} as any);
```

#### **2. Component Architecture**
- **Consistent prop interfaces** for all components
- **Theme integration** in every component using useTheme hook
- **Accessibility support** with proper labels and roles
- **TypeScript strict mode** with proper typing
- **Reusable component patterns**

#### **3. Animation System**
```typescript
// Animation utilities
export const animations = {
  fadeIn: {
    from: { opacity: 0, transform: [{ translateY: 10 }] },
    to: { opacity: 1, transform: [{ translateY: 0 }] },
  },
  slideIn: {
    from: { transform: [{ translateX: -20 }] },
    to: { transform: [{ translateX: 0 }] },
  },
  pulse: {
    from: { opacity: 1, transform: [{ scale: 1 }] },
    to: { opacity: 0.8, transform: [{ scale: 1.02 }] },
  },
};
```

#### **4. Responsive Design**
- **Flexible layouts** that work on different screen sizes
- **Proper text scaling** for accessibility
- **Touch-friendly** button sizes (minimum 44px)
- **Keyboard navigation** support

### **SCREEN-SPECIFIC REQUIREMENTS:**

#### **HomeScreen**
```typescript
// Update existing HomeScreen.tsx
const HomeScreen = () => {
  const { theme } = useTheme();
  
  return (
    <ScrollView style={[styles.container, { backgroundColor: theme.colors.bgPrimary }]}>
      <Card style={styles.welcomeCard}>
        <Text style={[styles.title, { color: theme.colors.textPrimary }]}>
          Welcome to Knowledge App
        </Text>
        <Text style={[styles.subtitle, { color: theme.colors.textSecondary }]}>
          Test your knowledge with AI-powered quizzes
        </Text>
        
        <View style={styles.statsGrid}>
          <StatsCard title="0" subtitle="Quizzes Taken" />
          <StatsCard title="0%" subtitle="Average Score" />
          <StatsCard title="0" subtitle="Questions Answered" />
        </View>
        
        <View style={styles.dashboardInfo}>
          <Text style={styles.infoText}>📊 View your learning progress and statistics above</Text>
          <Text style={styles.infoText}>🎯 Use the sidebar to start a new quiz or review questions</Text>
        </View>
      </Card>
    </ScrollView>
  );
};
```

#### **QuizSetupScreen**
```typescript
// Update existing QuizSetupScreen.tsx with proper form styling
const QuizSetupScreen = () => {
  const { theme } = useTheme();
  
  return (
    <ScrollView style={styles.container}>
      <Card style={styles.quizSetup}>
        <Text style={styles.title}>Quiz Setup</Text>
        
        <FormGroup label="Topic">
          <Input 
            placeholder="Enter topic (e.g., Science, History)"
            value={topic}
            onChangeText={setTopic}
          />
        </FormGroup>
        
        <FormGroup label="Mode">
          <Dropdown
            options={modeOptions}
            value={mode}
            onChange={setMode}
          />
          <Text style={styles.modeInfo}>🤖 Auto-selecting best available method</Text>
        </FormGroup>
        
        {/* Add all other form groups */}
        
        <Button
          title="⭐ START QUIZ"
          onPress={startQuiz}
          style={styles.startButton}
        />
      </Card>
    </ScrollView>
  );
};
```

#### **QuizScreen**
```typescript
// Update existing QuizScreen.tsx with proper quiz interface
const QuizScreen = () => {
  const { theme } = useTheme();
  
  return (
    <View style={styles.container}>
      <StatusDisplay />
      
      <View style={styles.quizHeader}>
        <Text style={styles.questionNumber}>Question 1 of 5</Text>
        <Text style={styles.timer}>Time: 30s</Text>
      </View>
      
      <Card style={styles.questionContainer}>
        <Text style={styles.questionText}>{question}</Text>
        <AnswerOptions 
          options={options}
          selectedAnswer={selectedAnswer}
          onSelect={setSelectedAnswer}
        />
      </Card>
      
      <View style={styles.quizFooter}>
        <Button title="Submit" onPress={submitAnswer} />
        <Button title="Skip" variant="secondary" onPress={skipQuestion} />
      </View>
    </View>
  );
};
```

### **TECHNICAL REQUIREMENTS:**

#### **Dependencies to Install**
```bash
npm install react-native-vector-icons react-native-linear-gradient react-native-svg react-native-reanimated @react-native-async-storage/async-storage
```

#### **File Structure to Create**
```
KnowledgeAppRNW/src/
├── components/
│   ├── layout/
│   │   ├── MainLayout.tsx
│   │   ├── Sidebar.tsx
│   │   ├── TopBar.tsx
│   │   └── ContentArea.tsx
│   ├── navigation/
│   │   ├── NavigationItem.tsx
│   │   └── NavigationGroup.tsx
│   ├── quiz/
│   │   ├── QuizCard.tsx
│   │   ├── QuestionDisplay.tsx
│   │   ├── AnswerOptions.tsx
│   │   ├── QuizProgress.tsx
│   │   ├── QuizTimer.tsx
│   │   ├── QuizResults.tsx
│   │   ├── QuizSetup.tsx
│   │   └── StatusDisplay.tsx
│   ├── expert/
│   │   ├── ExpertPanel.tsx
│   │   ├── ModelSelector.tsx
│   │   └── DeepSeekSection.tsx
│   ├── dashboard/
│   │   ├── StatsCard.tsx
│   │   └── PerformanceChart.tsx
│   ├── settings/
│   │   ├── SettingsPanel.tsx
│   │   ├── ApiProviderCard.tsx
│   │   └── ApiKeyInput.tsx
│   └── common/
│       ├── Card.tsx
│       ├── Button.tsx
│       ├── Input.tsx
│       ├── Modal.tsx
│       ├── Dropdown.tsx
│       ├── ProgressBar.tsx
│       ├── LoadingSpinner.tsx
│       ├── FormGroup.tsx
│       └── StatusMessage.tsx
├── styles/
│   ├── theme.ts
│   ├── colors.ts
│   ├── typography.ts
│   └── spacing.ts
├── hooks/
│   ├── useTheme.ts
│   └── useAnimation.ts
└── utils/
    ├── animations.ts
    └── helpers.ts
```

### **STYLING APPROACH:**

Use StyleSheet.create for all components with theme integration:

```typescript
const createStyles = (theme: Theme) => StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.bgPrimary,
    padding: theme.spacing.lg,
  },
  card: {
    backgroundColor: theme.colors.bgSecondary,
    borderRadius: theme.borderRadius.lg,
    padding: theme.spacing.lg,
    shadowColor: theme.colors.shadow,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 4,
  },
  button: {
    backgroundColor: theme.colors.primaryColor,
    paddingVertical: theme.spacing.md,
    paddingHorizontal: theme.spacing.lg,
    borderRadius: theme.borderRadius.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonText: {
    color: '#ffffff',
    fontSize: theme.typography.button.fontSize,
    fontWeight: theme.typography.button.fontWeight,
  },
});
```

### **FINAL REQUIREMENTS:**

1. **Exact visual match** to the original Qt WebEngine app
2. **Smooth animations** and transitions using React Native Reanimated
3. **Professional dark theme** throughout with light theme support
4. **Consistent component patterns** and prop interfaces
5. **Proper TypeScript typing** for all components
6. **Accessibility compliance** with proper labels and roles
7. **Performance optimization** with proper memoization
8. **Responsive design** that works on different screen sizes

### **IMPLEMENTATION STEPS:**

1. **Create the theme system** first (theme.ts, colors.ts, typography.ts)
2. **Build common components** (Card, Button, Input, etc.)
3. **Create layout components** (MainLayout, Sidebar, TopBar)
4. **Update existing screens** to use new components and styling
5. **Add animations and transitions**
6. **Test on different screen sizes**
7. **Ensure accessibility compliance**

**Create all these components and update the existing screens to use them. The final result should look and feel exactly like the original Qt WebEngine Knowledge App but as a native Windows application.**

---

**Paste this entire prompt into Cursor and it will create all the UI components to match your original app exactly!**