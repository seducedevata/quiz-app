# 🚨 CRITICAL: Qt WebEngine to Next.js UI Migration Guide

## ⚠️ MANDATORY FIRST STEP: READ QT WEBENGINE FILES!

**STOP! Before making ANY changes, you MUST read and analyze these Qt WebEngine source files to understand the EXACT UI structure and styling. DO NOT ASSUME anything about the UI - READ THE ACTUAL CODE!**

---

## 📁 REQUIRED FILES TO EXAMINE FIRST (IN ORDER):

### 1. **PRIMARY Qt WebEngine UI Files (READ THESE COMPLETELY):**

#### A. `src/knowledge_app/web/app.js` (MOST IMPORTANT - READ ENTIRELY)
**This file contains the COMPLETE UI structure and styling. You MUST read it to understand:**
- Screen layouts and HTML structure
- CSS classes and styling definitions  
- JavaScript functions for UI interactions
- Navigation system implementation
- Quiz interface structure
- Status display system
- Logging system (AppLogger)
- Button styling and hover effects
- Theme system (dark/light mode)

#### B. `src/knowledge_app/webengine_app.py` (READ FOR CONTEXT)
**This file shows how the Qt WebEngine loads and manages the UI:**
- Window setup and configuration
- Bridge between Python and JavaScript
- Event handling system
- Screen management logic

#### C. `knowledge-app-next/src/app/globals.css` (CURRENT NEXT.JS STYLING)
**Compare this with the Qt styling to see what's missing**

### 2. **Qt HTML Debug Files (FOR REFERENCE ONLY):**
- `debug_navigation.html` - Navigation debug interface
- `test_js_syntax.html` - JavaScript testing interface  
- `BUTTON_CLICKABILITY_AND_LOGGING_FIXES.md` - Button fix documentation
- `fix_button_overlap.js` - Button overlap fixes

### 3. **Next.js Files to Update (AFTER examining Qt files):**
- `knowledge-app-next/src/app/page.tsx` - Home screen
- `knowledge-app-next/src/app/quiz/page.tsx` - Quiz setup screen  
- `knowledge-app-next/src/components/layout/MainLayout.tsx` - Main layout
- `knowledge-app-next/src/components/layout/Sidebar.tsx` - Navigation sidebar
- `knowledge-app-next/src/components/layout/TopBar.tsx` - Top navigation bar

---

## 🔍 DETAILED STEP-BY-STEP ANALYSIS PROCESS:

### STEP 1: Analyze Qt WebEngine Structure (MANDATORY)

**READ `src/knowledge_app/web/app.js` and document:**

#### A. Screen System:
- Find the `showScreen()` function - how does screen switching work?
- Find all screen IDs (home-screen, quiz-screen, review-screen, etc.)
- Document the HTML structure for each screen
- Note how screens are hidden/shown

#### B. Navigation System:
- Find `setupNavigationButtons()` function
- Document navigation button structure and styling
- Find CSS classes: `.nav-item`, `.nav-item.active`, etc.
- Note hover effects and transitions

#### C. Home Screen Layout:
- Find the home screen HTML structure
- Document the welcome card layout
- Find statistics grid structure (3 columns)
- Note all CSS classes used

#### D. Quiz Setup Screen:
- Find quiz setup form structure
- Document all form fields and their styling
- Find expert mode section structure
- Note button styling and interactions

#### E. Styling System:
- Find all CSS variable definitions (--bg-primary, --text-primary, etc.)
- Document all CSS classes (.btn-primary, .welcome-card, etc.)
- Note color schemes for light/dark themes
- Find animation and transition definitions

### STEP 2: Create Qt UI Documentation

**Create a detailed map of the Qt UI by documenting:**

#### A. HTML Structure Map:
```
Main Layout:
├── Header (.app-header)
│   ├── Title (.app-title)
│   └── Theme Toggle
├── Sidebar (.side-nav)
│   └── Navigation Items (.nav-item)
└── Content Area (.content-area)
    ├── Home Screen (.welcome-card)
    ├── Quiz Screen (.quiz-setup-card)
    └── Other Screens
```

#### B. CSS Classes Inventory:
- List EVERY CSS class found in app.js
- Document their properties and values
- Note which classes are used where

#### C. JavaScript Functions Inventory:
- List all functions that affect UI
- Document their parameters and behavior
- Note event handlers and interactions

### STEP 3: Compare with Next.js Implementation

**For each Next.js file, compare element by element:**

#### A. Layout Comparison:
**Qt Reference:** `src/knowledge_app/web/app.js` (main layout structure)
**Next.js Target:** `knowledge-app-next/src/components/layout/MainLayout.tsx`

**Check:**
- [ ] Does the Next.js layout match Qt's HTML structure exactly?
- [ ] Are all CSS classes present and identical?
- [ ] Is the header structure the same?
- [ ] Is the sidebar structure the same?
- [ ] Is the content area structure the same?

#### B. Navigation Comparison:
**Qt Reference:** `src/knowledge_app/web/app.js` (search for "nav-item", "setupNavigationButtons")
**Next.js Target:** `knowledge-app-next/src/components/layout/Sidebar.tsx`

**Check:**
- [ ] Navigation button HTML structure matches
- [ ] CSS classes match exactly (.nav-item, .nav-item.active)
- [ ] Hover effects match
- [ ] Active state styling matches
- [ ] Icon positioning matches
- [ ] Text styling matches

#### C. Home Screen Comparison:
**Qt Reference:** `src/knowledge_app/web/app.js` (search for "home-screen", "welcome-card")
**Next.js Target:** `knowledge-app-next/src/app/page.tsx`

**Check:**
- [ ] Welcome card structure matches
- [ ] Statistics grid is exactly 3 columns
- [ ] Card styling matches (padding, margins, shadows)
- [ ] Typography matches (font sizes, weights, colors)
- [ ] Spacing matches exactly

#### D. Quiz Setup Comparison:
**Qt Reference:** `src/knowledge_app/web/app.js` (search for "quiz-setup", "quiz-setup-card")
**Next.js Target:** `knowledge-app-next/src/app/quiz/page.tsx`

**Check:**
- [ ] Form layout structure matches
- [ ] Input field styling matches
- [ ] Dropdown styling matches  
- [ ] Button styling matches
- [ ] Expert mode section matches
- [ ] Form validation behavior matches

### STEP 4: Identify Missing Elements

**Create detailed lists of what's missing in Next.js:**

#### A. Missing CSS Classes:
- Compare globals.css with Qt CSS classes
- List every missing class
- Note property differences for existing classes

#### B. Missing HTML Structure:
- Compare component JSX with Qt HTML
- Note missing elements or incorrect nesting
- Document structural differences

#### C. Missing JavaScript Functionality:
- Compare Next.js functions with Qt functions
- Note missing event handlers
- Document behavioral differences

#### D. Missing Styling:
- Compare visual appearance
- Note color differences
- Document spacing/sizing differences

---

## 🎯 SPECIFIC IMPLEMENTATION TASKS:

### TASK 1: Fix Global Styling
**File:** `knowledge-app-next/src/app/globals.css`

**Actions:**
1. Read Qt CSS variables from app.js
2. Ensure ALL Qt CSS variables are present
3. Add any missing CSS classes
4. Match colors, fonts, and spacing exactly
5. Ensure dark/light theme support matches

### TASK 2: Fix Main Layout
**File:** `knowledge-app-next/src/components/layout/MainLayout.tsx`

**Actions:**
1. Compare with Qt main layout structure
2. Ensure HTML structure matches exactly
3. Add missing CSS classes
4. Fix any structural differences

### TASK 3: Fix Navigation Sidebar
**File:** `knowledge-app-next/src/components/layout/Sidebar.tsx`

**Actions:**
1. Compare with Qt navigation structure
2. Ensure navigation items match exactly
3. Fix active state styling
4. Fix hover effects
5. Ensure icon positioning matches

### TASK 4: Fix Home Screen
**File:** `knowledge-app-next/src/app/page.tsx`

**Actions:**
1. Compare with Qt home screen structure
2. Ensure welcome card matches exactly
3. Fix statistics grid (must be exactly 3 columns)
4. Match typography and spacing
5. Add any missing elements

### TASK 5: Fix Quiz Setup Screen
**File:** `knowledge-app-next/src/app/quiz/page.tsx`

**Actions:**
1. Compare with Qt quiz setup structure
2. Ensure form layout matches exactly
3. Fix input field styling
4. Fix button styling
5. Add expert mode section if missing

---

## 🎨 VISUAL PARITY REQUIREMENTS:

The Next.js app must be a **PIXEL-PERFECT** match to Qt WebEngine:

### Colors:
- Use EXACT same CSS variables as Qt
- Match light/dark theme colors exactly
- Ensure all text colors match

### Typography:
- Use same font families
- Match font sizes exactly
- Match font weights exactly
- Match line heights

### Spacing:
- Match padding values exactly
- Match margin values exactly
- Match gap values in grids/flexbox

### Styling:
- Match border-radius values
- Match box-shadow values
- Match transition/animation timing
- Match hover effects exactly

### Layout:
- Match grid structures exactly
- Match flexbox layouts exactly
- Match element positioning

---

## 🚀 TESTING CHECKLIST:

After implementing changes, verify:

### Visual Testing:
- [ ] Home screen looks identical to Qt
- [ ] Navigation sidebar looks identical to Qt
- [ ] Quiz setup screen looks identical to Qt
- [ ] All buttons look and behave identically
- [ ] All cards and layouts match exactly
- [ ] Dark/light theme switching works identically

### Functional Testing:
- [ ] Navigation works the same way
- [ ] Screen transitions work the same way
- [ ] Form interactions work the same way
- [ ] All hover effects work identically
- [ ] All click behaviors work identically

### Responsive Testing:
- [ ] Layout works on different screen sizes
- [ ] Mobile responsiveness matches Qt behavior
- [ ] All elements scale appropriately

---

## ⚠️ CRITICAL SUCCESS FACTORS:

1. **READ QT FILES FIRST** - Never assume, always verify against source
2. **EXACT MATCHING** - Pixel-perfect visual parity is required
3. **COMPLETE ANALYSIS** - Don't skip any sections of the Qt code
4. **SYSTEMATIC COMPARISON** - Compare every element methodically
5. **THOROUGH TESTING** - Test every screen and interaction

---

## 🔄 ITERATIVE PROCESS:

1. **Analyze** Qt WebEngine files thoroughly
2. **Document** findings in detail
3. **Compare** with Next.js implementation
4. **Identify** specific gaps and differences
5. **Implement** fixes systematically
6. **Test** for visual and functional parity
7. **Refine** until perfect match is achieved

---

## 📝 DOCUMENTATION REQUIREMENTS:

For each change made, document:
- What Qt element/styling was being matched
- What was changed in the Next.js code
- Why the change was necessary
- How it improves visual parity

This ensures a systematic approach to achieving perfect UI migration from Qt WebEngine to Next.js.