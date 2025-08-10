# Knowledge App Next.js Frontend

This is the Next.js frontend for the Knowledge App, designed to replace the Qt WebEngine UI. It provides a modern, responsive, and interactive user interface for the application.

## 🚀 Getting Started

To run the development server:

1. Navigate to the `knowledge-app-next` directory:
   ```bash
   cd knowledge-app-next
   ```

2. Install the dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

   This will start the Next.js development server and the API bridge. Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

### Connect to Python backend (Flask bridge)

The frontend talks to the Python bridge at `NEXT_PUBLIC_PYTHON_BRIDGE_URL` (default `http://localhost:8000`).

1) Ensure Python bridge deps are installed:
    - From the `knowledge-app-next` folder run:
       - Windows PowerShell:
          ```powershell
          pip install -r api-server/requirements.txt
          ```

2) Start both servers in dev:
    - Next.js + bridge (concurrently) via:
       ```powershell
       npm run dev
       ```
    - Or start bridge only:
       ```powershell
       python api-server/bridge-server.py
       ```

3) Verify health:
    - Frontend proxy: http://localhost:3000/health
    - Bridge direct:  http://localhost:8000/health

## 📂 Project Structure

- `src/app`: Contains the main application pages.
- `src/components`: Reusable UI components.
- `src/context`: React Context API for global state management (e.g., screen navigation).
- `src/hooks`: Custom React hooks.
- `src/lib`: Utility functions and Python bridge integration.
- `public`: Static assets.

## ✨ Features Implemented

- **Basic Next.js Project Setup**: Project structure created with TypeScript.
- **Core Layout Components**: `MainLayout`, `Sidebar`, `TopBar` components implemented.
- **Screen Navigation System**: `ScreenContext` and navigation logic working.
- **Theme System**: Dark/light theme toggle with CSS variables.
- **Basic Styling Foundation**: `globals.css` with Qt-inspired styling.
- **Home Screen Structure**: Welcome card and stats grid implemented.
- **Quiz Screen Basic Structure**: Quiz setup page created.
- **Review Screen**: Basic review page implemented.
- **Settings Screen**: Settings page created.
- **Training Screen**: Training page with components.
- **MathJax Integration**: LaTeX rendering support added.
- **Socket.io Setup**: WebSocket communication for real-time updates.
- **Python Bridge Integration**: Basic communication layer implemented.
- **AppLogger System**: Basic logging system implemented.
- **Question History System**: Basic components for displaying question history.
- **File Upload System**: Basic components for file uploads.
- **Training Progress Visualization**: Enhanced components for displaying training progress.
- **Error Handling and Status System**: Basic components for error boundaries and status displays.
- **Responsive Design**: Initial refactoring for responsive layouts.
- **Performance Optimization**: Implemented code splitting and lazy loading for screen components.
- **Testing Implementation**: Basic unit test for `AppLogger` and `package.json` updated with a test script.

## 🛠️ Development

### Scripts

- `npm run dev`: Starts the development server.
- `npm run build`: Builds the application for production.
- `npm run start`: Starts a production Next.js server.
- `npm run lint`: Runs ESLint for code linting.
- `npm run test`: Runs Jest tests.

### Styling

This project uses Tailwind CSS for styling, configured in `tailwind.config.ts`.

## 🤝 Contributing

Contributions are welcome! Please refer to the `GEMINI.md` file for the migration roadmap and detailed tasks.
