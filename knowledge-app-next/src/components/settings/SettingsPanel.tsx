'use client';

import React, { useState, useEffect } from 'react';
import { callPythonMethod } from '../../lib/pythonBridge';

export const SettingsPanel: React.FC = () => {
  const [darkMode, setDarkMode] = useState(false);
  const [defaultTimer, setDefaultTimer] = useState('30');
  const [showAnswers, setShowAnswers] = useState(true);
  const [saveQuestions, setSaveQuestions] = useState(true);
  const [mcqModel, setMcqModel] = useState('');
  const [thinkingModel, setThinkingModel] = useState('');
  const [selectionStrategy, setSelectionStrategy] = useState('intelligent');
  const [isAnimating, setIsAnimating] = useState(false);
  const [activeSection, setActiveSection] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);

  // API Provider states - EXACT Qt structure
  const [providers, setProviders] = useState({
    openai: { enabled: true, key: '', status: '❌', testing: false },
    anthropic: { enabled: true, key: '', status: '❌', testing: false },
    google: { enabled: true, key: '', status: '❌', testing: false },
    groq: { enabled: true, key: '', status: '❌', testing: false },
    openrouter: { enabled: true, key: '', status: '❌', testing: false },
    tavily: { enabled: true, key: '', status: '❌', testing: false }
  });

  const [savedNotification, setSavedNotification] = useState(false);

  useEffect(() => {
    // Animate sections on mount
    const sections = document.querySelectorAll('.settings-section');
    sections.forEach((section, index) => {
      setTimeout(() => {
        section.classList.add('animate-in');
      }, index * 150);
    });

    const fetchSettings = async () => {
      const settings = await callPythonMethod('get_settings');
      setProviders(settings.providers);
      setDarkMode(settings.userPreferences.darkMode);
      setDefaultTimer(settings.userPreferences.defaultTimer);
      setShowAnswers(settings.userPreferences.showAnswers);
      setSaveQuestions(settings.userPreferences.saveQuestions);
      setMcqModel(settings.modelPreferences.mcqModel);
      setThinkingModel(settings.modelPreferences.thinkingModel);
      setSelectionStrategy(settings.modelPreferences.selectionStrategy);
    };
    fetchSettings();
  }, []);

  const showSavedNotification = () => {
    setSavedNotification(true);
    setTimeout(() => setSavedNotification(false), 2000);
  };

  const handleProviderChange = (provider: string, field: string, value: any) => {
    setProviders(prev => ({
      ...prev,
      [provider]: { ...prev[provider], [field]: value }
    }));
    
    // Animate the provider card
    const card = document.querySelector(`[data-provider="${provider}"]`);
    if (card) {
      card.classList.add('provider-updated');
      setTimeout(() => card.classList.remove('provider-updated'), 300);
    }
    
    console.log('Provider settings updated'); // Qt compatibility
  };

  const testProvider = async (provider: string) => {
    setProviders(prev => ({
      ...prev,
      [provider]: { ...prev[provider], testing: true }
    }));

    const key = providers[provider].key;
    const success = await callPythonMethod('test_api_key', provider, key);

    setProviders(prev => ({
      ...prev,
      [provider]: { 
        ...prev[provider], 
        testing: false,
        status: success ? '✅' : '❌'
      }
    }));
  };

  const saveSettings = async () => {
    setIsAnimating(true);
    await callPythonMethod('save_settings', { providers });
    await saveUserPreferences();
    await saveModelPreferences();
    setTimeout(() => {
      setIsAnimating(false);
      showSavedNotification();
    }, 600);
    console.log('Settings saved'); // Qt compatibility placeholder
  };

  const testAllProviders = () => {
    Object.keys(providers).forEach(provider => testProvider(provider));
    console.log('Testing all APIs...'); // Qt compatibility placeholder
  };

  const clearAllApiKeys = () => {
    const updatedProviders = { ...providers };
    Object.keys(updatedProviders).forEach(provider => {
      updatedProviders[provider].key = '';
      updatedProviders[provider].status = '❌';
    });
    setProviders(updatedProviders);
    showSavedNotification();
    console.log('Clearing all API keys...'); // Qt compatibility placeholder
  };

  const resetApiKeysAdvanced = () => {
    console.log('Advanced reset...'); // Qt compatibility placeholder
  };

  const validateAllApiKeys = () => {
    console.log('Validating all API keys...'); // Qt compatibility placeholder
  };

  const refreshAvailableModels = async () => {
    const models = await callPythonMethod('get_available_models');
    setAvailableModels(models);
  };

  const saveModelPreferences = async () => {
    await callPythonMethod('save_model_preferences', { mcqModel, thinkingModel, selectionStrategy });
    showSavedNotification();
    console.log('Model preferences saved'); // Qt compatibility placeholder
  };

  const saveUserPreferences = async () => {
    await callPythonMethod('save_user_preferences', { darkMode, defaultTimer, showAnswers, saveQuestions });
  };

  const toggleTheme = () => {
    setDarkMode(!darkMode);
    showSavedNotification();
    console.log('Theme toggled'); // Qt compatibility placeholder
  };

  const toggleSection = (sectionId: string) => {
    setActiveSection(activeSection === sectionId ? null : sectionId);
  };

  const getProviderIcon = (provider: string) => {
    const icons = {
      openai: '🧠',
      anthropic: '🔮',
      google: '🎯',
      groq: '⚡',
      openrouter: '🌐',
      tavily: '🔍'
    };
    return icons[provider] || '🤖';
  };

  const getProviderName = (provider: string) => {
    const names = {
      openai: 'OpenAI',
      anthropic: 'Anthropic',
      google: 'Google Gemini',
      groq: 'Groq',
      openrouter: 'OpenRouter',
      tavily: 'Tavily Search'
    };
    return names[provider] || provider;
  };

  const getProviderPlaceholder = (provider: string) => {
    const placeholders = {
      openai: 'sk-...',
      anthropic: 'sk-ant-...',
      google: 'AIza...',
      groq: 'gsk_...',
      openrouter: 'sk-or-...',
      tavily: 'tvly-...'
    };
    return placeholders[provider] || 'api-key...';
  };

  const getProviderDescription = (provider: string) => {
    const descriptions = {
      openai: 'Powers GPT-4, GPT-3.5, and other OpenAI models',
      anthropic: 'Powers Claude 3.5 Sonnet and other Claude models',
      google: 'Powers Gemini Pro and other Google AI models',
      groq: 'Ultra-fast inference with Mixtral and Llama models',
      openrouter: 'Access to multiple models through one API',
      tavily: 'Real-time web search for up-to-date quiz content'
    };
    return descriptions[provider] || 'AI Provider';
  };

  return (
    <div className="settings-container enhanced">
      {/* Saved Notification */}
      <div className={`save-notification ${savedNotification ? 'show' : ''}`}>
        <span className="save-icon">💾</span>
        <span>Settings Saved Successfully!</span>
      </div>

      <div className="settings-header">
        <h2 className="settings-title">
          <span className="title-icon">⚙️</span>
          Settings
        </h2>
        <p className="settings-subtitle">Configure your quiz application preferences</p>
      </div>
      
      {/* API Configuration Section - Enhanced */}
      <div className="settings-section" data-section="api">
        <div 
          className="section-header clickable"
          onClick={() => toggleSection('api')}
        >
          <h3>
            <span className="section-icon">🤖</span>
            AI Providers
            <span className={`expand-icon ${activeSection === 'api' ? 'rotated' : ''}`}>▼</span>
          </h3>
        </div>
        
        <div className={`section-content ${activeSection === 'api' || activeSection === null ? 'expanded' : 'collapsed'}`}>
          <p className="section-description">Configure API keys for cloud AI providers. Keys are stored locally and encrypted.</p>
          
          <div className="api-providers-grid enhanced">
            {Object.entries(providers).map(([provider, config]) => (
              <div 
                key={provider}
                className={`api-provider-card enhanced ${config.testing ? 'testing' : ''}`}
                data-provider={provider}
              >
                <div className="provider-header">
                  <span className="provider-icon">{getProviderIcon(provider)}</span>
                  <span className="provider-name">{getProviderName(provider)}</span>
                  <span className={`provider-status ${config.testing ? 'testing' : ''}`}>
                    {config.testing ? '🔄' : config.status}
                  </span>
                  <label className="provider-toggle enhanced">
                    <input 
                      type="checkbox" 
                      checked={config.enabled}
                      onChange={(e) => handleProviderChange(provider, 'enabled', e.target.checked)}
                    />
                    <span className="toggle-slider"></span>
                    <span className="toggle-text">Enable</span>
                  </label>
                </div>
                
                <div className="provider-input-group">
                  <input 
                    type="password" 
                    placeholder={getProviderPlaceholder(provider)}
                    className="api-key-input enhanced"
                    value={config.key}
                    onChange={(e) => handleProviderChange(provider, 'key', e.target.value)}
                  />
                  <button 
                    className="test-btn"
                    onClick={() => testProvider(provider)}
                    disabled={config.testing || !config.key}
                    title="Test API Key"
                  >
                    {config.testing ? '🔄' : '🧪'}
                  </button>
                </div>
                
                <div className="provider-info">
                  <small>{getProviderDescription(provider)}</small>
                </div>
              </div>
            ))}
          </div>
          
          <div className="api-actions enhanced">
            <button className="btn btn-secondary enhanced" onClick={testAllProviders}>
              <span className="btn-icon">🧪</span>
              Test All APIs
            </button>
            <button className="btn btn-secondary enhanced" onClick={clearAllApiKeys}>
              <span className="btn-icon">🗑️</span>
              Clear All Keys
            </button>
            <button className="btn btn-warning enhanced" onClick={resetApiKeysAdvanced}>
              <span className="btn-icon">🔄</span>
              Advanced Reset
            </button>
            <button className="btn btn-info enhanced" onClick={validateAllApiKeys}>
              <span className="btn-icon">🔍</span>
              Validate Keys
            </button>
          </div>
          
          {/* API Key Management Status */}
          <div className="api-status-summary enhanced">
            <div className="status-indicators">
              <div className="status-item">
                <span className="status-icon">💾</span>
                <span>Auto-save: Active</span>
              </div>
              <div className="status-item">
                <span className="status-icon">🔄</span>
                <span>Session backup: Ready</span>
              </div>
            </div>
          </div>
          
          <div className="api-help enhanced">
            <details>
              <summary>
                <span className="help-icon">🔑</span>
                Where to get API keys
              </summary>
              <div className="help-content">
                <ul>
                  <li><strong>OpenAI:</strong> <a href="https://platform.openai.com/api-keys" target="_blank">platform.openai.com/api-keys</a></li>
                  <li><strong>Anthropic:</strong> <a href="https://console.anthropic.com/settings/keys" target="_blank">console.anthropic.com/settings/keys</a></li>
                  <li><strong>Google:</strong> <a href="https://aistudio.google.com/app/apikey" target="_blank">aistudio.google.com/app/apikey</a></li>
                  <li><strong>Groq:</strong> <a href="https://console.groq.com/keys" target="_blank">console.groq.com/keys</a></li>
                  <li><strong>OpenRouter:</strong> <a href="https://openrouter.ai/keys" target="_blank">openrouter.ai/keys</a></li>
                  <li><strong>Tavily:</strong> <a href="https://tavily.com" target="_blank">tavily.com</a> (Free tier available)</li>
                </ul>
              </div>
            </details>
          </div>
        </div>
      </div>
      
      {/* Appearance Section */}
      <div className="settings-section" data-section="appearance">
        <div 
          className="section-header clickable"
          onClick={() => toggleSection('appearance')}
        >
          <h3>
            <span className="section-icon">🎨</span>
            Appearance
            <span className={`expand-icon ${activeSection === 'appearance' ? 'rotated' : ''}`}>▼</span>
          </h3>
        </div>
        
        <div className={`section-content ${activeSection === 'appearance' || activeSection === null ? 'expanded' : 'collapsed'}`}>
          <div className="setting-item enhanced">
            <div className="setting-info">
              <label className="setting-label">
                <span className="setting-icon">🌙</span>
                Dark Mode
              </label>
              <small className="setting-description">Switch between light and dark themes</small>
            </div>
            <label className="theme-toggle">
              <input 
                type="checkbox" 
                checked={darkMode}
                onChange={toggleTheme}
              />
              <span className="theme-slider">
                <span className="theme-icon sun">☀️</span>
                <span className="theme-icon moon">🌙</span>
              </span>
            </label>
          </div>
        </div>
      </div>
      
      {/* Quiz Settings Section */}
      <div className="settings-section" data-section="quiz">
        <div 
          className="section-header clickable"
          onClick={() => toggleSection('quiz')}
        >
          <h3>
            <span className="section-icon">🎯</span>
            Quiz Settings
            <span className={`expand-icon ${activeSection === 'quiz' ? 'rotated' : ''}`}>▼</span>
          </h3>
        </div>
        
        <div className={`section-content ${activeSection === 'quiz' || activeSection === null ? 'expanded' : 'collapsed'}`}>
          <div className="setting-item enhanced">
            <div className="setting-info">
              <label className="setting-label">
                <span className="setting-icon">⏱️</span>
                Default Timer (seconds)
              </label>
              <small className="setting-description">Set the default time limit for quiz questions</small>
            </div>
            <div className="number-input-group">
              <input 
                type="number" 
                value={defaultTimer} 
                min="10" 
                max="120" 
                className="number-input enhanced"
                onChange={(e) => {
                  setDefaultTimer(e.target.value);
                  saveSettings();
                }}
              />
              <span className="input-unit">sec</span>
            </div>
          </div>
          
          <div className="setting-item enhanced">
            <div className="setting-info">
              <label className="setting-label">
                <span className="setting-icon">✅</span>
                Show Correct Answers
              </label>
              <small className="setting-description">Display correct answers after quiz completion</small>
            </div>
            <label className="custom-checkbox">
              <input 
                type="checkbox" 
                checked={showAnswers}
                onChange={(e) => {
                  setShowAnswers(e.target.checked);
                  saveSettings();
                }}
              />
              <span className="checkmark"></span>
            </label>
          </div>
          
          <div className="setting-item enhanced">
            <div className="setting-info">
              <label className="setting-label">
                <span className="setting-icon">📚</span>
                Save Generated Questions
              </label>
              <small className="setting-description">Automatically save all generated questions to review history for later practice</small>
            </div>
            <label className="custom-checkbox">
              <input 
                type="checkbox" 
                checked={saveQuestions}
                onChange={(e) => {
                  setSaveQuestions(e.target.checked);
                  saveSettings();
                }}
              />
              <span className="checkmark"></span>
            </label>
          </div>
        </div>
      </div>

      {/* Model Selection Section */}
      <div className="settings-section" data-section="models">
        <div 
          className="section-header clickable"
          onClick={() => toggleSection('models')}
        >
          <h3>
            <span className="section-icon">🤖</span>
            Model Selection & Preferences
            <span className={`expand-icon ${activeSection === 'models' ? 'rotated' : ''}`}>▼</span>
          </h3>
        </div>
        
        <div className={`section-content ${activeSection === 'models' || activeSection === null ? 'expanded' : 'collapsed'}`}>
          <div className="setting-item enhanced">
            <div className="setting-info">
              <label className="setting-label">
                <span className="setting-icon">🔄</span>
                Available Models
              </label>
              <small className="setting-description">Refresh the list of available AI models</small>
            </div>
            <button className="btn btn-secondary enhanced" onClick={refreshAvailableModels}>
              <span className="btn-icon">🔄</span>
              Refresh Available Models
            </button>
          </div>
          
          <div className="models-list enhanced">
            {availableModels.length > 0 ? (
              <ul>
                {availableModels.map(model => (
                  <li key={model}>{model}</li>
                ))}
              </ul>
            ) : (
              <p className="loading-text">Click "Refresh" to load available models...</p>
            )}
          </div>
          
          <div className="setting-item enhanced">
            <div className="setting-info">
              <label className="setting-label">
                <span className="setting-icon">❓</span>
                Preferred Model for MCQ Generation
              </label>
              <small className="setting-description">Choose your preferred model for generating quiz questions</small>
            </div>
            <select 
              value={mcqModel} 
              className="custom-select enhanced"
              onChange={(e) => {
                setMcqModel(e.target.value);
                saveModelPreferences();
              }}
            >
              <option value="">Auto-select best available model</option>
              {availableModels.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>
          
          <div className="setting-item enhanced">
            <div className="setting-info">
              <label className="setting-label">
                <span className="setting-icon">🧠</span>
                Preferred Model for Expert/Thinking Mode
              </label>
              <small className="setting-description">Choose your preferred model for complex reasoning tasks</small>
            </div>
            <select 
              value={thinkingModel} 
              className="custom-select enhanced"
              onChange={(e) => {
                setThinkingModel(e.target.value);
                saveModelPreferences();
              }}
            >
              <option value="">Auto-select best thinking model</option>
              {availableModels.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>
          
          <div className="setting-item enhanced">
            <div className="setting-info">
              <label className="setting-label">
                <span className="setting-icon">🎯</span>
                Model Selection Strategy
              </label>
              <small className="setting-description">How the system should choose models when multiple options are available</small>
            </div>
            <select 
              value={selectionStrategy} 
              className="custom-select enhanced"
              onChange={(e) => {
                setSelectionStrategy(e.target.value);
                saveModelPreferences();
              }}
            >
              <option value="intelligent">🤖 Intelligent (Recommended)</option>
              <option value="user_preference">👤 User Preference Only</option>
              <option value="first_available">⚡ First Available</option>
            </select>
          </div>
        </div>
      </div>

      {/* Save Button */}
      <div className="settings-footer">
        <button 
          className={`btn btn-primary enhanced save-all-btn ${isAnimating ? 'saving' : ''}`} 
          onClick={saveSettings}
          disabled={isAnimating}
        >
          <span className="btn-icon">💾</span>
          {isAnimating ? 'Saving...' : 'Save All Settings'}
        </button>
      </div>
    </div>
  );
};
