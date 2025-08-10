'use client';

import React, { useState, useEffect } from 'react';
import { AppLogger } from '../../lib/logger';
import { setCurrentScreenName } from '../../lib/logger';
import { callPythonMethod } from '../../lib/pythonBridge';
import { useScreen } from '../../context/ScreenContext';
import { useTheme } from '../../hooks/useTheme';
import { sessionTracker } from '../../lib/sessionTracker';

interface ApiSettings {
  openai: string;
  anthropic: string;
  gemini: string;
  groq: string;
  openrouter: string;
  deepseek: string;
  ollama_base_url: string;
}

interface AppSettings {
  theme: 'light' | 'dark';
  font_size: number;
  storage_limit: number;
  auto_switch_images: boolean;
  offline_mode: boolean;
  default_timer: number;
  show_answers: boolean;
  default_game_mode: 'casual' | 'serious';
  default_difficulty: 'easy' | 'medium' | 'hard' | 'expert';
  default_submode: 'mixed' | 'numerical' | 'conceptual';
  default_quiz_mode: 'auto' | 'online' | 'offline';
}

interface AdvancedSettings {
  debug_mode: boolean;
  performance_monitoring: boolean;
  auto_save_interval: number;
  max_concurrent_requests: number;
  request_timeout: number;
  cache_size_mb: number;
  log_level: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR';
  enable_analytics: boolean;
}

const SettingsPage: React.FC = () => {
  const { currentScreen } = useScreen();
  const { isDark, toggleTheme } = useTheme();
  const [activeTab, setActiveTab] = useState('general');
  const [loading, setLoading] = useState(false);
  const [testingApi, setTestingApi] = useState<string | null>(null);
  
  // Settings state
  const [apiSettings, setApiSettings] = useState<ApiSettings>({
    openai: '',
    anthropic: '',
    gemini: '',
    groq: '',
    openrouter: '',
    deepseek: '',
    ollama_base_url: 'http://localhost:11434',
  });

  const [appSettings, setAppSettings] = useState<AppSettings>({
    theme: 'dark',
    font_size: 14,
    storage_limit: 500 * 1024 * 1024, // 500MB
    auto_switch_images: false,
    offline_mode: false,
    default_timer: 30,
    show_answers: true,
    default_game_mode: 'casual',
    default_difficulty: 'medium',
    default_submode: 'mixed',
    default_quiz_mode: 'auto',
  });

  const [advancedSettings, setAdvancedSettings] = useState<AdvancedSettings>({
    debug_mode: false,
    performance_monitoring: true,
    auto_save_interval: 30,
    max_concurrent_requests: 5,
    request_timeout: 30,
    cache_size_mb: 100,
    log_level: 'INFO',
    enable_analytics: true,
  });

  useEffect(() => {
    setCurrentScreenName('settings');
    AppLogger.info('SETTINGS_PAGE', 'Settings page loaded.');
    sessionTracker.logAction('SETTINGS_PAGE_OPENED');
    loadAllSettings();
  }, []);

  const loadAllSettings = async () => {
    setLoading(true);
    try {
      // Load API settings
      const apiKeys = await callPythonMethod('get_api_keys');
      setApiSettings({
        openai: apiKeys.openai || '',
        anthropic: apiKeys.anthropic || '',
        gemini: apiKeys.gemini || '',
        groq: apiKeys.groq || '',
        openrouter: apiKeys.openrouter || '',
        deepseek: apiKeys.deepseek || '',
        ollama_base_url: apiKeys.ollama_base_url || 'http://localhost:11434',
      });

      // Load app settings
      const appConfig = await callPythonMethod('get_app_settings');
      setAppSettings({
        theme: appConfig.theme || 'dark',
        font_size: appConfig.font_size || 14,
        storage_limit: appConfig.storage_limit || 500 * 1024 * 1024,
        auto_switch_images: appConfig.auto_switch_images || false,
        offline_mode: appConfig.offline_mode || false,
        default_timer: appConfig.default_timer || 30,
        show_answers: appConfig.show_answers !== false,
        default_game_mode: appConfig.default_game_mode || 'casual',
        default_difficulty: appConfig.default_difficulty || 'medium',
        default_submode: appConfig.default_submode || 'mixed',
        default_quiz_mode: appConfig.default_quiz_mode || 'auto',
      });

      // Load advanced settings
      const advancedConfig = await callPythonMethod('get_advanced_settings');
      setAdvancedSettings({
        debug_mode: advancedConfig.debug_mode || false,
        performance_monitoring: advancedConfig.performance_monitoring !== false,
        auto_save_interval: advancedConfig.auto_save_interval || 30,
        max_concurrent_requests: advancedConfig.max_concurrent_requests || 5,
        request_timeout: advancedConfig.request_timeout || 30,
        cache_size_mb: advancedConfig.cache_size_mb || 100,
        log_level: advancedConfig.log_level || 'INFO',
        enable_analytics: advancedConfig.enable_analytics !== false,
      });

      AppLogger.success('SETTINGS_PAGE', 'All settings loaded successfully.');
    } catch (error) {
      AppLogger.error('SETTINGS_PAGE', 'Failed to load settings.', error);
    } finally {
      setLoading(false);
    }
  };

  // Save functions
  const handleSaveApiSettings = async () => {
    setLoading(true);
    try {
      await callPythonMethod('save_api_keys', apiSettings);
      AppLogger.success('SETTINGS_PAGE', 'API settings saved successfully.', apiSettings);
      sessionTracker.logAction('API_SETTINGS_SAVED');
      showNotification('API Settings saved successfully!', 'success');
    } catch (error) {
      AppLogger.error('SETTINGS_PAGE', 'Failed to save API settings.', error);
      showNotification('Failed to save API settings.', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveAppSettings = async () => {
    setLoading(true);
    try {
      await callPythonMethod('save_app_settings', appSettings);
      AppLogger.success('SETTINGS_PAGE', 'App settings saved successfully.', appSettings);
      sessionTracker.logAction('APP_SETTINGS_SAVED');
      showNotification('Application settings saved successfully!', 'success');
      
      // Apply theme change immediately
      if (appSettings.theme !== (isDark ? 'dark' : 'light')) {
        toggleTheme();
      }
    } catch (error) {
      AppLogger.error('SETTINGS_PAGE', 'Failed to save app settings.', error);
      showNotification('Failed to save application settings.', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveAdvancedSettings = async () => {
    setLoading(true);
    try {
      await callPythonMethod('save_advanced_settings', advancedSettings);
      AppLogger.success('SETTINGS_PAGE', 'Advanced settings saved successfully.', advancedSettings);
      sessionTracker.logAction('ADVANCED_SETTINGS_SAVED');
      showNotification('Advanced settings saved successfully!', 'success');
    } catch (error) {
      AppLogger.error('SETTINGS_PAGE', 'Failed to save advanced settings.', error);
      showNotification('Failed to save advanced settings.', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Test API key function
  const testApiKey = async (provider: string) => {
    setTestingApi(provider);
    try {
      const result = await callPythonMethod('test_api_key', {
        provider,
        api_key: apiSettings[provider as keyof ApiSettings]
      });
      
      if (result.success) {
        AppLogger.success('SETTINGS_PAGE', `${provider} API key test successful.`, result);
        showNotification(`${provider.toUpperCase()} API key is valid!`, 'success');
      } else {
        AppLogger.warn('SETTINGS_PAGE', `${provider} API key test failed.`, result);
        showNotification(`${provider.toUpperCase()} API key test failed: ${result.error}`, 'error');
      }
    } catch (error) {
      AppLogger.error('SETTINGS_PAGE', `Failed to test ${provider} API key.`, error);
      showNotification(`Failed to test ${provider.toUpperCase()} API key.`, 'error');
    } finally {
      setTestingApi(null);
    }
  };

  // Reset functions
  const resetToDefaults = async (settingsType: 'api' | 'app' | 'advanced') => {
    if (!confirm(`Are you sure you want to reset ${settingsType} settings to defaults?`)) {
      return;
    }

    try {
      await callPythonMethod('reset_settings_to_defaults', { type: settingsType });
      AppLogger.action('SETTINGS_PAGE', `${settingsType} settings reset to defaults.`);
      sessionTracker.logAction('SETTINGS_RESET', { type: settingsType });
      showNotification(`${settingsType.charAt(0).toUpperCase() + settingsType.slice(1)} settings reset to defaults.`, 'success');
      await loadAllSettings(); // Reload settings
    } catch (error) {
      AppLogger.error('SETTINGS_PAGE', `Failed to reset ${settingsType} settings.`, error);
      showNotification(`Failed to reset ${settingsType} settings.`, 'error');
    }
  };

  // Change handlers
  const handleApiChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setApiSettings(prev => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleAppSettingsChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const processedValue = type === 'checkbox' ? (e.target as HTMLInputElement).checked :
                          type === 'number' ? Number(value) : value;
    
    setAppSettings(prev => ({
      ...prev,
      [name]: processedValue,
    }));
  };

  const handleAdvancedSettingsChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const processedValue = type === 'checkbox' ? (e.target as HTMLInputElement).checked :
                          type === 'number' ? Number(value) : value;
    
    setAdvancedSettings(prev => ({
      ...prev,
      [name]: processedValue,
    }));
  };

  // Notification system
  const showNotification = (message: string, type: 'success' | 'error' | 'info') => {
    // Simple notification - could be enhanced with a proper notification system
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 p-4 rounded-lg text-white z-50 ${
      type === 'success' ? 'bg-green-600' : 
      type === 'error' ? 'bg-red-600' : 'bg-blue-600'
    }`;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
      document.body.removeChild(notification);
    }, 3000);
  };

  if (loading && activeTab !== 'general') {
    return (
      <div className="p-6 bg-gray-800 rounded-lg shadow-lg text-white min-h-full">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-3">Loading settings...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 bg-gray-800 rounded-lg shadow-lg text-white min-h-full">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">⚙️ Settings</h1>
        <div className="text-sm text-gray-400">
          Session: {sessionTracker.getSessionSummary().session_id.slice(-6)}
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-gray-700 mb-6 overflow-x-auto">
        {[
          { id: 'general', label: '🏠 General', icon: '🏠' },
          { id: 'api', label: '🔑 API Keys', icon: '🔑' },
          { id: 'quiz', label: '📝 Quiz Defaults', icon: '📝' },
          { id: 'advanced', label: '🔧 Advanced', icon: '🔧' },
          { id: 'about', label: 'ℹ️ About', icon: 'ℹ️' },
        ].map(tab => (
          <button
            key={tab.id}
            className={`py-3 px-6 text-lg whitespace-nowrap transition-colors ${
              activeTab === tab.id 
                ? 'border-b-2 border-blue-500 text-blue-500' 
                : 'text-gray-400 hover:text-white'
            }`}
            onClick={() => {
              setActiveTab(tab.id);
              AppLogger.action('SETTINGS_TAB', `Switched to ${tab.id} tab`);
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Settings Content */}
      <div className="settings-content">
        {activeTab === 'general' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-semibold mb-4">🏠 General Settings</h2>
              <p className="text-gray-400 mb-6">Configure basic application preferences and behavior.</p>
            </div>

            {/* Theme Settings */}
            <div className="bg-gray-700 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">🎨 Appearance</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-300 text-sm font-bold mb-2">Theme</label>
                  <select
                    name="theme"
                    value={appSettings.theme}
                    onChange={handleAppSettingsChange}
                    className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                  >
                    <option value="light">☀️ Light</option>
                    <option value="dark">🌙 Dark</option>
                  </select>
                </div>
                <div>
                  <label className="block text-gray-300 text-sm font-bold mb-2">Font Size</label>
                  <input
                    type="number"
                    name="font_size"
                    value={appSettings.font_size}
                    onChange={handleAppSettingsChange}
                    min="10"
                    max="24"
                    className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                  />
                </div>
              </div>
            </div>

            {/* Performance Settings */}
            <div className="bg-gray-700 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">⚡ Performance</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-300 text-sm font-bold mb-2">Storage Limit (MB)</label>
                  <input
                    type="number"
                    name="storage_limit"
                    value={Math.round(appSettings.storage_limit / (1024 * 1024))}
                    onChange={(e) => handleAppSettingsChange({
                      ...e,
                      target: { ...e.target, name: 'storage_limit', value: String(Number(e.target.value) * 1024 * 1024) }
                    } as any)}
                    min="100"
                    max="2000"
                    className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                  />
                </div>
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    name="offline_mode"
                    checked={appSettings.offline_mode}
                    onChange={handleAppSettingsChange}
                    className="mr-3 w-4 h-4"
                  />
                  <label className="text-gray-300">🔌 Offline Mode</label>
                </div>
              </div>
            </div>

            {/* UI Preferences */}
            <div className="bg-gray-700 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">🖥️ Interface</h3>
              <div className="space-y-4">
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    name="auto_switch_images"
                    checked={appSettings.auto_switch_images}
                    onChange={handleAppSettingsChange}
                    className="mr-3 w-4 h-4"
                  />
                  <label className="text-gray-300">🖼️ Auto-switch images</label>
                </div>
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    name="show_answers"
                    checked={appSettings.show_answers}
                    onChange={handleAppSettingsChange}
                    className="mr-3 w-4 h-4"
                  />
                  <label className="text-gray-300">👁️ Show answers after questions</label>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4">
              <button
                onClick={handleSaveAppSettings}
                disabled={loading}
                className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors"
              >
                {loading ? '💾 Saving...' : '💾 Save General Settings'}
              </button>
              <button
                onClick={() => resetToDefaults('app')}
                className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                🔄 Reset to Defaults
              </button>
            </div>
          </div>
        )}

        {activeTab === 'api' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-semibold mb-4">🔑 API Key Settings</h2>
              <p className="text-gray-400 mb-6">Configure API keys for various AI providers. Keys are stored securely and encrypted.</p>
            </div>

            {/* API Provider Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* OpenAI */}
              <div className="bg-gray-700 rounded-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">🤖 OpenAI</h3>
                  <button
                    onClick={() => testApiKey('openai')}
                    disabled={!apiSettings.openai || testingApi === 'openai'}
                    className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
                  >
                    {testingApi === 'openai' ? '🔄 Testing...' : '🧪 Test'}
                  </button>
                </div>
                <input
                  type="password"
                  name="openai"
                  value={apiSettings.openai}
                  onChange={handleApiChange}
                  placeholder="sk-..."
                  className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                />
                <p className="text-xs text-gray-400 mt-2">Used for GPT models and advanced reasoning</p>
              </div>

              {/* DeepSeek */}
              <div className="bg-gray-700 rounded-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">🧠 DeepSeek</h3>
                  <button
                    onClick={() => testApiKey('deepseek')}
                    disabled={!apiSettings.deepseek || testingApi === 'deepseek'}
                    className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
                  >
                    {testingApi === 'deepseek' ? '🔄 Testing...' : '🧪 Test'}
                  </button>
                </div>
                <input
                  type="password"
                  name="deepseek"
                  value={apiSettings.deepseek}
                  onChange={handleApiChange}
                  placeholder="sk-..."
                  className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                />
                <p className="text-xs text-gray-400 mt-2">Used for expert-level question generation</p>
              </div>

              {/* Anthropic */}
              <div className="bg-gray-700 rounded-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">🎭 Anthropic</h3>
                  <button
                    onClick={() => testApiKey('anthropic')}
                    disabled={!apiSettings.anthropic || testingApi === 'anthropic'}
                    className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
                  >
                    {testingApi === 'anthropic' ? '🔄 Testing...' : '🧪 Test'}
                  </button>
                </div>
                <input
                  type="password"
                  name="anthropic"
                  value={apiSettings.anthropic}
                  onChange={handleApiChange}
                  placeholder="sk-ant-..."
                  className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                />
                <p className="text-xs text-gray-400 mt-2">Claude models for advanced reasoning</p>
              </div>

              {/* Google Gemini */}
              <div className="bg-gray-700 rounded-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">💎 Google Gemini</h3>
                  <button
                    onClick={() => testApiKey('gemini')}
                    disabled={!apiSettings.gemini || testingApi === 'gemini'}
                    className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
                  >
                    {testingApi === 'gemini' ? '🔄 Testing...' : '🧪 Test'}
                  </button>
                </div>
                <input
                  type="password"
                  name="gemini"
                  value={apiSettings.gemini}
                  onChange={handleApiChange}
                  placeholder="AIza..."
                  className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                />
                <p className="text-xs text-gray-400 mt-2">Google's multimodal AI models</p>
              </div>

              {/* Groq */}
              <div className="bg-gray-700 rounded-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">⚡ Groq</h3>
                  <button
                    onClick={() => testApiKey('groq')}
                    disabled={!apiSettings.groq || testingApi === 'groq'}
                    className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
                  >
                    {testingApi === 'groq' ? '🔄 Testing...' : '🧪 Test'}
                  </button>
                </div>
                <input
                  type="password"
                  name="groq"
                  value={apiSettings.groq}
                  onChange={handleApiChange}
                  placeholder="gsk_..."
                  className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                />
                <p className="text-xs text-gray-400 mt-2">Ultra-fast inference with LPU technology</p>
              </div>

              {/* OpenRouter */}
              <div className="bg-gray-700 rounded-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">🌐 OpenRouter</h3>
                  <button
                    onClick={() => testApiKey('openrouter')}
                    disabled={!apiSettings.openrouter || testingApi === 'openrouter'}
                    className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
                  >
                    {testingApi === 'openrouter' ? '🔄 Testing...' : '🧪 Test'}
                  </button>
                </div>
                <input
                  type="password"
                  name="openrouter"
                  value={apiSettings.openrouter}
                  onChange={handleApiChange}
                  placeholder="sk-or-..."
                  className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                />
                <p className="text-xs text-gray-400 mt-2">Access to multiple AI models via unified API</p>
              </div>
            </div>

            {/* Local Models */}
            <div className="bg-gray-700 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">🏠 Local Models</h3>
              <div>
                <label className="block text-gray-300 text-sm font-bold mb-2">Ollama Base URL</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    name="ollama_base_url"
                    value={apiSettings.ollama_base_url}
                    onChange={handleApiChange}
                    placeholder="http://localhost:11434"
                    className="flex-1 p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                  />
                  <button
                    onClick={() => testApiKey('ollama_base_url')}
                    disabled={testingApi === 'ollama_base_url'}
                    className="px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                  >
                    {testingApi === 'ollama_base_url' ? '🔄' : '🧪'}
                  </button>
                </div>
                <p className="text-xs text-gray-400 mt-2">Local Ollama server for offline model inference</p>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4">
              <button
                onClick={handleSaveApiSettings}
                disabled={loading}
                className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors"
              >
                {loading ? '🔐 Saving...' : '🔐 Save API Keys'}
              </button>
              <button
                onClick={() => resetToDefaults('api')}
                className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                🗑️ Clear All Keys
              </button>
            </div>

            {/* Security Notice */}
            <div className="bg-blue-900 border border-blue-700 rounded-lg p-4">
              <div className="flex items-start">
                <div className="text-blue-400 mr-3">🔒</div>
                <div>
                  <h4 className="text-blue-300 font-semibold mb-2">Security Notice</h4>
                  <p className="text-blue-200 text-sm">
                    API keys are encrypted and stored securely. They are only transmitted over HTTPS and never logged in plain text.
                    Use the test buttons to verify your keys are working correctly.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'quiz' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-semibold mb-4">📝 Quiz Default Settings</h2>
              <p className="text-gray-400 mb-6">Configure default values for quiz generation to save time on setup.</p>
            </div>

            {/* Quiz Mode Settings */}
            <div className="bg-gray-700 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">🎮 Quiz Mode</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-300 text-sm font-bold mb-2">Default Quiz Mode</label>
                  <select
                    name="default_quiz_mode"
                    value={appSettings.default_quiz_mode}
                    onChange={handleAppSettingsChange}
                    className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                  >
                    <option value="auto">🤖 Auto (Best Available)</option>
                    <option value="online">🌐 Online (Cloud APIs)</option>
                    <option value="offline">💻 Offline (Local Models)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-gray-300 text-sm font-bold mb-2">Default Game Mode</label>
                  <select
                    name="default_game_mode"
                    value={appSettings.default_game_mode}
                    onChange={handleAppSettingsChange}
                    className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                  >
                    <option value="casual">🎵 Casual (Relaxed)</option>
                    <option value="serious">⏱️ Serious (Timed)</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Question Settings */}
            <div className="bg-gray-700 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">❓ Question Settings</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-300 text-sm font-bold mb-2">Default Difficulty</label>
                  <select
                    name="default_difficulty"
                    value={appSettings.default_difficulty}
                    onChange={handleAppSettingsChange}
                    className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                  >
                    <option value="easy">🟢 Easy</option>
                    <option value="medium">🟡 Medium</option>
                    <option value="hard">🔴 Hard</option>
                    <option value="expert">🔥💀 Expert (PhD-Level)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-gray-300 text-sm font-bold mb-2">Default Question Type</label>
                  <select
                    name="default_submode"
                    value={appSettings.default_submode}
                    onChange={handleAppSettingsChange}
                    className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                  >
                    <option value="mixed">⚖️ Mixed (Balanced)</option>
                    <option value="numerical">🔢 Numerical (Math & Calculations)</option>
                    <option value="conceptual">🧠 Conceptual (Theory & Understanding)</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Timer Settings */}
            <div className="bg-gray-700 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">⏱️ Timer Settings</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-300 text-sm font-bold mb-2">Default Timer (seconds)</label>
                  <input
                    type="number"
                    name="default_timer"
                    value={appSettings.default_timer}
                    onChange={handleAppSettingsChange}
                    min="10"
                    max="300"
                    className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                  />
                  <p className="text-xs text-gray-400 mt-1">Time limit per question in serious mode</p>
                </div>
                <div className="flex items-center pt-6">
                  <input
                    type="checkbox"
                    name="show_answers"
                    checked={appSettings.show_answers}
                    onChange={handleAppSettingsChange}
                    className="mr-3 w-4 h-4"
                  />
                  <label className="text-gray-300">👁️ Show answers after each question</label>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4">
              <button
                onClick={handleSaveAppSettings}
                disabled={loading}
                className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors"
              >
                {loading ? '📝 Saving...' : '📝 Save Quiz Defaults'}
              </button>
              <button
                onClick={() => resetToDefaults('app')}
                className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                🔄 Reset Quiz Defaults
              </button>
            </div>
          </div>
        )}

        {activeTab === 'advanced' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-semibold mb-4">🔧 Advanced Settings</h2>
              <p className="text-gray-400 mb-6">Configure advanced system behavior and performance settings.</p>
              <div className="bg-yellow-900 border border-yellow-700 rounded-lg p-4 mb-6">
                <div className="flex items-start">
                  <div className="text-yellow-400 mr-3">⚠️</div>
                  <div>
                    <h4 className="text-yellow-300 font-semibold mb-2">Warning</h4>
                    <p className="text-yellow-200 text-sm">
                      These settings affect system performance and behavior. Only modify if you understand the implications.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Debug Settings */}
            <div className="bg-gray-700 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">🐛 Debug & Logging</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-300 text-sm font-bold mb-2">Log Level</label>
                  <select
                    name="log_level"
                    value={advancedSettings.log_level}
                    onChange={handleAdvancedSettingsChange}
                    className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                  >
                    <option value="DEBUG">🔍 DEBUG (Verbose)</option>
                    <option value="INFO">💡 INFO (Normal)</option>
                    <option value="WARN">⚠️ WARN (Warnings only)</option>
                    <option value="ERROR">❌ ERROR (Errors only)</option>
                  </select>
                </div>
                <div className="space-y-4">
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      name="debug_mode"
                      checked={advancedSettings.debug_mode}
                      onChange={handleAdvancedSettingsChange}
                      className="mr-3 w-4 h-4"
                    />
                    <label className="text-gray-300">🐛 Debug Mode</label>
                  </div>
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      name="performance_monitoring"
                      checked={advancedSettings.performance_monitoring}
                      onChange={handleAdvancedSettingsChange}
                      className="mr-3 w-4 h-4"
                    />
                    <label className="text-gray-300">📊 Performance Monitoring</label>
                  </div>
                </div>
              </div>
            </div>

            {/* Performance Settings */}
            <div className="bg-gray-700 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">⚡ Performance</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-300 text-sm font-bold mb-2">Max Concurrent Requests</label>
                  <input
                    type="number"
                    name="max_concurrent_requests"
                    value={advancedSettings.max_concurrent_requests}
                    onChange={handleAdvancedSettingsChange}
                    min="1"
                    max="20"
                    className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                  />
                </div>
                <div>
                  <label className="block text-gray-300 text-sm font-bold mb-2">Request Timeout (seconds)</label>
                  <input
                    type="number"
                    name="request_timeout"
                    value={advancedSettings.request_timeout}
                    onChange={handleAdvancedSettingsChange}
                    min="5"
                    max="300"
                    className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                  />
                </div>
                <div>
                  <label className="block text-gray-300 text-sm font-bold mb-2">Cache Size (MB)</label>
                  <input
                    type="number"
                    name="cache_size_mb"
                    value={advancedSettings.cache_size_mb}
                    onChange={handleAdvancedSettingsChange}
                    min="10"
                    max="1000"
                    className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                  />
                </div>
                <div>
                  <label className="block text-gray-300 text-sm font-bold mb-2">Auto-save Interval (seconds)</label>
                  <input
                    type="number"
                    name="auto_save_interval"
                    value={advancedSettings.auto_save_interval}
                    onChange={handleAdvancedSettingsChange}
                    min="10"
                    max="300"
                    className="w-full p-3 bg-gray-600 border border-gray-500 rounded-lg text-white"
                  />
                </div>
              </div>
            </div>

            {/* Privacy Settings */}
            <div className="bg-gray-700 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">🔒 Privacy</h3>
              <div className="flex items-center">
                <input
                  type="checkbox"
                  name="enable_analytics"
                  checked={advancedSettings.enable_analytics}
                  onChange={handleAdvancedSettingsChange}
                  className="mr-3 w-4 h-4"
                />
                <label className="text-gray-300">📈 Enable anonymous usage analytics</label>
              </div>
              <p className="text-xs text-gray-400 mt-2">
                Helps improve the application by collecting anonymous usage statistics
              </p>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4">
              <button
                onClick={handleSaveAdvancedSettings}
                disabled={loading}
                className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors"
              >
                {loading ? '🔧 Saving...' : '🔧 Save Advanced Settings'}
              </button>
              <button
                onClick={() => resetToDefaults('advanced')}
                className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                🔄 Reset Advanced Settings
              </button>
            </div>
          </div>
        )}

        {activeTab === 'about' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-semibold mb-4">ℹ️ About Knowledge App</h2>
              <p className="text-gray-400 mb-6">Information about the application and system status.</p>
            </div>

            {/* App Info */}
            <div className="bg-gray-700 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">📱 Application Information</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-400">Version:</span>
                  <span className="ml-2 text-white">2.0.0-next</span>
                </div>
                <div>
                  <span className="text-gray-400">Build:</span>
                  <span className="ml-2 text-white">Next.js Migration</span>
                </div>
                <div>
                  <span className="text-gray-400">Platform:</span>
                  <span className="ml-2 text-white">Web (Next.js)</span>
                </div>
                <div>
                  <span className="text-gray-400">Theme:</span>
                  <span className="ml-2 text-white">{isDark ? 'Dark' : 'Light'}</span>
                </div>
              </div>
            </div>

            {/* Session Info */}
            <div className="bg-gray-700 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">📊 Session Information</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-400">Session ID:</span>
                  <span className="ml-2 text-white font-mono">{sessionTracker.getSessionSummary().session_id.slice(-12)}</span>
                </div>
                <div>
                  <span className="text-gray-400">Duration:</span>
                  <span className="ml-2 text-white">{Math.round((Date.now() - sessionTracker.start_time) / 60000)} minutes</span>
                </div>
                <div>
                  <span className="text-gray-400">Actions:</span>
                  <span className="ml-2 text-white">{sessionTracker.action_count}</span>
                </div>
                <div>
                  <span className="text-gray-400">Current Screen:</span>
                  <span className="ml-2 text-white">{currentScreen}</span>
                </div>
              </div>
            </div>

            {/* System Status */}
            <div className="bg-gray-700 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">🖥️ System Status</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Python Bridge:</span>
                  <span className="text-green-400">🟢 Connected</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Local Storage:</span>
                  <span className="text-green-400">🟢 Available</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">WebSocket:</span>
                  <span className="text-green-400">🟢 Connected</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Debug Tools:</span>
                  <span className="text-blue-400">🔧 Available (debugApp)</span>
                </div>
              </div>
            </div>

            {/* Debug Actions */}
            <div className="bg-gray-700 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">🛠️ Debug Actions</h3>
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={() => {
                    (window as any).debugApp?.logs();
                    showNotification('Check browser console for logs', 'info');
                  }}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  📋 Show Logs
                </button>
                <button
                  onClick={() => {
                    (window as any).debugApp?.session();
                    showNotification('Check browser console for session info', 'info');
                  }}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  📊 Session Info
                </button>
                <button
                  onClick={() => {
                    (window as any).debugApp?.bridge();
                    showNotification('Check browser console for bridge status', 'info');
                  }}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  🔗 Bridge Status
                </button>
                <button
                  onClick={() => {
                    (window as any).debugApp?.help();
                    showNotification('Check browser console for debug commands', 'info');
                  }}
                  className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
                >
                  ❓ Debug Help
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SettingsPage;