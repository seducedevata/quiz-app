'use client';

import React, { useState, useEffect } from 'react';
import { callPythonMethod } from '../../lib/pythonBridge';
import { AppLogger } from '../../lib/logger';

interface UserSettings {
  api_keys: {
    openai?: string;
    anthropic?: string;
    gemini?: string;
    groq?: string;
    openrouter?: string;
    deepseek?: string;
    tavily?: string;
  };
  default_quiz_mode?: string;
  default_game_mode?: string;
  default_submode?: string;
  default_difficulty?: string;
  theme?: string;
  // Add other settings as needed
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<UserSettings>({
    api_keys: {},
    default_quiz_mode: 'auto',
    default_game_mode: 'casual',
    default_submode: 'mixed',
    default_difficulty: 'medium',
    theme: 'light',
  });
  const [saveStatus, setSaveStatus] = useState(''); // success, error, saving

  const loadSettings = async () => {
    AppLogger.info('SETTINGS', 'Loading user settings');
    try {
      const settingsJson = await callPythonMethod<string>('getUserSettings');
      const loadedSettings: UserSettings = JSON.parse(settingsJson);
      setSettings(prevSettings => ({
        ...prevSettings,
        ...loadedSettings,
        api_keys: { ...prevSettings.api_keys, ...loadedSettings.api_keys },
      }));
      AppLogger.success('SETTINGS', 'Settings loaded from backend', { settingsCount: Object.keys(loadedSettings).length });
      updateApiKeyStatusIndicators(loadedSettings.api_keys);
    } catch (error: any) {
      AppLogger.error('SETTINGS', 'Failed to load settings', { error: error.message });
      setSaveStatus('error');
    }
  };

  const saveSettings = async () => {
    AppLogger.info('SETTINGS', 'Saving user settings');
    setSaveStatus('saving');
    try {
      await callPythonMethod<void>('saveUserSettings', JSON.stringify(settings));
      AppLogger.success('SETTINGS', 'Settings saved successfully');
      setSaveStatus('success');
      setTimeout(() => setSaveStatus(''), 3000);
    } catch (error: any) {
      AppLogger.error('SETTINGS', 'Failed to save settings', { error: error.message });
      setSaveStatus('error');
      setTimeout(() => setSaveStatus(''), 3000);
    }
  };

  const handleApiKeyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setSettings(prevSettings => ({
      ...prevSettings,
      api_keys: {
        ...prevSettings.api_keys,
        [name]: value,
      },
    }));
  };

  const handleSettingChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type, checked } = e.target;
    setSettings(prevSettings => ({
      ...prevSettings,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const updateApiKeyStatusIndicators = (currentApiKeys: UserSettings['api_keys']) => {
    AppLogger.debug('API_KEYS', 'Updating API key status indicators');
    const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter', 'deepseek', 'tavily'];
    providers.forEach(provider => {
      const indicator = document.getElementById(`${provider}-status`);
      if (indicator) {
        const hasKey = (currentApiKeys[provider as keyof typeof currentApiKeys]?.trim() || '').length > 0;
        indicator.textContent = hasKey ? '✅' : '❌';
        indicator.className = hasKey ? 'status-indicator ready' : 'status-indicator error';
      }
    });
  };

  const updateProviderStatuses = async () => {
    AppLogger.debug('PROVIDERS', 'Updating provider statuses');
    const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter', 'deepseek', 'tavily'];
    for (const provider of providers) {
      try {
        const statusJson = await callPythonMethod<string>('checkProviderStatus', provider);
        const status = JSON.parse(statusJson);
        const statusElement = document.getElementById(`${provider}-provider-status`);
        if (statusElement) {
          statusElement.textContent = status.available ? '🟢 Available' : '🔴 Unavailable';
          statusElement.className = status.available ? 'provider-status available' : 'provider-status unavailable';
        }
      } catch (error: any) {
        AppLogger.error('PROVIDERS', `Failed to check ${provider} status`, { error: error.message });
        const statusElement = document.getElementById(`${provider}-provider-status`);
        if (statusElement) {
          statusElement.textContent = '🔴 Error';
          statusElement.className = 'provider-status unavailable';
        }
      }
    }
  };

  useEffect(() => {
    loadSettings();
    updateProviderStatuses();
  }, []);

  return (
    <div className="settings-page p-8">
      <h1 className="text-3xl font-bold text-text-primary mb-6">Application Settings</h1>

      <div className="settings-card bg-bg-secondary p-6 rounded-lg shadow-md mb-6">
        <h2 className="text-2xl font-bold text-text-primary mb-4">API Configuration</h2>
        <p className="text-text-secondary mb-4">Enter your API keys for various services. These are stored securely.</p>
        <div className="api-key-inputs grid grid-cols-1 md:grid-cols-2 gap-4">
          {['openai', 'anthropic', 'gemini', 'groq', 'openrouter', 'deepseek', 'tavily'].map(provider => (
            <div className="form-group" key={provider}>
              <label htmlFor={`${provider}-api-key`} className="block text-text-secondary text-sm font-bold mb-2 capitalize">
                {provider} API Key
                <span id={`${provider}-status`} className="status-indicator ml-2"></span>
              </label>
              <input
                type="password"
                id={`${provider}-api-key`}
                name={provider}
                value={settings.api_keys[provider as keyof typeof settings.api_keys] || ''}
                onChange={handleApiKeyChange}
                onBlur={saveSettings} // Save on blur
                placeholder={`Enter ${provider} API Key`}
                className="w-full p-2 border border-border-color rounded-md bg-bg-primary text-text-primary"
              />
              <div id={`${provider}-provider-status`} className="provider-status text-xs mt-1"></div>
            </div>
          ))}
        </div>
      </div>

      <div className="settings-card bg-bg-secondary p-6 rounded-lg shadow-md mb-6">
        <h2 className="text-2xl font-bold text-text-primary mb-4">Quiz Defaults</h2>
        <div className="form-group mb-4">
          <label htmlFor="default_quiz_mode" className="block text-text-secondary text-sm font-bold mb-2">Default Quiz Mode</label>
          <select
            id="default_quiz_mode"
            name="default_quiz_mode"
            value={settings.default_quiz_mode}
            onChange={handleSettingChange}
            onBlur={saveSettings}
            className="w-full p-2 border border-border-color rounded-md bg-bg-primary text-text-primary"
          >
            <option value="auto">Auto (Best Available)</option>
            <option value="offline">Offline (Local AI - TURBO)</option>
            <option value="online">Online (Cloud APIs)</option>
          </select>
        </div>
        <div className="form-group mb-4">
          <label htmlFor="default_game_mode" className="block text-text-secondary text-sm font-bold mb-2">Default Game Mode</label>
          <select
            id="default_game_mode"
            name="default_game_mode"
            value={settings.default_game_mode}
            onChange={handleSettingChange}
            onBlur={saveSettings}
            className="w-full p-2 border border-border-color rounded-md bg-bg-primary text-text-primary"
          >
            <option value="casual">Casual Mode</option>
            <option value="serious">Serious Mode</option>
          </select>
        </div>
        <div className="form-group mb-4">
          <label htmlFor="default_submode" className="block text-text-secondary text-sm font-bold mb-2">Default Question Type</label>
          <select
            id="default_submode"
            name="default_submode"
            value={settings.default_submode}
            onChange={handleSettingChange}
            onBlur={saveSettings}
            className="w-full p-2 border border-border-color rounded-md bg-bg-primary text-text-primary"
          >
            <option value="mixed">Mixed</option>
            <option value="numerical">Numerical</option>
            <option value="conceptual">Conceptual</option>
          </select>
        </div>
        <div className="form-group mb-4">
          <label htmlFor="default_difficulty" className="block text-text-secondary text-sm font-bold mb-2">Default Difficulty</label>
          <select
            id="default_difficulty"
            name="default_difficulty"
            value={settings.default_difficulty}
            onChange={handleSettingChange}
            onBlur={saveSettings}
            className="w-full p-2 border border-border-color rounded-md bg-bg-primary text-text-primary"
          >
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
            <option value="expert">Expert</option>
          </select>
        </div>
      </div>

      <div className="settings-card bg-bg-secondary p-6 rounded-lg shadow-md mb-6">
        <h2 className="text-2xl font-bold text-text-primary mb-4">Theme Settings</h2>
        <div className="form-group mb-4">
          <label htmlFor="theme" className="block text-text-secondary text-sm font-bold mb-2">Theme</label>
          <select
            id="theme"
            name="theme"
            value={settings.theme}
            onChange={handleSettingChange}
            onBlur={saveSettings}
            className="w-full p-2 border border-border-color rounded-md bg-bg-primary text-text-primary"
          >
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </div>
      </div>

      <button className="btn-primary w-full py-3 px-4 rounded-md" onClick={saveSettings}>
        {saveStatus === 'saving' ? 'Saving...' :
         saveStatus === 'success' ? 'Settings Saved!' :
         saveStatus === 'error' ? 'Save Failed!' :
         'Save Settings'}
      </button>
      {saveStatus === 'error' && <p className="error-message text-error-color mt-4"><span>Error:</span> {error}</p>}
    </div>
  );