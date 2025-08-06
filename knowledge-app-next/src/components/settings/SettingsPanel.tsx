'use client';

import React from 'react';
import { ApiProviderCard } from './ApiProviderCard';
import { SettingsGroup } from './SettingsGroup';
import { SettingItem } from './SettingItem';
import { ThemeSelector } from './ThemeSelector';
import { PreferenceToggle } from './PreferenceToggle';
import { Input } from '@/components/common/Input';
import { Button } from '@/components/common/Button';

export const SettingsPanel: React.FC = () => {
  return (
    <div className="max-w-5xl mx-auto">
      <h1 className="text-h1 font-h1 text-textPrimary mb-xl">Settings</h1>

      <SettingsGroup title="Appearance">
        <SettingItem label="Dark Mode">
          <ThemeSelector />
        </SettingItem>
      </SettingsGroup>

      <SettingsGroup title="Quiz Settings">
        <SettingItem label="Default Timer (seconds)">
          <Input placeholder="30" value="30" onChange={() => {}} />
        </SettingItem>
        <SettingItem label="Show Correct Answers">
          <PreferenceToggle label="" checked={true} onChange={() => {}} />
        </SettingItem>
        <SettingItem label="Save Generated Questions" helpText="Automatically save all generated questions to review history for later practice">
          <PreferenceToggle label="" checked={true} onChange={() => {}} />
        </SettingItem>
      </SettingsGroup>

      <SettingsGroup title="AI Providers" description="Configure API keys for cloud AI providers. Keys are stored locally and encrypted.">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-lg">
          <ApiProviderCard provider="OpenAI" />
          <ApiProviderCard provider="Anthropic" />
          <ApiProviderCard provider="Google" />
          <ApiProviderCard provider="DeepSeek" />
        </div>
        <div className="flex justify-start space-x-md mt-lg">
          <Button onClick={() => {}} variant="secondary">
            Test All APIs
          </Button>
          <Button onClick={() => {}} variant="secondary">
            Clear All Keys
          </Button>
          <Button onClick={() => {}} variant="secondary">
            Advanced Reset
          </Button>
          <Button onClick={() => {}} variant="secondary">
            Validate Keys
          </Button>
        </div>
      </SettingsGroup>

      <SettingsGroup title="Model Selection & Preferences">
        <SettingItem label="Available Models">
          <Button onClick={() => {}} variant="secondary">
            Refresh Available Models
          </Button>
        </SettingItem>
        <SettingItem label="Preferred Model for MCQ Generation" helpText="Auto-select best available model">
          <Input placeholder="Auto-select best available model" value="" onChange={() => {}} />
        </SettingItem>
        <SettingItem label="Preferred Model for Expert/Thinking Mode" helpText="Auto-select best thinking model">
          <Input placeholder="Auto-select best thinking model" value="" onChange={() => {}} />
        </SettingItem>
        <SettingItem label="Model Selection Strategy" helpText="How the system should choose models when multiple options are available">
          <Input placeholder="Intelligent (Recommended)" value="" onChange={() => {}} />
        </SettingItem>
      </SettingsGroup>

      <Button onClick={() => {}}>
        Save Settings
      </Button>
    </div>
  );
};