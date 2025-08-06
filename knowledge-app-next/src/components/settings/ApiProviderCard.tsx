
'use client';

import React, { useState } from 'react';
import { Card } from '@/components/common/Card';
import { Input } from '@/components/common/Input';
import { Button } from '@/components/common/Button';
import { Icon } from '@/components/common/Icon';

interface ApiProviderCardProps {
  provider: string;
}

export const ApiProviderCard: React.FC<ApiProviderCardProps> = ({ provider }) => {
  const [apiKey, setApiKey] = useState('');
  const [connectionStatus, setConnectionStatus] = useState('disconnected'); // connected, error, testing

  const testConnection = () => {
    setConnectionStatus('testing');
    // Simulate API call
    setTimeout(() => {
      if (apiKey.length > 0) {
        setConnectionStatus('connected');
      } else {
        setConnectionStatus('error');
      }
    }, 1500);
  };

  const getStatusColor = () => {
    if (connectionStatus === 'connected') return 'border-successColor';
    if (connectionStatus === 'error') return 'border-dangerColor';
    if (connectionStatus === 'testing') return 'border-warningColor';
    return 'border-borderColor';
  };

  const getStatusBgColor = () => {
    if (connectionStatus === 'connected') return 'bg-successColor';
    if (connectionStatus === 'error') return 'bg-dangerColor';
    if (connectionStatus === 'testing') return 'bg-warningColor';
    return 'bg-bgSecondary';
  };

  const getStatusTextColor = () => {
    if (connectionStatus === 'testing') return 'text-black';
    return 'text-white';
  };

  const getProviderIcon = () => {
    switch (provider) {
      case 'OpenAI':
        return 'FaBrain';
      case 'Anthropic':
        return 'FaGem';
      case 'Google':
        return 'FaGoogle';
      case 'DeepSeek':
        return 'FaSearch';
      default:
        return 'FaQuestion';
    }
  };

  return (
    <Card className={`border-l-4 ${getStatusColor()}`}>
      <div className="flex justify-between items-center mb-lg">
        <div className="flex items-center">
          <Icon name={getProviderIcon()} className="mr-sm" />
          <h3 className="text-h3 font-h3 text-textPrimary m-0">{provider}</h3>
        </div>
        <div className={`px-sm py-xs rounded-sm text-caption font-medium uppercase ${getStatusBgColor()} ${getStatusTextColor()}`}>
          {connectionStatus}
        </div>
      </div>
      <Input
        placeholder={`Enter ${provider} API Key`}
        value={apiKey}
        onChange={(e) => setApiKey(e.target.value)}
      />
      <Button onClick={testConnection} variant="secondary" className="mt-md">
        Test Connection
      </Button>
    </Card>
  );
};
