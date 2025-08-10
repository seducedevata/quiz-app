#!/usr/bin/env node

const http = require('http');

console.log('🔍 Testing Express API Server Connection...\n');

// Test the health endpoint
const options = {
  hostname: 'localhost',
  port: 8000,
  path: '/health',
  method: 'GET'
};

const req = http.request(options, (res) => {
  let data = '';
  
  res.on('data', (chunk) => {
    data += chunk;
  });
  
  res.on('end', () => {
    if (res.statusCode === 200) {
      console.log('✅ Express API Server is running successfully!');
      console.log('📊 Health check response:', JSON.parse(data));
      console.log('\n🎉 You can now use npm run dev safely');
    } else {
      console.log('❌ Express API Server responded with status:', res.statusCode);
      console.log('Response:', data);
    }
  });
});

req.on('error', (err) => {
  console.log('❌ Express API Server is not running');
  console.log('💡 Error:', err.message);
  console.log('\n🔧 To fix this:');
  console.log('   1. Make sure api-server/server.js exists');
  console.log('   2. Run: npm run api:start');
  console.log('   3. Or just run: npm run dev (it should start automatically)');
});

req.end();
