// Test script to directly call the AI API and debug the response format
const https = require('https');

const API_URL = 'https://api.minimax.chat/v1/chat/completions';
const API_KEY = process.env.MINIMAX_API_KEY || '';
const QUERY = '使用浏览器测试项目的skills';

if (!API_KEY) {
  console.error('Please set MINIMAX_API_KEY environment variable');
  process.exit(1);
}

const systemPrompt = `你是一个 AI Agent 技能搜索助手。

## 任务
将用户的自然语言描述转换为 3-5 个简洁的英文搜索关键词,并用中文输出你的思考过程。

## 输出格式
你必须在 <think></think> 标签内输出中文思考过程,然后每行输出一个英文关键词。

示例:
<think>
用户的需求是关于量化分析,需要将数据转化为可度量的形式。这涉及数据分析、统计计算和指标评估等能力。相关的英文关键词包括:
</think>
quantification
data analysis
metrics
statistical analysis
calculation`;

const userPrompt = `我需要一个能够实现以下功能的 skill: ${QUERY}

请按照上述格式回复。`;

const requestBody = {
  model: 'MiniMax-M2.7',
  messages: [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt }
  ],
  temperature: 0.7,
  max_tokens: 1024
};

const data = JSON.stringify(requestBody);

const options = {
  hostname: 'api.minimax.chat',
  path: '/v1/chat/completions',
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json',
    'Content-Length': data.length
  }
};

const req = https.request(options, (res) => {
  let body = '';
  
  res.on('data', (chunk) => {
    body += chunk;
  });
  
  res.on('end', () => {
    console.log('=== STATUS ===');
    console.log(res.statusCode);
    
    console.log('\n=== RAW RESPONSE ===');
    console.log(body);
    
    try {
      const response = JSON.parse(body);
      const content = response.choices[0].message.content;
      
      console.log('\n=== EXTRACTED CONTENT ===');
      console.log(content);
      
      console.log('\n=== ANALYSIS ===');
      console.log('Has <think> tag:', content.includes('<think>'));
      console.log('Has </think> tag:', content.includes('</think>'));
      console.log('Content length:', content.length);
      
      // Try to extract thinking and keywords
      const lines = content.split('\n').map(l => l.trim()).filter(l => l);
      console.log('\n=== LINES ===');
      lines.forEach((line, i) => {
        console.log(`${i}: ${line.substring(0, 100)}`);
      });
      
    } catch (e) {
      console.error('Failed to parse JSON:', e.message);
    }
  });
});

req.on('error', (error) => {
  console.error('Request error:', error);
});

req.write(data);
req.end();
