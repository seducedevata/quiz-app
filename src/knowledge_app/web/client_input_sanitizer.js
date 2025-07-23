/**
 * 🛡️ SECURITY FIX #17: Client-Side Input Sanitization
 * 
 * Provides comprehensive input validation and sanitization on the client-side
 * to prevent XSS, injection attacks, and other security vulnerabilities.
 * 
 * This works in conjunction with server-side sanitization for defense in depth.
 */

class ClientInputSanitizer {
    constructor() {
        this.initialized = false;
        this.securityLevel = 'balanced'; // strict, balanced, permissive
        this.initPatterns();
        
        console.log('🛡️ ClientInputSanitizer initialized');
    }
    
    initPatterns() {
        // XSS Prevention Patterns
        this.xssPatterns = [
            /<script[^>]*>.*?<\/script>/gi,
            /javascript:/gi,
            /vbscript:/gi,
            /onload\s*=/gi,
            /onerror\s*=/gi,
            /onclick\s*=/gi,
            /onmouseover\s*=/gi,
            /<iframe[^>]*>/gi,
            /<object[^>]*>/gi,
            /<embed[^>]*>/gi,
            /<link[^>]*>/gi,
            /<meta[^>]*>/gi,
            /<style[^>]*>.*?<\/style>/gi,
        ];
        
        // SQL Injection Patterns
        this.sqlPatterns = [
            /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)/gi,
            /(\b(UNION|OR|AND)\s+\d+\s*=\s*\d+)/gi,
            /(--|#|\/\*|\*\/)/g,
            /(\bxp_cmdshell\b)/gi,
        ];
        
        // Command Injection Patterns
        this.commandPatterns = [
            /[;&|`$(){}[\]\\]/g,
            /\b(rm|del|format|fdisk|kill|shutdown|reboot)\b/gi,
            /\b(wget|curl|nc|netcat)\b/gi,
        ];
        
        // Prompt Injection Patterns
        this.promptPatterns = [
            /ignore\s+(?:all\s+)?(?:previous\s+)?instructions?/gi,
            /forget\s+(?:all\s+)?(?:previous\s+)?instructions?/gi,
            /new\s+task/gi,
            /your\s+(?:new\s+)?task\s+is/gi,
            /do\s+not\s+(?:output|generate|create)\s+json/gi,
            /output\s+(?:a\s+)?(?:poem|story|essay|text)/gi,
            /write\s+(?:a\s+)?(?:poem|story|essay)/gi,
            /pretend\s+(?:to\s+be|you\s+are)/gi,
            /act\s+as\s+(?:a\s+)?(?:different|another)/gi,
            /role\s*play/gi,
        ];
        
        // Dangerous file extensions
        this.dangerousExtensions = [
            '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js', '.jar',
            '.app', '.deb', '.pkg', '.dmg', '.iso', '.msi', '.run', '.sh', '.ps1'
        ];
        
        // Safe file extensions
        this.safeExtensions = [
            '.txt', '.pdf', '.docx', '.doc', '.rtf', '.odt', '.md', '.csv',
            '.json', '.xml', '.yaml', '.yml', '.png', '.jpg', '.jpeg', '.gif'
        ];
    }
    
    /**
     * 🛡️ Main sanitization method
     * @param {any} input - Input to sanitize
     * @param {string} type - Input type (topic, filename, general, etc.)
     * @param {number} maxLength - Maximum allowed length
     * @returns {string} Sanitized input
     */
    sanitizeInput(input, type = 'general', maxLength = null) {
        try {
            // Convert to string and handle null/undefined
            if (input === null || input === undefined) {
                return '';
            }
            
            let sanitized = String(input);
            
            // Remove null bytes and control characters
            sanitized = this.removeControlCharacters(sanitized);
            
            // Apply context-specific sanitization
            sanitized = this.applyContextSanitization(sanitized, type);
            
            // Apply security filtering
            sanitized = this.applySecurityFiltering(sanitized);
            
            // Apply length limits
            sanitized = this.applyLengthLimits(sanitized, type, maxLength);
            
            return sanitized.trim();
            
        } catch (error) {
            console.error('❌ Client-side sanitization failed:', error);
            return ''; // Fail safe
        }
    }
    
    removeControlCharacters(text) {
        // Remove null bytes and dangerous control characters
        text = text.replace(/\x00/g, '');
        
        // Remove other dangerous control characters but preserve common ones
        text = text.replace(/[\x01-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]/g, '');
        
        // Normalize line endings
        text = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
        
        return text;
    }
    
    applyContextSanitization(text, type) {
        switch (type) {
            case 'filename':
                return this.sanitizeFilename(text);
            case 'topic':
                return this.sanitizeTopic(text);
            case 'api_key':
                return this.sanitizeApiKey(text);
            case 'url':
                return this.sanitizeUrl(text);
            case 'email':
                return this.sanitizeEmail(text);
            case 'html':
                return this.sanitizeHtml(text);
            case 'json':
                return this.sanitizeJson(text);
            case 'quiz_answer':
                return this.sanitizeQuizAnswer(text);
            default:
                return this.sanitizeGeneral(text);
        }
    }
    
    sanitizeFilename(filename) {
        // Remove path components
        filename = filename.split(/[/\\]/).pop();
        
        // Remove dangerous characters
        filename = filename.replace(/[<>:"/\\|?*\x00-\x1f]/g, '');
        
        // Check extension
        const ext = this.getFileExtension(filename);
        if (this.dangerousExtensions.includes(ext)) {
            console.warn('⚠️ Dangerous file extension blocked:', ext);
            return '';
        }
        
        // Limit length
        if (filename.length > 255) {
            const nameWithoutExt = filename.substring(0, filename.lastIndexOf('.'));
            const extension = filename.substring(filename.lastIndexOf('.'));
            filename = nameWithoutExt.substring(0, 200) + extension;
        }
        
        return filename;
    }
    
    sanitizeTopic(topic) {
        // Remove prompt injection patterns
        this.promptPatterns.forEach(pattern => {
            topic = topic.replace(pattern, '[FILTERED]');
        });
        
        // Remove excessive special characters
        topic = topic.replace(/[{}]+/g, '');
        topic = topic.replace(/#{3,}/g, '##');
        topic = topic.replace(/`{3,}/g, '``');
        
        return topic;
    }
    
    sanitizeApiKey(apiKey) {
        // Remove whitespace and dangerous characters
        apiKey = apiKey.replace(/[<>"\'\\\x00-\x1f\x7f-\x9f]/g, '').trim();
        
        // Basic format validation (alphanumeric, hyphens, underscores only)
        if (!/^[a-zA-Z0-9_-]+$/.test(apiKey)) {
            console.warn('⚠️ Invalid API key format');
            return '';
        }
        
        return apiKey;
    }
    
    sanitizeUrl(url) {
        try {
            const urlObj = new URL(url);
            
            // Only allow safe schemes
            const safeSchemes = ['http:', 'https:', 'ftp:', 'ftps:'];
            if (!safeSchemes.includes(urlObj.protocol)) {
                console.warn('⚠️ Unsafe URL scheme:', urlObj.protocol);
                return '';
            }
            
            return urlObj.toString();
        } catch (error) {
            console.warn('⚠️ Invalid URL format');
            return '';
        }
    }
    
    sanitizeEmail(email) {
        // Basic email validation
        const emailPattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
        if (!emailPattern.test(email)) {
            return '';
        }
        return email.toLowerCase().trim();
    }
    
    sanitizeHtml(html) {
        // Remove dangerous HTML patterns
        this.xssPatterns.forEach(pattern => {
            html = html.replace(pattern, '');
        });
        
        // HTML escape remaining content
        return this.escapeHtml(html);
    }
    
    sanitizeJson(jsonStr) {
        try {
            // Parse and re-stringify to ensure valid JSON
            const parsed = JSON.parse(jsonStr);
            return JSON.stringify(parsed);
        } catch (error) {
            console.warn('⚠️ Invalid JSON format');
            return '{}';
        }
    }
    
    sanitizeQuizAnswer(answer) {
        // Basic text sanitization
        answer = this.escapeHtml(answer);
        answer = answer.replace(/[<>"\'\\\x00-\x1f]/g, '');
        return answer;
    }
    
    sanitizeGeneral(text) {
        // HTML escape
        text = this.escapeHtml(text);
        
        // Remove dangerous characters based on security level
        if (this.securityLevel === 'strict') {
            // Remove all special characters except basic punctuation
            text = text.replace(/[^a-zA-Z0-9\s.,!?-]/g, '');
        } else if (this.securityLevel === 'balanced') {
            // Remove dangerous characters but allow common symbols
            text = text.replace(/[<>"\'\\\x00-\x1f\x7f-\x9f]/g, '');
        }
        
        return text;
    }
    
    applySecurityFiltering(text) {
        // Check for XSS patterns
        this.xssPatterns.forEach(pattern => {
            if (pattern.test(text)) {
                console.warn('⚠️ XSS attempt blocked:', pattern);
                text = text.replace(pattern, '[FILTERED]');
            }
        });
        
        // Check for SQL injection patterns
        this.sqlPatterns.forEach(pattern => {
            if (pattern.test(text)) {
                console.warn('⚠️ SQL injection attempt blocked:', pattern);
                text = text.replace(pattern, '[FILTERED]');
            }
        });
        
        // Check for command injection patterns
        this.commandPatterns.forEach(pattern => {
            if (pattern.test(text)) {
                console.warn('⚠️ Command injection attempt blocked:', pattern);
                text = text.replace(pattern, '[FILTERED]');
            }
        });
        
        return text;
    }
    
    applyLengthLimits(text, type, maxLength) {
        // Default length limits by input type
        const defaultLimits = {
            'topic': 200,
            'filename': 255,
            'api_key': 200,
            'url': 2048,
            'email': 254,
            'quiz_answer': 500,
            'general': 10000,
            'html': 50000,
            'json': 100000,
        };
        
        const limit = maxLength || defaultLimits[type] || 1000;
        
        if (text.length > limit) {
            text = text.substring(0, limit);
            console.warn(`⚠️ Input truncated to ${limit} characters`);
        }
        
        return text;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    getFileExtension(filename) {
        return filename.substring(filename.lastIndexOf('.')).toLowerCase();
    }
    
    /**
     * 🛡️ Validate file upload
     * @param {string} filename - File name
     * @param {number} fileSize - File size in bytes
     * @param {number} maxSize - Maximum allowed size
     * @returns {object} Validation result
     */
    validateFileUpload(filename, fileSize, maxSize = 10 * 1024 * 1024) {
        try {
            // Sanitize filename
            const cleanFilename = this.sanitizeInput(filename, 'filename');
            if (!cleanFilename) {
                return { isValid: false, error: 'Invalid filename' };
            }
            
            // Check file size
            if (fileSize > maxSize) {
                return { 
                    isValid: false, 
                    error: `File too large (max ${Math.round(maxSize / 1024 / 1024)}MB)` 
                };
            }
            
            // Check extension
            const ext = this.getFileExtension(cleanFilename);
            if (!this.safeExtensions.includes(ext)) {
                return { 
                    isValid: false, 
                    error: `File type not allowed: ${ext}` 
                };
            }
            
            return { isValid: true, error: '' };
            
        } catch (error) {
            console.error('❌ File validation error:', error);
            return { isValid: false, error: 'File validation failed' };
        }
    }
}

// Global instance
let clientSanitizer = null;

function getClientSanitizer() {
    if (!clientSanitizer) {
        clientSanitizer = new ClientInputSanitizer();
    }
    return clientSanitizer;
}

// 🛡️ SECURITY FIX #17: Global sanitization functions
window.sanitizeInput = function(input, type = 'general', maxLength = null) {
    const sanitizer = getClientSanitizer();
    return sanitizer.sanitizeInput(input, type, maxLength);
};

window.validateFileUpload = function(filename, fileSize, maxSize = null) {
    const sanitizer = getClientSanitizer();
    return sanitizer.validateFileUpload(filename, fileSize, maxSize);
};

console.log('🛡️ Client-side input sanitization loaded');
