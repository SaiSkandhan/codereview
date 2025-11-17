// CodeSense AI - Frontend JavaScript

const API_BASE_URL = '/api';

// DOM Elements
const codeInput = document.getElementById('code-input');
const languageSelect = document.getElementById('language-select');
const analyzeBtn = document.getElementById('analyze-btn');
const lineNumbers = document.getElementById('line-numbers');
const outputContent = document.getElementById('output-content');
const errorsContent = document.getElementById('errors-content');
const suggestionsContent = document.getElementById('suggestions-content');
const optimizationsContent = document.getElementById('optimizations-content');
const errorCount = document.getElementById('error-count');

// Update line numbers
function updateLineNumbers() {
    const lines = codeInput.value.split('\n');
    lineNumbers.innerHTML = lines.map((_, index) => 
        `<div>${index + 1}</div>`
    ).join('');
}

// Update analyze button state
function updateAnalyzeButton() {
    const hasCode = codeInput.value.trim().length > 0;
    const hasLanguage = languageSelect.value !== '';
    analyzeBtn.disabled = !hasCode || !hasLanguage;
}

// Show toast notification
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast show ${type}`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Analyze code
async function analyzeCode() {
    const code = codeInput.value.trim();
    const language = languageSelect.value;
    
    if (!code || !language) {
        showToast('Please enter code and select a language', 'error');
        return;
    }
    
    // Disable button and show loading state
    analyzeBtn.disabled = true;
    const originalContent = analyzeBtn.innerHTML;
    analyzeBtn.innerHTML = '<div class="spinner"></div> <span>Analyzing...</span>';
    
    try {
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                code: code,
                language: language
            })
        });
        
        // Read response once and reuse
        const responseText = await response.text();
        
        if (!response.ok) {
            let errorMessage = 'Failed to analyze code';
            try {
                const error = JSON.parse(responseText);
                errorMessage = error.detail || error.message || errorMessage;
            } catch (e) {
                // If response is not JSON (e.g., HTML error page), use text
                errorMessage = responseText.substring(0, 200) || errorMessage;
            }
            throw new Error(errorMessage);
        }
        
        // Parse the response as JSON
        let data;
        try {
            data = JSON.parse(responseText);
        } catch (e) {
            // If response is not valid JSON, show error
            throw new Error(`Invalid response from server: ${responseText.substring(0, 100)}`);
        }
        
        // Display results
        displayResults(data);
        
        showToast('Analysis Complete!', 'success');
        
    } catch (error) {
        console.error('Error analyzing code:', error);
        showToast(error.message || 'Failed to analyze code. Please try again.', 'error');
        
        // Show error state
        displayResults({
            errors: [{ line: 1, message: error.message || 'Failed to analyze code', severity: 'error' }],
            suggestions: ['Please try again or check your internet connection'],
            optimizations: ['Analysis unavailable'],
            output: 'Analysis failed'
        });
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = originalContent;
    }
}

// Display analysis results
function displayResults(data) {
    // Display output
    if (data.output) {
        outputContent.innerHTML = `<pre>${escapeHtml(data.output)}</pre>`;
    } else {
        outputContent.innerHTML = '<p class="placeholder">No output detected.</p>';
    }
    
    // Display errors
    if (data.errors && data.errors.length > 0) {
        errorCount.textContent = data.errors.length;
        errorCount.style.display = 'inline-block';
        
        errorsContent.innerHTML = data.errors.map(error => `
            <div class="error-item">
                <span class="badge badge-danger">Line ${error.line}</span>
                <div style="flex: 1;">
                    <p style="font-weight: 500; font-size: 0.875rem;">${escapeHtml(error.message)}</p>
                    <p style="font-size: 0.75rem; color: var(--muted-foreground); text-transform: capitalize;">${error.severity}</p>
                </div>
            </div>
        `).join('');
    } else {
        errorCount.style.display = 'none';
        errorsContent.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.5rem; color: var(--success);">
                <i class="fas fa-check-circle"></i>
                <p style="font-size: 0.875rem;">No issues found! Your code looks good.</p>
            </div>
        `;
    }
    
    // Display suggestions
    if (data.suggestions && data.suggestions.length > 0) {
        suggestionsContent.innerHTML = data.suggestions.map(suggestion => `
            <div class="suggestion-item">
                <p>${escapeHtml(suggestion)}</p>
            </div>
        `).join('');
    } else {
        suggestionsContent.innerHTML = '<p class="placeholder">No suggestions available.</p>';
    }
    
    // Display optimizations
    if (data.optimizations && data.optimizations.length > 0) {
        optimizationsContent.innerHTML = data.optimizations.map(optimization => `
            <div class="optimization-item">
                <p>${escapeHtml(optimization)}</p>
            </div>
        `).join('');
    } else {
        optimizationsContent.innerHTML = '<p class="placeholder">No optimizations suggested.</p>';
    }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Event Listeners
codeInput.addEventListener('input', () => {
    updateLineNumbers();
    updateAnalyzeButton();
});

languageSelect.addEventListener('change', updateAnalyzeButton);

analyzeBtn.addEventListener('click', analyzeCode);

// Initialize
updateLineNumbers();
updateAnalyzeButton();

