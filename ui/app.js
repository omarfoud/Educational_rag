const API_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000';

// DOM Elements
const tabs = document.querySelectorAll('.tab-btn');
const contents = document.querySelectorAll('.tab-content');
const statusIndicator = document.getElementById('status-indicator');
const statusText = statusIndicator.querySelector('.text');
const logBox = document.getElementById('progress-log'); // Global log box

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupTabs();
    setupUpload();
    setupQA();
    setupDescription();
    setupSummary();
    setupTranscript();
    setupMagicFill();
    setupFlashcards();
    setupQuiz();
    setupAskAI();
    setupAssistant();
    setupAnalytics();

    // Periodic health check
    setInterval(checkHealth, 30000);
});

// Global Utilities
function log(msg, type = 'info') {
    if (!logBox) return;
    const div = document.createElement('div');
    div.textContent = `> ${msg}`;
    div.className = type;
    div.style.color = type === 'error' ? '#ef4444' : (type === 'success' ? '#22c55e' : '#4ade80');
    logBox.appendChild(div);
    logBox.scrollTop = logBox.scrollHeight;
}

function normalizeLanguageLabel(value) {
    return String(value || '')
        .trim()
        .toLowerCase()
        .replace(/[\u064b-\u065f\u0670]/g, '')
        .replace(/[\u0623\u0625\u0622]/g, '\u0627')
        .replace(/\u0649/g, '\u064a')
        .replace(/\u0629/g, '\u0647')
        .replace(/\s+/g, ' ');
}

function inferOutputLanguage(...values) {
    const normalized = values.map(normalizeLanguageLabel).filter(Boolean);
    const joined = normalized.join(' ');

    const englishMarkers = [
        'english',
        'english language',
        'language english',
        '\u0644\u063a\u0647 \u0627\u0646\u062c\u0644\u064a\u0632\u064a',
        '\u0627\u0644\u0644\u063a\u0647 \u0627\u0644\u0627\u0646\u062c\u0644\u064a\u0632\u064a\u0647',
        '\u0644\u063a\u0647 \u0627\u0646\u062c\u0644\u064a\u0632\u064a\u0647',
        '\u0627\u0646\u062c\u0644\u064a\u0632\u064a',
        '\u0627\u0646\u062c\u0644\u064a\u0632\u064a\u0647',
        '\u0627\u0646\u062c\u0644\u0634'
    ];

    const arabicMarkers = [
        'arabic',
        'arabic language',
        'language arabic',
        '\u0644\u063a\u0647 \u0639\u0631\u0628\u064a',
        '\u0627\u0644\u0644\u063a\u0647 \u0627\u0644\u0639\u0631\u0628\u064a\u0647',
        '\u0644\u063a\u0647 \u0639\u0631\u0628\u064a\u0647',
        '\u0639\u0631\u0628\u064a',
        '\u0639\u0631\u0628\u064a\u0647'
    ];

    if (englishMarkers.some(marker => joined.includes(marker))) return 'en';
    if (arabicMarkers.some(marker => joined.includes(marker))) return 'ar';
    if (/[\u0600-\u06ff]/.test(joined)) return 'ar';
    return 'en';
}

function renderPayloadDebug(container, endpoint, payload) {
    const debug = document.createElement('pre');
    debug.className = 'json-block';
    debug.style.marginBottom = '12px';
    debug.textContent = `${endpoint} payload:\n${JSON.stringify(payload, null, 2)}`;
    container.appendChild(debug);
}

async function suggestMetadata(fileId) {
    if (!fileId) return;

    log('Generating metadata suggestions...', 'info');
    try {
        const response = await fetch(`${API_URL}/api/suggest-metadata/${fileId}`);
        if (!response.ok) throw new Error('Failed to fetch suggestions');

        const data = await response.json();

        // Fill Q&A Fields (Now dropdowns)
        const subjectSelect = document.getElementById('qa-subject');
        const gradeSelect = document.getElementById('qa-grade');

        if (data.subject) selectOptionByValue(subjectSelect, data.subject);
        if (data.grade) selectOptionByValue(gradeSelect, data.grade);

        document.getElementById('qa-prompt').value = data.prompt || '';

        // Fill Description Title
        document.getElementById('desc-title').value = data.title || '';

        log('Metadata suggested ✨', 'success');

        // Also fetch multiple topic suggestions
        suggestTopics(fileId);

    } catch (error) {
        console.error('Suggestion failed:', error);
        log('Metadata suggestion failed.', 'info');
    }
}

function selectOptionByValue(selectElement, value) {
    if (!value) return;
    // Try exact match
    for (let option of selectElement.options) {
        if (option.value.toLowerCase() === value.toLowerCase()) {
            selectElement.value = option.value;
            return;
        }
    }
    // Try partial match if no exact match
    for (let option of selectElement.options) {
        if (value.toLowerCase().includes(option.value.toLowerCase())) {
            selectElement.value = option.value;
            return;
        }
    }
}

async function suggestTopics(fileId) {
    const topicSelect = document.getElementById('qa-topic-suggestions');
    if (!fileId || !topicSelect) return;

    try {
        const response = await fetch(`${API_URL}/api/suggest-topics/${fileId}`);
        if (!response.ok) throw new Error('Failed to fetch topics');

        const { topics } = await response.json();

        // Populate dropdown
        topicSelect.innerHTML = '<option value="">Suggested Topics...</option>';
        topics.forEach(topic => {
            const option = document.createElement('option');
            option.value = topic;
            option.textContent = topic;
            topicSelect.appendChild(option);
        });

        topicSelect.classList.remove('hidden');

    } catch (error) {
        console.error('Topic suggestions failed:', error);
    }
}

function setupMagicFill() {
    const qaBtn = document.getElementById('magic-fill-qa');
    const descBtn = document.getElementById('magic-fill-desc');

    qaBtn.addEventListener('click', () => {
        const fileId = document.getElementById('summary-file-id').value; // Borrow fileId from other fields
        if (!fileId) {
            alert('Please process a file first to get suggestions.');
            return;
        }
        suggestMetadata(fileId);
    });

    descBtn.addEventListener('click', () => {
        const fileId = document.getElementById('summary-file-id').value;
        if (!fileId) {
            alert('Please process a file first to get suggestions.');
            return;
        }
        suggestMetadata(fileId);
    });

    // Topic Selection Handler
    const topicSelect = document.getElementById('qa-topic-suggestions');
    topicSelect.addEventListener('change', () => {
        if (topicSelect.value) {
            document.getElementById('qa-prompt').value = topicSelect.value;
        }
    });
}

// Health Check
async function checkHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            statusIndicator.classList.remove('offline');
            statusIndicator.classList.add('online');
            statusText.textContent = 'Connected';
        } else {
            throw new Error('Server returned error');
        }
    } catch (error) {
        statusIndicator.classList.remove('online');
        statusIndicator.classList.add('offline');
        statusText.textContent = 'Disconnected';
        console.error('Health check failed:', error);
    }
}

// Tabs Logic
function setupTabs() {
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            document.getElementById(tab.dataset.tab).classList.add('active');
        });
    });
}

// Ingestion Logic
function setupUpload() {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const dropArea = document.getElementById('drop-area');
    const progressContainer = document.getElementById('upload-progress');
    const progressBar = document.querySelector('.progress-bar-fill');
    const progressStage = document.getElementById('progress-stage');
    const progressPercent = document.getElementById('progress-percent');

    // Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
    });

    dropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            fileInput.files = files;
            updateFileMsg(files[0].name);
            autoDetectMetadata(files[0].name);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            const fileName = fileInput.files[0].name;
            updateFileMsg(fileName);
            autoDetectMetadata(fileName);
        }
    });

    function autoDetectMetadata(filename) {
        log(`Auto-detecting metadata from: ${filename}`, 'info');
        const name = filename.toLowerCase();

        // 1. Detect Subject
        const subjects = {
            'math': 'Math',
            'science': 'Science',
            'physic': 'Physics',
            'chemist': 'Chemistry',
            'biolog': 'Biology',
            'histor': 'History',
            'geogr': 'Geography',
            'arabic': 'Arabic',
            'english': 'English',
            'social': 'Social Studies',
            'computer': 'Computer Science',
            'islamic': 'Islamic Studies'
        };

        for (const [key, val] of Object.entries(subjects)) {
            if (name.includes(key)) {
                document.getElementById('qa-subject').value = val;
                log(`Detected Subject: ${val}`, 'success');
                break;
            }
        }

        // 2. Detect Grade
        const grades = {
            'pri': 'Primary',
            'prp': 'Preparatory',
            'prep': 'Preparatory',
            'sec': 'Secondary',
            'uni': 'University'
        };

        for (const [key, val] of Object.entries(grades)) {
            if (name.includes(key)) {
                const gradeSelect = document.getElementById('qa-grade');
                gradeSelect.value = val;
                log(`Detected Grade: ${val}`, 'success');
                break;
            }
        }

        // 3. Detect Semester
        const semesters = {
            'tr1': 'Term 1',
            'tr2': 'Term 2',
            'term1': 'Term 1',
            'term2': 'Term 2',
            'fall': 'Fall 2023',
            'spring': 'Spring 2024'
        };

        for (const [key, val] of Object.entries(semesters)) {
            if (name.includes(key)) {
                document.getElementById('semester').value = val;
                document.getElementById('qa-semester').value = val;
                log(`Detected Semester: ${val}`, 'success');
                break;
            }
        }
    }

    function updateFileMsg(name) {
        dropArea.querySelector('.file-msg').textContent = name;
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const file = fileInput.files[0];
        if (!file) {
            alert('Please select a file');
            return;
        }

        const type = document.getElementById('file-type').value;
        const fileId = document.getElementById('file-id').value || `file-${Date.now()}`;
        const jobId = `job-${Date.now()}`;
        const translateToEnglish = document.getElementById('translate-to-english').checked;
        const semester = document.getElementById('semester').value;
        const isCourseBook = document.getElementById('is-course-book').checked;

        // Reset UI
        progressContainer.classList.remove('hidden');
        progressBar.style.width = '0%';
        progressPercent.textContent = '0%';
        progressStage.textContent = 'Starting upload...';
        logBox.innerHTML = '';

        // Start WebSocket
        connectWebSocket(jobId, fileId);

        // Upload
        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', type);
        formData.append('fileId', fileId);
        formData.append('jobId', jobId);
        formData.append('translateToEnglish', translateToEnglish);
        formData.append('semester', semester);
        formData.append('isCourseBook', isCourseBook);

        try {
            log('Uploading file...', 'info');
            const response = await fetch(`${API_URL}/api/embed-and-transcribe`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(await response.text());

            const result = await response.json();
            log(`Upload complete!`, 'success');
            log(`Job ID: ${result.jobId}`, 'info');
            log(`File ID: ${result.fileId}`, 'success');

            // Auto-fill File ID in other tabs for convenience
            document.getElementById('qa-file-id').value = result.fileId;
            document.getElementById('quiz-file-id').value = result.fileId;
            document.getElementById('summary-file-id').value = result.fileId;
            document.getElementById('transcript-file-id').value = result.fileId;
            if (document.getElementById('ast-file-id')) document.getElementById('ast-file-id').value = result.fileId;

        } catch (error) {
            log(`Error: ${error.message}`, 'error');
            progressStage.textContent = 'Failed';
        }
    });

    function connectWebSocket(jobId, fileId) {
        const ws = new WebSocket(`${WS_URL}/ws/progress/${jobId}`);

        ws.onopen = () => {
            log('WebSocket connected', 'info');
            // Keep alive
            setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) ws.send('ping');
            }, 10000);
        };

        ws.onmessage = (event) => {
            if (event.data === 'pong') return;

            const data = JSON.parse(event.data);
            const percent = Math.round(data.progress);

            progressBar.style.width = `${percent}%`;
            progressPercent.textContent = `${percent}%`;
            progressStage.textContent = data.stage;

            if (data.message) {
                log(`${data.stage}: ${data.message}`, 'info');
            }

            if (percent >= 100) {
                ws.close();
                log('Processing complete!', 'success');
                // Auto-fill metadata after a short delay
                setTimeout(() => suggestMetadata(fileId), 1000);
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket Error:', error);
        };
    }

    // [suggestMetadata moved to global scope]

    // [log moved to global scope]
}

// Q&A Logic
function setupQA() {
    const form = document.getElementById('qa-form');
    const resultsContainer = document.getElementById('qa-results');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const btn = form.querySelector('button');
        btn.disabled = true;
        btn.textContent = 'Generating...';
        resultsContainer.classList.add('hidden');
        resultsContainer.innerHTML = '';

        const payload = {
            metadata: {
                subject: document.getElementById('qa-subject').value,
                grade: document.getElementById('qa-grade').value,
                semester: document.getElementById('qa-semester').value,
                is_course_book: document.getElementById('qa-is-course-book').checked,
                file_id: document.getElementById('qa-file-id').value.trim()
            },
            prompt: document.getElementById('qa-prompt').value,
            questionsNumber: parseInt(document.getElementById('qa-count').value),
            difficulty: document.getElementById('qa-difficulty').value,
            type: document.getElementById('qa-type').value
        };
        payload.language = inferOutputLanguage(
            payload.metadata.subject,
            payload.metadata.grade,
            payload.prompt
        );
        console.log('/api/ai-generate-questions payload', payload);

        try {
            const response = await fetch(`${API_URL}/api/ai-generate-questions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error(await response.text());

            const questions = await response.json();
            renderQuestions(questions, payload);

        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Generate Questions';
        }
    });

    function renderQuestions(questions, payload) {
        resultsContainer.classList.remove('hidden');
        resultsContainer.innerHTML = '';
        renderPayloadDebug(resultsContainer, '/api/ai-generate-questions', payload);

        if (!Array.isArray(questions) || questions.length === 0) {
            resultsContainer.insertAdjacentHTML('beforeend', '<p>No questions generated.</p>');
            return;
        }

        questions.forEach((q, index) => {
            const div = document.createElement('div');
            div.className = 'question-item';

            let optionsHtml = '';
            if (q.options && q.options.length > 0) {
                optionsHtml = `<ul class="options-list">
                    ${q.options.map(opt => `
                        <li class="${opt.isCorrect ? 'correct' : ''}">
                            ${opt.id}) ${opt.label} ${opt.isCorrect ? '✓' : ''}
                        </li>
                    `).join('')}
                </ul>`;
            }

            const correctAnswer = q.correctAnswer || q.answer || '';
            const marks = q.marks ? ` &nbsp;·&nbsp; ${q.marks} marks` : '';

            div.innerHTML = `
                <div class="question-header">
                    <span class="tag ${q.difficulty}">${q.difficulty || 'medium'}</span>
                    <span class="tag">${q.type || 'mcq'}</span>
                    <span style="margin-left:auto; font-size:0.8rem; color:#94a3b8;">Q${index + 1}${marks}</span>
                </div>
                <h3>${q.question}</h3>
                ${optionsHtml}
                <p style="margin-top: 0.75rem; font-size: 0.85rem; color: #4ade80; border-top: 1px solid #1e293b; padding-top: 0.5rem;">
                    <strong>✓ Answer:</strong> ${correctAnswer}
                </p>
            `;
            resultsContainer.appendChild(div);
        });
    }
}

// Description Logic
function setupDescription() {
    const form = document.getElementById('desc-form');
    const resultsContainer = document.getElementById('desc-results');
    const contentBox = resultsContainer.querySelector('.content-box');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const btn = form.querySelector('button');
        btn.disabled = true;
        btn.textContent = 'Generating...';
        resultsContainer.classList.add('hidden');

        const title = document.getElementById('desc-title').value;

        try {
            const response = await fetch(`${API_URL}/api/generate-description`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title, type: document.getElementById('desc-type')?.value || 'lesson' })
            });

            if (!response.ok) throw new Error(await response.text());

            const result = await response.json();
            contentBox.textContent = result.description;
            resultsContainer.classList.remove('hidden');

        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Generate';
        }
    });
}

// Summary Logic
function setupSummary() {
    const form = document.getElementById('summary-form');
    const resultsContainer = document.getElementById('summary-results');
    const summaryText = resultsContainer.querySelector('.summary-text');
    const keyPointsList = resultsContainer.querySelector('.points-list');
    const summaryLang = document.getElementById('summary-lang');
    const summaryWords = document.getElementById('summary-words');
    const summaryChunks = document.getElementById('summary-chunks');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const btn = form.querySelector('button');
        btn.disabled = true;
        btn.textContent = 'Generating...';
        resultsContainer.classList.add('hidden');

        const payload = {
            fileId: document.getElementById('summary-file-id').value,
            summaryLength: document.getElementById('summary-length').value,
            language: document.getElementById('summary-language').value || null
        };

        try {
            const response = await fetch(`${API_URL}/api/summarise`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error(await response.text());

            const result = await response.json();
            renderSummary(result);

        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Generate Summary';
        }
    });

    function renderSummary(summary) {
        resultsContainer.classList.remove('hidden');

        // Update metadata
        summaryLang.textContent = summary.language;
        summaryWords.textContent = `${summary.wordCount} words`;
        summaryChunks.textContent = `${summary.chunksUsed} chunks`;

        // Update content
        summaryText.textContent = summary.summary;

        // Update key points
        keyPointsList.innerHTML = '';
        summary.keyPoints.forEach(point => {
            const li = document.createElement('li');
            li.textContent = point;
            keyPointsList.appendChild(li);
        });
    }
}

// Transcript Logic
function setupTranscript() {
    const form = document.getElementById('transcript-form');
    const resultsContainer = document.getElementById('transcript-results');
    const transcriptText = resultsContainer.querySelector('.transcript-text');
    const transcriptLang = document.getElementById('transcript-lang');
    const transcriptDuration = document.getElementById('transcript-duration');
    const transcriptWords = document.getElementById('transcript-words');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const btn = form.querySelector('button');
        btn.disabled = true;
        btn.textContent = 'Retrieving...';
        resultsContainer.classList.add('hidden');

        const fileId = document.getElementById('transcript-file-id').value;
        const format = document.getElementById('transcript-format').value;

        try {
            if (format === 'json') {
                const jsonResponse = await fetch(`${API_URL}/api/get-transcript-raw/${fileId}`);
                if (!jsonResponse.ok) throw new Error('Raw transcript JSON not found for this file.');
                const rawJson = await jsonResponse.json();
                renderRawJson(rawJson);
            } else {
                // Get chunks from embedding service
                const chunksResponse = await fetch(`${API_URL}/api/get-chunks/${fileId}`);
                if (!chunksResponse.ok) throw new Error('File not found or no chunks available');

                const chunks = await chunksResponse.json();
                renderTranscript(chunks, format);
            }

        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Get Transcript';
        }
    });

    function renderRawJson(data) {
        resultsContainer.classList.remove('hidden');
        transcriptLang.textContent = data.language || 'unknown';
        transcriptWords.textContent = 'Raw JSON';
        transcriptDuration.textContent = `${data.segments?.length || 0} segments`;

        transcriptText.innerHTML = `<pre class="json-block">${JSON.stringify(data, null, 2)}</pre>`;
    }

    function renderTranscript(chunks, format) {
        resultsContainer.classList.remove('hidden');

        // Calculate metadata
        const totalWords = chunks.reduce((sum, chunk) => sum + chunk.text.split(' ').length, 0);
        const detectedLang = chunks[0]?.metadata?.language || 'unknown';

        // Update metadata
        transcriptLang.textContent = detectedLang;
        transcriptWords.textContent = `${totalWords} words`;
        transcriptDuration.textContent = `${chunks.length} segments`;

        // Format transcript based on selected format
        let transcriptContent = '';

        if (format === 'plain') {
            transcriptContent = chunks.map(chunk => chunk.text).join('\n\n');
        } else if (format === 'timestamps') {
            transcriptContent = chunks.map(chunk => {
                const timestamp = chunk.metadata?.timestamp || '00:00:00';
                return `[${timestamp}] ${chunk.text}`;
            }).join('\n\n');
        } else if (format === 'segments') {
            transcriptContent = chunks.map((chunk, index) => {
                const metadata = chunk.metadata || {};
                return `
                    <div class="transcript-segment">
                        <div class="transcript-timestamp">
                            Segment ${index + 1}${metadata.timestamp ? ` - ${metadata.timestamp}` : ''}
                        </div>
                        <div>${chunk.text}</div>
                    </div>
                `;
            }).join('');
        }

        transcriptText.innerHTML = transcriptContent;

        // Setup copy/download buttons
        setupTranscriptActions(chunks, format);
    }

    function setupTranscriptActions(chunks, format) {
        const copyBtn = document.getElementById('copy-transcript');
        const downloadBtn = document.getElementById('download-transcript');

        // Copy functionality
        copyBtn.onclick = async () => {
            try {
                const text = chunks.map(chunk => chunk.text).join('\n\n');
                await navigator.clipboard.writeText(text);
                copyBtn.textContent = '✓ Copied!';
                setTimeout(() => {
                    copyBtn.textContent = '📋 Copy Transcript';
                }, 2000);
            } catch (error) {
                alert('Failed to copy to clipboard');
            }
        };

        // Download functionality
        downloadBtn.onclick = () => {
            let content = '';
            let filename = `transcript-${Date.now()}`;

            if (format === 'plain') {
                content = chunks.map(chunk => chunk.text).join('\n\n');
                filename += '.txt';
            } else if (format === 'timestamps') {
                content = chunks.map(chunk => {
                    const timestamp = chunk.metadata?.timestamp || '00:00:00';
                    return `[${timestamp}] ${chunk.text}`;
                }).join('\n\n');
                filename += '-timestamps.txt';
            } else if (format === 'segments') {
                content = chunks.map((chunk, index) => {
                    const metadata = chunk.metadata || {};
                    return `--- Segment ${index + 1}${metadata.timestamp ? ` (${metadata.timestamp})` : ''} ---\n${chunk.text}\n`;
                }).join('\n\n');
                filename += '-segments.txt';
            }

            const blob = new Blob([content], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        };
    }
}

// Flashcards Logic
function setupFlashcards() {
    const form = document.getElementById('flashcards-form');
    const resultsContainer = document.getElementById('flashcards-results');

    if(!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = form.querySelector('button');
        btn.disabled = true;
        btn.textContent = 'Generating...';
        resultsContainer.classList.add('hidden');
        resultsContainer.innerHTML = '';

        const payload = {
            subject: document.getElementById('flashcard-subject').value,
            chapter: document.getElementById('flashcard-chapter').value,
            topic: document.getElementById('flashcard-topic').value,
            goal: document.getElementById('flashcard-goal').value || null,
            numberOfCards: parseInt(document.getElementById('flashcard-count').value)
        };
        payload.language = inferOutputLanguage(payload.subject, payload.chapter, payload.topic, payload.goal);
        console.log('/api/generate-flashcards payload', payload);

        try {
            const response = await fetch(`${API_URL}/api/generate-flashcards`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error(await response.text());
            const cards = await response.json();
            
            resultsContainer.classList.remove('hidden');
            renderPayloadDebug(resultsContainer, '/api/generate-flashcards', payload);
            if (cards.length === 0) {
                resultsContainer.insertAdjacentHTML('beforeend', '<p>No flashcards generated.</p>');
                return;
            }

            cards.forEach((c, i) => {
                const div = document.createElement('div');
                div.className = 'content-box';
                div.style.marginBottom = '10px';
                div.innerHTML = `<strong>Card ${i+1} Front:</strong> ${c.front}<br><br><strong>Back:</strong> ${c.back}`;
                resultsContainer.appendChild(div);
            });
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Generate Flashcards';
        }
    });
}

// Quiz Logic
function setupQuiz() {
    const form = document.getElementById('quiz-form');
    const resultsContainer = document.getElementById('quiz-results');

    if(!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = form.querySelector('button');
        btn.disabled = true;
        btn.textContent = 'Generating...';
        resultsContainer.classList.add('hidden');
        resultsContainer.innerHTML = '';

        const payload = {
            subject: document.getElementById('quiz-subject').value,
            chapter: document.getElementById('quiz-chapter').value,
            difficulty: document.getElementById('quiz-difficulty').value,
            numberOfQuestions: parseInt(document.getElementById('quiz-count').value),
            fileId: document.getElementById('quiz-file-id').value.trim(),
            prompt: document.getElementById('quiz-prompt').value.trim()
        };
        payload.language = inferOutputLanguage(payload.subject, payload.chapter, payload.prompt);
        console.log('/api/generate-quiz payload', payload);

        try {
            const response = await fetch(`${API_URL}/api/generate-quiz`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error(await response.text());
            const questions = await response.json();
            
            resultsContainer.classList.remove('hidden');
            renderPayloadDebug(resultsContainer, '/api/generate-quiz', payload);
            if (questions.length === 0) {
                resultsContainer.insertAdjacentHTML('beforeend', '<p>No questions generated.</p>');
                return;
            }

            questions.forEach((q, i) => {
                const div = document.createElement('div');
                div.className = 'question-item';
                div.innerHTML = `
                    <div class="question-header">
                        <span class="tag">MCQ</span>
                    </div>
                    <h3>Q${i + 1}: ${q.question}</h3>
                    <ul class="options-list">
                        ${q.options.map(opt => `
                            <li class="${opt.isCorrect ? 'correct' : ''}">
                                ${opt.text} ${opt.isCorrect ? '✓' : ''}
                            </li>
                        `).join('')}
                    </ul>
                    <p style="margin-top: 1rem; font-size: 0.9rem; color: #94a3b8;">
                        <strong>Explanation:</strong> ${q.explanation}
                    </p>
                `;
                resultsContainer.appendChild(div);
            });
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Generate Quiz';
        }
    });
}

// Ask AI Logic
function setupAskAI() {
    const form = document.getElementById('ask-ai-form');
    const resultsContainer = document.getElementById('ask-ai-results');

    if(!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = form.querySelector('button');
        btn.disabled = true;
        btn.textContent = 'Asking...';
        resultsContainer.classList.add('hidden');

        const payload = {
            question: document.getElementById('ask-question').value,
            previousAnswer: document.getElementById('ask-previous').value || null
        };

        try {
            const response = await fetch(`${API_URL}/api/ask-ai`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error(await response.text());
            const res = await response.json();
            
            resultsContainer.classList.remove('hidden');
            document.getElementById('ask-res-q').textContent = res.question;
            document.getElementById('ask-res-expl').textContent = res.explanation;
            
            const ul = document.getElementById('ask-res-examples');
            ul.innerHTML = '';
            if (res.examples && res.examples.length) {
                res.examples.forEach(ex => {
                    const li = document.createElement('li');
                    li.textContent = ex;
                    ul.appendChild(li);
                });
            } else {
                ul.innerHTML = '<li>No examples provided.</li>';
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Ask';
        }
    });
}

// AI Assistant Logic
function setupAssistant() {
    const form = document.getElementById('assistant-form');
    const resultsContainer = document.getElementById('assistant-results');

    if(!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = form.querySelector('button');
        btn.disabled = true;
        btn.textContent = 'Thinking...';
        resultsContainer.classList.add('hidden');

        const payload = {
            fileId: document.getElementById('ast-file-id').value,
            student_id: document.getElementById('ast-student-id').value,
            tenant_id: document.getElementById('ast-tenant-id').value,
            message: document.getElementById('ast-message').value,
            course: document.getElementById('ast-course').value || "",
            module: document.getElementById('ast-module').value || "",
            lesson: document.getElementById('ast-lesson').value || ""
        };

        try {
            const response = await fetch(`${API_URL}/api/ai-assistant`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error(await response.text());
            const res = await response.json();
            
            resultsContainer.classList.remove('hidden');
            document.getElementById('ast-response').textContent = res.response;
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Send Message';
        }
    });
}

// Analytics Logic
function setupAnalytics() {
    const tenantInput = document.getElementById('analytics-tenant-id');
    const initialTenantId = new URLSearchParams(window.location.search).get('tenantId');
    if (initialTenantId && tenantInput) {
        tenantInput.value = initialTenantId;
    }
    const analyticsUrl = (path) => {
        const tenantId = tenantInput?.value?.trim();
        if (!tenantId || !Number.isInteger(Number(tenantId)) || Number(tenantId) < 1) {
            throw new Error('Enter a valid numeric Tenant ID.');
        }
        return `${API_URL}${path}?tenantId=${encodeURIComponent(tenantId)}`;
    };
    const btnCompletion = document.getElementById('btn-refresh-completion');
    const btnPerformance = document.getElementById('btn-refresh-performance');
    const btnRevenue = document.getElementById('btn-refresh-revenue');
    const btnAI = document.getElementById('btn-ai-analyze');
    const resultsContainer = document.getElementById('analytics-results');
    const dataView = document.getElementById('analytics-data-view');
    const aiView = document.getElementById('analytics-ai-view');
    const tableContainer = document.getElementById('analytics-table-container');
    const aiContent = document.getElementById('analytics-ai-content');
    const titleEl = document.getElementById('analytics-title');

    const fetchAndRender = async (endpoint, title) => {
        resultsContainer.classList.remove('hidden');
        dataView.classList.remove('hidden');
        aiView.classList.add('hidden');
        titleEl.textContent = title;
        tableContainer.innerHTML = '<p>Loading...</p>';

        try {
            const response = await fetch(analyticsUrl(`/api/analytics/${endpoint}`));
            if (!response.ok) throw new Error('Failed to fetch analytics');
            const data = await response.json();
            
            if (!data || data.length === 0) {
                tableContainer.innerHTML = '<p>No data available.</p>';
                return;
            }

            // Simple table rendering
            const keys = Object.keys(data[0]);
            let html = '<table class="analytics-table"><thead><tr>';
            keys.forEach(k => html += `<th>${k}</th>`);
            html += '</tr></thead><tbody>';
            data.forEach(row => {
                html += '<tr>';
                keys.forEach(k => {
                    let val = row[k];
                    if (typeof val === 'number') val = val.toFixed(2);
                    html += `<td>${val}</td>`;
                });
                html += '</tr>';
            });
            html += '</tbody></table>';
            tableContainer.innerHTML = html;
        } catch (error) {
            tableContainer.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        }
    };

    btnCompletion.addEventListener('click', () => fetchAndRender('completion', 'Course Completion Progress'));
    btnPerformance.addEventListener('click', () => fetchAndRender('performance', 'Student Grade Performance'));
    btnRevenue.addEventListener('click', () => fetchAndRender('revenue', 'Monthly Revenue Trends'));

    btnAI.addEventListener('click', async () => {
        resultsContainer.classList.remove('hidden');
        aiView.classList.remove('hidden');
        dataView.classList.add('hidden');
        aiContent.textContent = 'Analyzing data with AI...';
        btnAI.disabled = true;

        try {
            const response = await fetch(analyticsUrl('/api/ai-insights'));
            if (!response.ok) throw new Error('Failed to generate AI analysis');
            const data = await response.json();
            aiContent.textContent = data.length ? JSON.stringify(data, null, 2) : 'No actionable insights.';
        } catch (error) {
            aiContent.textContent = `Error: ${error.message}`;
        } finally {
            btnAI.disabled = false;
        }
    });
}
