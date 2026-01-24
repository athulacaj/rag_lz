document.addEventListener('DOMContentLoaded', () => {
    // Connect to Socket.IO
    const socket = io();
    const logContainer = document.getElementById('log-container');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');

    // Tabs
    const tabLogs = document.getElementById('tab-logs');
    const tabHistory = document.getElementById('tab-history');
    const tabSettings = document.getElementById('tab-settings');
    const historyContainer = document.getElementById('history-container');
    const settingsContainer = document.getElementById('settings-container');

    // Config Elements
    const dbNameDisplay = document.getElementById('db-name-display');
    const embeddingSelect = document.getElementById('embedding-model-select');
    const llmSelect = document.getElementById('llm-model-select');
    const parserSelect = document.getElementById('parser-select');

    // Status Bar Elements
    const statusBar = document.getElementById('model-status-bar');
    const statusEmbedding = document.getElementById('status-embedding');
    const statusLlm = document.getElementById('status-llm');

    // New Chat Button
    const newChatBtn = document.getElementById('new-chat-btn');

    // Load Config on Start
    loadConfig();
    // Load History on Start (since it is the default tab)
    loadHistory();

    async function loadConfig() {
        try {
            const res = await fetch('/config');
            const config = await res.json();

            dbNameDisplay.textContent = config.DB_NAME;

            // Helper to populate select
            const populate = (select, options, storageKey) => {
                select.innerHTML = '';
                options.forEach(opt => {
                    const el = document.createElement('option');
                    el.value = opt;
                    el.textContent = opt;
                    select.appendChild(el);
                });

                // Load from storage
                const saved = localStorage.getItem(storageKey);
                if (saved && options.includes(saved)) {
                    select.value = saved;
                }

                // Initial update of status bar if this is one of the relevant selects
                updateStatusBar();

                // Save on change
                select.addEventListener('change', () => {
                    localStorage.setItem(storageKey, select.value);
                    updateStatusBar();
                });
            };

            populate(embeddingSelect, config.EMBEDDING_MODELS, 'rag_embedding_model');
            populate(llmSelect, config.MODEL_COLLECTIONS, 'rag_llm_model');
            populate(parserSelect, config.PARSER_LIST, 'rag_parser');

            updateStatusBar(); // Ensure updated initially

        } catch (e) {
            console.error("Error loading config:", e);
        }
    }

    function updateStatusBar() {
        if (embeddingSelect.value) {
            statusEmbedding.textContent = `Embedding: ${embeddingSelect.value}`;
        }
        if (llmSelect.value) {
            statusLlm.textContent = `LLM: ${llmSelect.value}`;
        }
    }

    statusBar.addEventListener('click', () => {
        tabSettings.click();
    });

    newChatBtn.addEventListener('click', () => {
        // Reset Chat UI
        chatMessages.innerHTML = `
            <div class="message bot">
                <div class="message-content">Hello! How can I help you today?</div>
            </div>
        `;

        // Reset Logs
        logContainer.innerHTML = '<div class="log-entry system">Waiting for logs...</div>';

        // Deselect history items
        document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
    });

    tabLogs.addEventListener('click', () => {
        tabLogs.classList.add('active');
        tabHistory.classList.remove('active');
        tabSettings.classList.remove('active');
        logContainer.classList.remove('hidden');
        historyContainer.classList.add('hidden');
        settingsContainer.classList.add('hidden');
    });

    tabHistory.addEventListener('click', () => {
        tabHistory.classList.add('active');
        tabLogs.classList.remove('active');
        tabSettings.classList.remove('active');
        historyContainer.classList.remove('hidden');
        logContainer.classList.add('hidden');
        settingsContainer.classList.add('hidden');
        loadHistory();
    });

    tabSettings.addEventListener('click', () => {
        tabSettings.classList.add('active');
        tabLogs.classList.remove('active');
        tabHistory.classList.remove('active');
        settingsContainer.classList.remove('hidden');
        logContainer.classList.add('hidden');
        historyContainer.classList.add('hidden');
    });

    async function loadHistory() {
        historyContainer.innerHTML = '<div class="log-entry system">Loading history...</div>';
        try {
            const res = await fetch('/history');
            const history = await res.json();

            historyContainer.innerHTML = '';
            if (history.length === 0) {
                historyContainer.innerHTML = '<div class="log-entry system">No history found.</div>';
                return;
            }

            history.forEach(item => {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'history-item';
                itemDiv.dataset.id = item.id; // Store ID in dataset
                itemDiv.innerHTML = `
                    <div class="history-question">${item.question}</div>
                    <div class="history-date">${new Date(item.timestamp).toLocaleString()}</div>
                `;

                itemDiv.addEventListener('click', function () {
                    // Remove active class from all items
                    document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
                    // Add active class to clicked item
                    this.classList.add('active');

                    loadHistoryDetails(this.dataset.id);
                });

                historyContainer.appendChild(itemDiv);
            });
        } catch (e) {
            historyContainer.innerHTML = `<div class="log-entry ERROR">Error loading history: ${e.message}</div>`;
        }
    }

    async function loadHistoryDetails(id) {
        try {
            // Add timestamp to prevent caching
            const res = await fetch(`/history/${id}?t=${new Date().getTime()}`);
            const data = await res.json();

            if (data.error) {
                alert(data.error);
                return;
            }

            // Clear chat and show history
            chatMessages.innerHTML = '';

            // Add a separator or indicator
            const refDiv = document.createElement('div');
            refDiv.style.textAlign = 'center';
            refDiv.style.color = '#888';
            refDiv.style.margin = '10px 0';
            refDiv.innerText = `Viewing History: ${new Date(data.timestamp).toLocaleString()}`;
            chatMessages.appendChild(refDiv);

            addMessage(data.question, 'user');
            addMessage(data.answer, 'bot');

            // Show Context if available
            if (data.context) {
                const contextDiv = document.createElement('div');
                contextDiv.className = 'context-container';

                const toggleBtn = document.createElement('button');
                toggleBtn.textContent = 'Show Context';
                toggleBtn.className = 'toggle-context-btn';

                const contentPre = document.createElement('pre');
                contentPre.className = 'context-content hidden';
                contentPre.textContent = data.context;

                toggleBtn.addEventListener('click', () => {
                    if (contentPre.classList.contains('hidden')) {
                        contentPre.classList.remove('hidden');
                        toggleBtn.textContent = 'Hide Context';
                    } else {
                        contentPre.classList.add('hidden');
                        toggleBtn.textContent = 'Show Context';
                    }
                });

                contextDiv.appendChild(toggleBtn);
                contextDiv.appendChild(contentPre);
                chatMessages.appendChild(contextDiv);
            }

            // Show logs
            logContainer.innerHTML = '';
            if (data.logs) {
                const logs = data.logs.split('\n');
                logs.forEach(logLine => {
                    if (!logLine.trim()) return;
                    const entry = document.createElement('div');
                    entry.className = 'log-entry';
                    if (logLine.includes('ERROR')) entry.classList.add('ERROR');
                    else if (logLine.includes('WARNING')) entry.classList.add('WARNING');
                    else entry.classList.add('INFO');
                    entry.textContent = logLine;
                    logContainer.appendChild(entry);
                });
            } else {
                logContainer.innerHTML = '<div class="log-entry system">No logs recorded for this session.</div>';
            }

            // Switch to Logs tab to see the logs
            tabLogs.click();

        } catch (e) {
            console.error(e);
            alert('Error loading details');
        }
    }

    // Handle WebSocket Logs
    socket.on('log_message', (msg) => {
        const entry = document.createElement('div');
        entry.className = 'log-entry';

        // Simple heuristic for log level styling
        if (msg.data.includes('ERROR')) entry.classList.add('ERROR');
        else if (msg.data.includes('WARNING')) entry.classList.add('WARNING');
        else entry.classList.add('INFO');

        entry.textContent = msg.data;
        logContainer.appendChild(entry);

        // Auto-scroll
        logContainer.scrollTop = logContainer.scrollHeight;
    });

    // Handle Chat Submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        // Add user message to UI
        addMessage(text, 'user');
        userInput.value = '';

        // Disable input while processing
        const button = chatForm.querySelector('button');
        button.disabled = true;

        showLoading();


        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 480000); // 8 minutes timeout

            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: text,
                    db_name: dbNameDisplay.textContent,
                    embedding_model: embeddingSelect.value,
                    model: llmSelect.value,
                    parser: parserSelect.value
                }),
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            const data = await response.json();

            hideLoading();


            if (data.error) {
                addMessage('Error: ' + data.error, 'bot');
            } else {
                addMessage(data.response, 'bot');
            }
        } catch (error) {
            hideLoading();
            addMessage('Network Error: ' + error.message, 'bot');
        } finally {

            button.disabled = false;
        }
    });

    function addMessage(text, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        // Simple markdown formatting
        let formatted = text
            // Bold: **text**
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Bullet points: * text (simple replacement with bullet character)
            .replace(/(^|\n)\*\s/g, '$1• ')
            // Newlines
            .replace(/\n/g, '<br>');

        contentDiv.innerHTML = formatted;

        msgDiv.appendChild(contentDiv);
        chatMessages.appendChild(msgDiv);

        // Auto-scroll
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showLoading() {
        const loaderDiv = document.createElement('div');
        loaderDiv.className = 'typing-indicator-container';
        loaderDiv.id = 'loading-indicator';
        loaderDiv.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        chatMessages.appendChild(loaderDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function hideLoading() {
        const loader = document.getElementById('loading-indicator');
        if (loader) {
            loader.remove();
        }
    }
});

