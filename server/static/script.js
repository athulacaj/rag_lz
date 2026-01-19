document.addEventListener('DOMContentLoaded', () => {
    // Connect to Socket.IO
    const socket = io();
    const logContainer = document.getElementById('log-container');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');

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

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 480000); // 8 minutes timeout

            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: text }),
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            const data = await response.json();

            if (data.error) {
                addMessage('Error: ' + data.error, 'bot');
            } else {
                addMessage(data.response, 'bot');
            }
        } catch (error) {
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

        // Simple markdown-ish to html replacement for newlines
        // For full markdown we'd use a library, but basic whitespace is key
        contentDiv.innerHTML = text.replace(/\n/g, '<br>');

        msgDiv.appendChild(contentDiv);
        chatMessages.appendChild(msgDiv);

        // Auto-scroll
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});
