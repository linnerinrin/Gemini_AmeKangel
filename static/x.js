let abortController = null;
let currentReader = null;
let currentOutputDiv = null;

function getCurrentModeName() {
    const modeBadge = document.getElementById('modeBadge');
    if (modeBadge) {
        const modeText = modeBadge.innerHTML;
        if (modeText.includes('超天酱')) {
            return '超天酱';
        } else if (modeText.includes('糖糖')) {
            return '糖糖';
        }
    }
    const theme = document.documentElement.getAttribute('data-theme');
    return theme === 'kangel' ? '超天酱' : '糖糖';
}

// 限制结果区域的行数
function limitResultLines() {
    const resultDiv = document.getElementById("result");
    const children = resultDiv.children;
    const maxLines = 100; // 保留最近100行

    // 计算大约的行数（每个div元素算一行或一个块）
    if (children.length > maxLines) {
        const removeCount = children.length - maxLines;
        for (let i = 0; i < removeCount; i++) {
            if (children[i]) {
                children[i].remove();
            }
        }
    }
}

// 添加消息到结果区域
function addMessageToResult(content, type = 'text') {
    const resultDiv = document.getElementById("result");

    if (type === 'theme') {
        const themeDiv = document.createElement('div');
        themeDiv.className = 'theme-indicator';
        themeDiv.textContent = content;
        resultDiv.appendChild(themeDiv);
    } else if (type === 'status') {
        const statusDiv = document.createElement('div');
        statusDiv.className = 'status completion';
        statusDiv.textContent = content;
        resultDiv.appendChild(statusDiv);
    } else if (type === 'error') {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'status';
        errorDiv.style.color = 'red';
        errorDiv.textContent = content;
        resultDiv.appendChild(errorDiv);
    } else if (type === 'thinking') {
        const thinkingDiv = document.createElement('div');
        thinkingDiv.className = 'status';
        thinkingDiv.textContent = content;
        resultDiv.appendChild(thinkingDiv);
    } else {
        // 普通文本字符
        if (!currentOutputDiv) {
            currentOutputDiv = document.createElement('div');
            currentOutputDiv.className = 'output-text';
            resultDiv.appendChild(currentOutputDiv);
        }
        const charSpan = document.createElement('span');
        charSpan.className = 'char';
        charSpan.textContent = content;
        currentOutputDiv.appendChild(charSpan);
        charSpan.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // 限制行数
    limitResultLines();

    // 滚动到底部
    resultDiv.scrollTop = resultDiv.scrollHeight;
}

// 开始新的输出块
function startNewOutputBlock() {
    currentOutputDiv = document.createElement('div');
    currentOutputDiv.className = 'output-text';
    document.getElementById("result").appendChild(currentOutputDiv);
    return currentOutputDiv;
}

// 清空当前输出块
function clearCurrentOutput() {
    currentOutputDiv = null;
}

async function startStream() {
    const text = document.getElementById("inputText").value;
    if (!text.trim()) {
        alert("请输入内容");
        return;
    }

    if (abortController) {
        abortController.abort();
    }

    // 不清空整个结果区域，只添加思考状态
    addMessageToResult('思考中...', 'thinking');

    const sendBtn = document.getElementById("sendBtn");
    const cancelBtn = document.getElementById("cancelBtn");
    sendBtn.disabled = true;
    sendBtn.textContent = "输出中...";
    cancelBtn.style.display = "inline-block";

    abortController = new AbortController();

    try {
        const response = await fetch("/api/post-demo", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content: text }),
            signal: abortController.signal
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        // 开始新的输出块
        startNewOutputBlock();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || "";

            for (const line of lines) {
                const match = line.match(/^data:\s*(.*)$/);
                if (match && match[1]) {
                    const message = match[1];

                    if (message.trim() === '') continue;

                    if (message.includes('【') && message.includes('】')) {
                        addMessageToResult(message, 'theme');
                        startNewOutputBlock(); // 主题后新起一个输出块
                    } else if (message.includes('[完成]') || message.includes('✅')) {
                        addMessageToResult(message, 'status');
                        clearCurrentOutput();
                    } else {
                        const charSpan = document.createElement('span');
                        charSpan.className = 'char';
                        charSpan.textContent = message;
                        if (currentOutputDiv) {
                            currentOutputDiv.appendChild(charSpan);
                        } else {
                            startNewOutputBlock();
                            currentOutputDiv.appendChild(charSpan);
                        }
                        charSpan.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                        document.getElementById("result").scrollTop = document.getElementById("result").scrollHeight;
                    }
                }
            }
        }

        // 移除思考中的状态（如果还存在）
        const thinkingDivs = document.querySelectorAll('#result .status');
        for (let div of thinkingDivs) {
            if (div.textContent.includes('思考中')) {
                div.remove();
            }
        }

    } catch (error) {
        if (error.name === 'AbortError') {
            addMessageToResult('⏸️ 已取消输出', 'error');
        } else {
            console.error("Error:", error);
            addMessageToResult(`❌ 错误: ${error.message}`, 'error');
        }
    } finally {
        sendBtn.disabled = false;
        sendBtn.textContent = "发送";
        cancelBtn.style.display = "none";
        abortController = null;
        clearCurrentOutput();
    }
}

function cancelStream() {
    if (abortController) {
        abortController.abort();
        abortController = null;
    }
}

// ========== 历史记录功能 ==========

async function showHistoryModal() {
    const sidebar = document.getElementById('historySidebar');
    const historyContent = document.getElementById('historyContent');

    sidebar.classList.add('active');
    historyContent.innerHTML = '<div class="history-loading">📖 加载历史记录中...</div>';

    try {
        const response = await fetch("/api/history", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });

        const data = await response.json();

        if (response.ok && data.history) {
            renderHistory(data.history);
        } else {
            historyContent.innerHTML = '<div class="history-placeholder">❌ 加载失败，请重试</div>';
        }
    } catch (error) {
        console.error("获取历史记录失败:", error);
        historyContent.innerHTML = '<div class="history-placeholder">❌ 网络错误，请检查连接</div>';
    }
}

function renderHistory(history) {
    const historyContent = document.getElementById('historyContent');

    if (!history || history.length === 0) {
        historyContent.innerHTML = '<div class="history-placeholder">📭 暂无对话记录，开始聊天吧！</div>';
        return;
    }

    const assistantName = getCurrentModeName();

    let html = '';
    for (let i = 0; i < history.length; i++) {
        const item = history[i];
        const role = item.role;
        const content = item.content;

        if (role === 'user') {
            html += `
                <div class="history-item user">
                    <div class="history-role">用户</div>
                    <div class="history-content-text">${escapeHtml(content)}</div>
                </div>
            `;
        } else if (role === 'assistant') {
            html += `
                <div class="history-item assistant">
                    <div class="history-role">${assistantName}</div>
                    <div class="history-content-text">${escapeHtml(content)}</div>
                </div>
            `;
        }
    }

    historyContent.innerHTML = html;

    setTimeout(() => {
        historyContent.scrollTop = historyContent.scrollHeight;
    }, 100);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function closeHistorySidebar() {
    const sidebar = document.getElementById('historySidebar');
    sidebar.classList.remove('active');
}

// ========== 主题切换功能 ==========

async function selectTheme(theme) {
    closeModal();

    const resultDiv = document.getElementById('result');
    // 不清空整个结果，只添加主题切换标识
    addMessageToResult(`🎨 正在切换到 ${theme === 'kangel' ? '超天酱' : '糖糖'} 模式...`, 'status');

    const sendBtn = document.getElementById("sendBtn");
    sendBtn.disabled = true;

    try {
        const response = await fetch("/api/change-with-animation", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content: theme })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        startNewOutputBlock();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || "";

            for (const line of lines) {
                const match = line.match(/^data:\s*(.*)$/);
                if (match && match[1]) {
                    const message = match[1];

                    if (message.trim() === '') continue;

                    if (message.includes('【') && message.includes('】')) {
                        addMessageToResult(message, 'theme');
                        startNewOutputBlock();
                    } else if (message.includes('✅')) {
                        addMessageToResult(message, 'status');
                        clearCurrentOutput();
                    } else {
                        const charSpan = document.createElement('span');
                        charSpan.className = 'char';
                        charSpan.textContent = message;
                        if (currentOutputDiv) {
                            currentOutputDiv.appendChild(charSpan);
                        } else {
                            startNewOutputBlock();
                            currentOutputDiv.appendChild(charSpan);
                        }
                        charSpan.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                        document.getElementById("result").scrollTop = document.getElementById("result").scrollHeight;
                    }
                }
            }
        }

        applyTheme(theme);

    } catch (error) {
        console.error("主题切换请求失败:", error);
        addMessageToResult(`⚠️ 主题切换出错: ${error.message}`, 'error');
    } finally {
        sendBtn.disabled = false;
        clearCurrentOutput();
    }
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('selectedTheme', theme);

    const modeBadge = document.getElementById('modeBadge');
    if (modeBadge) {
        if (theme === 'kangel') {
            modeBadge.innerHTML = '🌸 超天酱';
            modeBadge.style.background = '#e83e8c';
        } else {
            modeBadge.innerHTML = '🖤 糖糖';
            modeBadge.style.background = '#6c757d';
        }
    }
}

function loadSavedTheme() {
    const savedTheme = localStorage.getItem('selectedTheme');
    if (savedTheme === 'kangel') {
        applyTheme('kangel');
    } else {
        applyTheme('ame');
    }
}

function showThemeModal() {
    const modal = document.getElementById('themeModal');
    modal.style.display = 'block';
}

function closeModal() {
    const modal = document.getElementById('themeModal');
    modal.style.display = 'none';
}

window.onclick = function(event) {
    const modal = document.getElementById('themeModal');

    if (event.target == modal) {
        closeModal();
    }
}

// 初始化
loadSavedTheme();