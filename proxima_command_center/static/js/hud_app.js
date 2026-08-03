// High-Performance Cyberpunk HUD Controller for Proxima X Command Center

let ws = null;

function initDashboard() {
    console.log("⚡ Initializing Proxima X Cyberpunk Telemetry HUD...");
    connectWebSocket();
    startLocalTimer();
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
    
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log("🟢 WebSocket Telemetry Feed Connected!");
        document.getElementById("connection-status").innerText = "CONNECTED";
        document.getElementById("connection-status").style.color = "#00ff88";
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        renderPredictions(data.predictions);
        renderMT5Trades(data.mt5_trades);
    };

    ws.onclose = () => {
        console.log("🔴 WebSocket Disconnected. Reconnecting in 3s...");
        document.getElementById("connection-status").innerText = "RECONNECTING";
        document.getElementById("connection-status").style.color = "#ffaa00";
        setTimeout(connectWebSocket, 3000);
    };
}

function renderPredictions(predictions) {
    const grid = document.getElementById("strategy-grid");
    if (!grid) return;

    grid.innerHTML = predictions.map(p => `
        <div class="strategy-card">
            <div class="card-top">
                <span class="card-title">${p.name}</span>
                <span class="conf-pill">${p.confidence}% CONF</span>
            </div>
            <div class="card-metrics">
                <div class="metric-box">
                    <div class="metric-label">NEXT SYMBOL</div>
                    <div class="metric-value" style="color:#00f0ff;">${p.next_symbol}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">DIRECTION</div>
                    <div class="metric-value" style="color:${p.direction === 'LONG' || p.direction === 'BUY' ? '#00ff88' : '#ff0055'};">${p.direction}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">WIN RATE / PF</div>
                    <div class="metric-value">${p.win_rate}% / PF ${p.profit_factor}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">PROJECTED WIN</div>
                    <div class="metric-value" style="color:#00ff88;">+$${p.avg_win_usd.toFixed(2)}</div>
                </div>
            </div>
            <div class="countdown-bar">
                <span>⚡ FIRE IN:</span>
                <span class="countdown-timer" data-seconds="${p.seconds_until_fire}">${formatTime(p.seconds_until_fire)}</span>
            </div>
        </div>
    `).join('');
}

function renderMT5Trades(trades) {
    const tbody = document.getElementById("mt5-trades-body");
    if (!tbody) return;

    tbody.innerHTML = trades.map(t => `
        <tr>
            <td style="font-family: Orbitron; color: #64748b;">#${t.ticket}</td>
            <td style="font-weight: 600;">${t.strategy}</td>
            <td style="font-family: Orbitron; color: #00f0ff;">${t.pair}</td>
            <td style="color:${t.type === 'BUY' ? '#00ff88' : '#ff0055'}; font-weight:700;">${t.type}</td>
            <td>${t.lot} L</td>
            <td style="font-family: Orbitron;">${t.open_price}</td>
            <td style="font-family: Orbitron;">${t.close_price}</td>
            <td class="win-text">+${t.pips} pips</td>
            <td class="win-text">+$${t.pnl.toFixed(2)}</td>
            <td><span class="conf-pill">${t.status}</span></td>
        </tr>
    `).join('');
}

function startLocalTimer() {
    setInterval(() => {
        const timers = document.querySelectorAll(".countdown-timer");
        timers.forEach(timer => {
            let secs = parseInt(timer.getAttribute("data-seconds")) || 0;
            if (secs > 0) {
                secs--;
                timer.setAttribute("data-seconds", secs);
                timer.innerText = formatTime(secs);
            }
        });
    }, 1000);
}

function formatTime(totalSeconds) {
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    
    if (hours > 0) {
        return `${hours}h ${minutes}m ${seconds}s`;
    }
    return `${minutes}m ${seconds}s`;
}

document.addEventListener("DOMContentLoaded", initDashboard);
