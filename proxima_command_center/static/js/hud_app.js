// Executive Financial Terminal Controller for Proxima X Command Center

let ws = null;

function initTerminal() {
    console.log("⚡ Initializing Proxima X Executive Financial Terminal...");
    connectWebSocket();
    startLocalTimer();
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
    
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log("🟢 WebSocket Connection Established!");
        const st = document.getElementById("terminal-status");
        if (st) { st.innerText = "ONLINE"; st.style.color = "#10b981"; }
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.config) {
            updateHeaderConfig(data.config);
        }
        if (data.predictions) {
            renderPredictions(data.predictions);
        }
        if (data.mt5_trades) {
            renderMT5Trades(data.mt5_trades);
        }
    };

    ws.onclose = () => {
        console.log("🔴 WebSocket Disconnected. Reconnecting in 3s...");
        const st = document.getElementById("terminal-status");
        if (st) { st.innerText = "RECONNECTING"; st.style.color = "#f59e0b"; }
        setTimeout(connectWebSocket, 3000);
    };
}

function updateHeaderConfig(config) {
    const tag = document.getElementById("account-profile-tag");
    if (tag && config.account_profile) {
        tag.innerText = config.account_profile.toUpperCase();
    }
}

function renderPredictions(predictions) {
    const grid = document.getElementById("strategy-grid");
    if (!grid) return;

    grid.innerHTML = predictions.map(p => `
        <div class="strategy-card">
            <div class="card-header">
                <span class="card-name">${p.name}</span>
                <span class="tag-confidence">${p.confidence}% CONF</span>
            </div>
            <table class="data-table">
                <tr>
                    <td class="label-cell">Target Symbol</td>
                    <td style="color: #3b82f6;">${p.next_symbol}</td>
                </tr>
                <tr>
                    <td class="label-cell">Direction</td>
                    <td style="color: ${p.direction === 'BUY' || p.direction === 'LONG' ? '#10b981' : '#f43f5e'};">${p.direction}</td>
                </tr>
                <tr>
                    <td class="label-cell">Trading Regime</td>
                    <td style="color: #94a3b8; font-size: 0.75rem;">${p.regime}</td>
                </tr>
                <tr>
                    <td class="label-cell">Effective Lot Size</td>
                    <td style="color: #f59e0b;">${p.effective_lot} Lots</td>
                </tr>
                <tr>
                    <td class="label-cell">Target Win ($)</td>
                    <td class="val-green">+$${p.target_win_usd.toFixed(2)}</td>
                </tr>
                <tr>
                    <td class="label-cell">Win Rate / PF</td>
                    <td>${p.win_rate}% / ${p.profit_factor} PF</td>
                </tr>
            </table>
            <div class="countdown-box">
                <span style="color: #64748b; font-size: 0.75rem;">NEXT TRIGGER IN:</span>
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
            <td style="font-family: JetBrains Mono; color: #64748b;">#${t.ticket}</td>
            <td style="font-weight: 600;">${t.strategy}</td>
            <td style="font-family: JetBrains Mono; color: #3b82f6;">${t.pair}</td>
            <td style="color:${t.type === 'BUY' ? '#10b981' : '#f43f5e'}; font-weight:700;">${t.type}</td>
            <td style="color: #f59e0b; font-family: JetBrains Mono; font-weight:700;">${t.lot} L</td>
            <td style="font-family: JetBrains Mono;">${t.open_price}</td>
            <td style="font-family: JetBrains Mono;">${t.close_price}</td>
            <td class="val-green">+${t.pips} pips</td>
            <td class="val-green">+$${t.pnl.toFixed(2)}</td>
            <td><span class="tag-confidence">${t.status}</span></td>
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

document.addEventListener("DOMContentLoaded", initTerminal);
