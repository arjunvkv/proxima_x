// High-Performance Cyberpunk HUD & Lightweight Chart Controller for Proxima X Command Center

let ws = null;
let chart = null;
let candlestickSeries = null;
let currentLotMultiplier = 1.0;

function initDashboard() {
    console.log("⚡ Initializing Proxima X Cyberpunk Telemetry HUD & Interactive Chart...");
    initChart();
    connectWebSocket();
    startLocalTimer();
}

function initChart() {
    const chartElement = document.getElementById('chart-container');
    if (!chartElement) return;

    chart = LightweightCharts.createChart(chartElement, {
        layout: {
            background: { type: 'solid', color: '#090d16' },
            textColor: '#94a3b8',
        },
        grid: {
            vertLines: { color: 'rgba(0, 240, 255, 0.05)' },
            horzLines: { color: 'rgba(0, 240, 255, 0.05)' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        rightPriceScale: {
            borderColor: 'rgba(0, 240, 255, 0.2)',
        },
        timeScale: {
            borderColor: 'rgba(0, 240, 255, 0.2)',
            timeVisible: true,
            secondsVisible: false,
        },
    });

    candlestickSeries = chart.addCandlestickSeries({
        upColor: '#00ff88',
        downColor: '#ff0055',
        borderDownColor: '#ff0055',
        borderUpColor: '#00ff88',
        wickDownColor: '#ff0055',
        wickUpColor: '#00ff88',
    });

    // Sample M5 Candlestick data for GBPAUD
    const sampleCandles = [
        { time: 1785800000, open: 1.9140, high: 1.9160, low: 1.9135, close: 1.9155 },
        { time: 1785800300, open: 1.9155, high: 1.9175, low: 1.9150, close: 1.9168 },
        { time: 1785800600, open: 1.9168, high: 1.9189, low: 1.9160, close: 1.9185 },
        { time: 1785800900, open: 1.9185, high: 1.9210, low: 1.9180, close: 1.9205 },
        { time: 1785801200, open: 1.9205, high: 1.9225, low: 1.9200, close: 1.9220 },
    ];
    candlestickSeries.setData(sampleCandles);

    // Add Trade Entry & Exit Markers
    candlestickSeries.setMarkers([
        { time: 1785800300, position: 'belowBar', color: '#00ff88', shape: 'arrowUp', text: 'BUY ENTRY @ 1.9155' },
        { time: 1785801200, position: 'aboveBar', color: '#00ff88', shape: 'arrowDown', text: 'CLOSE TP +21.5p' }
    ]);
}

function updateLotProfile() {
    const select = document.getElementById("lot-profile-select");
    if (!select) return;
    currentLotMultiplier = parseFloat(select.value);
    console.log(`⚡ Lot Profile Updated: Multiplier = ${currentLotMultiplier}`);
    
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ lot_multiplier: currentLotMultiplier }));
    }
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
    
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log("🟢 WebSocket Telemetry Feed Connected!");
        document.getElementById("connection-status").innerText = "ONLINE";
        document.getElementById("connection-status").style.color = "#00ff88";
        ws.send(JSON.stringify({ lot_multiplier: currentLotMultiplier }));
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
            <div style="font-size: 0.72rem; color: #ffaa00; margin-bottom: 6px; font-weight: 600;">
                REGIME: ${p.regime}
            </div>
            <div class="card-metrics">
                <div class="metric-box">
                    <div class="metric-label">NEXT SYMBOL</div>
                    <div class="metric-value" style="color:#00f0ff;">${p.next_symbol}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">DIRECTION</div>
                    <div class="metric-value" style="color:${p.direction === 'BUY' || p.direction === 'LONG' ? '#00ff88' : '#ff0055'};">${p.direction}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">VPS LOT SIZE</div>
                    <div class="metric-value" style="color:#ffaa00;">${p.effective_lot} L</div>
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
            <td style="color: #ffaa00; font-family: Orbitron; font-weight:700;">${t.lot} L</td>
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
