/**
 * Lottery Hub v13.0 - Frontend Engine
 */

class LotteryApp {
    constructor() {
        this.currentMode = 'LOTTO'; 
        this.currentTab = 'accuracy';
        this.predictionData = null;
        this.charts = {};
        this.isAuthorized = false;

        // Bind DOM elements
        this.modeBtn = document.getElementById('mode-btn');
        this.navItems = document.querySelectorAll('.nav-item');
        this.tabs = document.querySelectorAll('.tab-content');
        this.best5Container = document.getElementById('best-5-container');
        this.expertNav = document.getElementById('expert-tabs-nav');
        this.expertContent = document.getElementById('expert-content');

        this.initLogin();
        this.init();
    }

    initLogin() {
        const overlay = document.getElementById('login-overlay');
        const passInput = document.getElementById('login-password');
        const submitBtn = document.getElementById('login-submit');
        const errorMsg = document.getElementById('login-error');

        overlay?.classList.remove('active');

        submitBtn?.addEventListener('click', async () => {
            errorMsg.style.display = 'none';
            submitBtn.disabled = true;
            try {
                const res = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ password: passInput.value })
                });
                if (!res.ok) throw new Error('login failed');
                this.isAuthorized = true;
                overlay.classList.remove('active');
                passInput.value = '';
                await this.refreshContent();
            } catch (e) {
                errorMsg.style.display = 'block';
                passInput.value = '';
            } finally {
                submitBtn.disabled = false;
            }
        });

        document.getElementById('login-back')?.addEventListener('click', () => {
            this.switchTab('accuracy');
        });

        passInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') submitBtn.click();
        });
    }

    showLoginGate() {
        const overlay = document.getElementById('login-overlay');
        overlay?.classList.add('active');
        const errorMsg = document.getElementById('login-error');
        if (errorMsg) errorMsg.style.display = 'none';
        const passInput = document.getElementById('login-password');
        if (passInput) passInput.value = '';
    }

    async init() {
        this.setupEventListeners();
        await this.checkAuth();
        await this.refreshContent();
        this.startStatusCheck();
    }

    async checkAuth() {
        try {
            const res = await fetch('/api/status');
            const status = await res.json();
            this.isAuthorized = !!status.authenticated;
        } catch (e) {
            this.isAuthorized = false;
        }
    }

    async apiFetch(path, options = {}) {
        const res = await fetch(path, options);
        if (res.status === 401) {
            this.isAuthorized = false;
            this.showLoginGate();
            throw new Error('unauthorized');
        }
        return res;
    }

    setupEventListeners() {
        this.modeBtn.addEventListener('click', () => this.toggleMode());
        this.navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const tabId = item.getAttribute('data-tab');
                this.switchTab(tabId);
            });
        });
        document.getElementById('evaluate-btn').addEventListener('click', () => this.runEvaluation());
        document.getElementById('backfill-btn')?.addEventListener('click', () => this.startAccuracyBackfill());
    }

    toggleMode() {
        this.currentMode = (this.currentMode === 'LOTTO') ? 'PENSION' : 'LOTTO';
        document.body.classList.toggle('pension-mode', this.currentMode === 'PENSION');
        document.querySelector('.app-container')?.classList.toggle('pension-mode', this.currentMode === 'PENSION');
        this.modeBtn.textContent = (this.currentMode === 'LOTTO') ? '모드: 로또 6/45' : '모드: 연금 720+';
        this.modeBtn.className = (this.currentMode === 'LOTTO') ? 'lotto-mode' : 'pension-mode';
        document.getElementById('lotto-input-group').classList.toggle('hidden', this.currentMode !== 'LOTTO');
        document.getElementById('pension-input-group').classList.toggle('hidden', this.currentMode !== 'PENSION');
        
        // Update dynamic primary color
        const primaryColor = this.currentMode === 'LOTTO' ? 'var(--accent-lotto)' : 'var(--accent-pension)';
        document.documentElement.style.setProperty('--primary', primaryColor);
        
        this.refreshContent();
    }

    switchTab(tabId) {
        if (tabId === 'accuracy') {
            document.getElementById('login-overlay')?.classList.remove('active');
        }
        if (tabId !== 'accuracy' && !this.isAuthorized) {
            this.showLoginGate();
            return;
        }
        this.currentTab = tabId;
        this.navItems.forEach(nav => nav.classList.toggle('active', nav.getAttribute('data-tab') === tabId));
        this.tabs.forEach(tab => tab.classList.toggle('active', tab.id === `${tabId}-tab`));
        if (tabId === 'analyze') this.refreshCharts();
        if (tabId === 'accuracy') this.refreshAccuracy();
    }

    async refreshContent() {
        await this.loadHistory();
        if (this.currentTab === 'accuracy') await this.refreshAccuracy();
        if (this.isAuthorized) {
            await this.loadPredictions();
            if (this.currentTab === 'analyze') await this.refreshCharts();
        }
    }

    async loadHistory() {
        const path = this.currentMode === 'LOTTO' ? '/api/history/lotto?count=8' : '/api/history/pension?count=8';
        try {
            const res = await fetch(path);
            const history = await res.json();
            const container = document.getElementById('latest-history');
            if (!container) return;
            container.innerHTML = '';
            history.forEach(item => {
                const div = document.createElement('div');
                div.className = 'history-item';
                const date = (item.draw_date || '').substring(0, 10);
                let val = '';
                if (this.currentMode === 'LOTTO') {
                    val = [item.win1, item.win2, item.win3, item.win4, item.win5, item.win6].join('-');
                } else {
                    const digits = [item.n1, item.n2, item.n3, item.n4, item.n5, item.n6].join('');
                    val = `${item.group_no}조 ${digits}`;
                }
                div.innerHTML = `<span style="font-weight:600;">${item.draw_no}회차</span> <span style="font-size:10px; color:#aaa; margin:0 8px;">(${date})</span><span style="font-weight:bold; color:var(--primary);">${val}</span>`;
                container.appendChild(div);
            });
        } catch (e) { console.error(e); }
    }

    async loadPredictions() {
        if (!this.isAuthorized) return;
        const path = this.currentMode === 'LOTTO' ? '/api/predict/lotto' : '/api/predict/pension';
        try {
            const res = await this.apiFetch(path);
            const data = await res.json();
            this.predictionData = data;
            this.renderPredictions();
        } catch (e) { console.error(e); }
    }

    renderPredictions() {
        if (!this.predictionData || !this.predictionData.best5) return;
        this.best5Container.innerHTML = '';
        this.predictionData.best5.forEach((set, idx) => {
            const card = document.createElement('div');
            card.className = 'best-card';
            card.innerHTML = `<div class="score-tag">${set.score.toFixed(1)}%</div><h3>추천 조합 #${idx + 1}</h3><div class="ball-row">${this.renderBalls(set)}</div><div class="subtitle">${set.logic || ''}</div>`;
            this.best5Container.appendChild(card);
        });
        // Expert Nav
        this.expertNav.innerHTML = '';
        Object.keys(this.predictionData.methods).forEach((m, idx) => {
            const btn = document.createElement('button');
            btn.className = `expert-tab-btn ${idx === 0 ? 'active' : ''}`;
            btn.textContent = m;
            btn.onclick = () => {
                document.querySelectorAll('.expert-tab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.renderExpertMethod(m);
            };
            this.expertNav.appendChild(btn);
        });
        this.renderExpertMethod(Object.keys(this.predictionData.methods)[0]);
    }

    renderExpertMethod(name) {
        const data = this.predictionData.methods[name];
        this.expertContent.innerHTML = `<div class="method-desc">${data.desc}</div><div class="expert-sets">${data.sets.map((set, i) => `<div class="ball-row"><span style="width:30px; font-weight:bold;">${i+1}</span>${this.renderBalls(set)}<span style="margin-left:10px; font-size:12px;">[${set.score.toFixed(1)}%]</span></div>`).join('')}</div>`;
    }

    renderBalls(set) {
        if (this.currentMode === 'LOTTO') {
            return (set.numbers || []).map(n => `<div class="ball" style="background:${this.getBallColor(n)}">${n}</div>`).join('');
        } else {
            return `<div class="pension-numbers"><div class="pension-group">${set.group}조</div>${(set.digits || []).map(d => `<div class="pension-digit">${d}</div>`).join('')}</div>`;
        }
    }

    async refreshCharts() {
        if (!this.isAuthorized) return;
        // Basic charts logic for both modes
        try {
            const path = this.currentMode === 'LOTTO' ? '/api/analyze/lotto' : '/api/analyze/pension';
            const res = await fetch(path);
            const data = await res.json();
            Object.keys(this.charts).forEach(id => this.charts[id]?.destroy());
            this.charts = {};
            if (this.currentMode === 'LOTTO') this.renderLottoCharts(data);
            else this.renderPensionCharts(data);
        } catch (e) { console.error(e); }
    }

    renderLottoCharts(data) {
        const labels45 = Array.from({length: 45}, (_, i) => i + 1);
        const drawLabels = (data.draw_nos || []).slice(-100);
        this.setChartTitle(1, '① 최근 100회 총합 추이');
        this.setChartTitle(2, '② 50회차 이동평균');
        this.setChartTitle(3, '③ 개별 번호 출현 빈도');
        this.setChartTitle(4, '④ 홀수 개수 분포');
        this.setChartTitle(5, '⑤ 연속 번호 쌍 분포');
        this.setChartTitle(6, '⑥ 번호대 색상 분포');
        this.setChartTitle(7, '⑦ 고번호 개수 분포');
        this.setChartTitle(8, '⑧ AC 값 분포');
        this.renderChart('chart-1', 'line', drawLabels, (data.sums || []).slice(-100), 'Sum', '#007bff');
        this.renderChart('chart-2', 'line', drawLabels.slice(-(data.rolling_sums || []).length), data.rolling_sums || [], 'Rolling Sum', '#0097a7');
        this.renderChart('chart-3', 'bar', labels45, data.freqs || [], 'Freq', '#ffb100');
        this.renderChart('chart-4', 'bar', ['0','1','2','3','4','5','6'], data.odd_counts || [], 'Odd Count', '#e74c3c');
        this.renderChart('chart-5', 'bar', ['0','1','2','3','4','5'], data.consec_counts || [], 'Consecutive', '#2ecc71');
        this.renderChart('chart-6', 'bar', ['1-10','11-20','21-30','31-40','41-45'], data.colors || [], 'Band', '#95a5a6');
        this.renderChart('chart-7', 'bar', ['0','1','2','3','4','5','6'], data.high_counts || [], 'High', '#8e44ad');
        this.renderChart('chart-8', 'bar', Array.from({length: 11}, (_, i) => i), data.ac_counts || [], 'AC', '#34495e');
    }

    renderPensionCharts(data) {
        const digitLabels = Array.from({length: 10}, (_, i) => i);
        const drawLabels = (data.draw_nos || []).slice(-100);
        this.setChartTitle(1, '① 최근 100회 자릿수 합계 추이');
        this.setChartTitle(2, '② 50회차 이동평균');
        this.setChartTitle(3, '③ 전체 숫자 출현 빈도');
        this.setChartTitle(4, '④ 홀수 개수 분포');
        this.setChartTitle(5, '⑤ 동일 연속 숫자 분포');
        this.setChartTitle(6, '⑥ 첫째 자리 숫자 빈도');
        this.setChartTitle(7, '⑦ 고숫자 개수 분포');
        this.setChartTitle(8, '⑧ 마지막 자리 숫자 빈도');
        this.renderChart('chart-1', 'line', drawLabels, (data.sums || []).slice(-100), 'Digit Sum', '#007bff');
        this.renderChart('chart-2', 'line', drawLabels.slice(-(data.rolling_sums || []).length), data.rolling_sums || [], 'Rolling Sum', '#0097a7');
        this.renderChart('chart-3', 'bar', digitLabels, data.digit_freqs || [], 'Digit Freq', '#ffb100');
        this.renderChart('chart-4', 'bar', ['0','1','2','3','4','5','6'], data.odd_counts || [], 'Odd Count', '#e74c3c');
        this.renderChart('chart-5', 'bar', ['0','1','2','3','4','5'], data.identical_counts || [], 'Identical', '#2ecc71');
        this.renderChart('chart-6', 'bar', digitLabels, data.pos1_freqs || [], 'D1 Freq', '#95a5a6');
        this.renderChart('chart-7', 'bar', ['0','1','2','3','4','5','6'], data.high_counts || [], 'High', '#8e44ad');
        this.renderChart('chart-8', 'bar', digitLabels, data.pos6_freqs || [], 'D6 Freq', '#34495e');
    }

    setChartTitle(index, title) {
        const el = document.getElementById(`c-title-${index}`);
        if (el) el.textContent = title;
    }

    renderChart(canvasId, type, labels, data, label, color) {
        if (this.charts[canvasId]) this.charts[canvasId].destroy();
        const ctx = document.getElementById(canvasId)?.getContext('2d');
        if (!ctx) return;
        this.charts[canvasId] = new Chart(ctx, {
            type: type,
            data: { labels: labels, datasets: [{ label: label, data: data, backgroundColor: color + '55', borderColor: color, borderWidth: 2, tension: 0.4, pointRadius: 0 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });
    }

    async runEvaluation() {
        if (!this.isAuthorized) return;
        const resultCard = document.getElementById('eval-result');
        resultCard.classList.remove('hidden');
        resultCard.innerHTML = 'Evaluating...';
        try {
            let body = this.currentMode === 'LOTTO' ? { numbers: Array.from(document.querySelectorAll('#lotto-input-group input')).map(i => parseInt(i.value)) } : { group: parseInt(document.querySelector('#pension-input-group input').value), digits: Array.from(document.querySelectorAll('#pension-input-group input')).slice(1).map(i => parseInt(i.value)) };
            const endpoint = this.currentMode === 'LOTTO' ? '/api/evaluate/lotto' : '/api/evaluate/pension';
            const res = await this.apiFetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'evaluation failed');
            resultCard.innerHTML = `<div class="grade-badge">${data.grade}</div><div class="score-display">${data.score}%</div><div class="details-text">${data.details}</div>`;
        } catch (e) { resultCard.innerHTML = e.message === 'unauthorized' ? '' : `Error: ${e.message}`; }
    }

    async refreshAccuracy() {
        try {
            const res = await fetch(`/api/accuracy?mode=${this.currentMode}`);
            const data = await res.json();
            this.updateLeaderboard(data);
            await this.updateAccuracySummary(data);
            const tbody = document.getElementById('accuracy-tbody');
            if (!tbody) return;
            tbody.innerHTML = '';
            data.slice().reverse().forEach(r => {
                const tr = document.createElement('tr');
                tr.className = 'clickable-row';
                const actual = r.actual_nums || [];
                const ensembleHits = r.ensemble_hits || [];
                const best = r.best_1st || 0;
                let actualHtml = '', resultMsg = '', hitDetail = '';
                
                if (this.currentMode === 'LOTTO') {
                    actualHtml = `<div class="mini-balls">${actual.map(n => `<div class="mini-ball hit" style="background:${this.getBallColor(n)}">${n}</div>`).join('')}</div>`;
                    resultMsg = `<span style="color:${best>=3?'#ef4444':'#94a3b8'}; font-weight:800;">최고 ${best}개 (${this.getLottoRankLabel(best)})</span>`;
                    hitDetail = `[ ${ensembleHits.join(', ')} ]`;
                } else {
                    const a_1st = actual[0] || [];
                    actualHtml = `<div>1등: ${a_1st[0]}조 ${a_1st.slice(1).join('')}</div>`;
                    const b_val = (r.best_bonus || 0);
                    resultMsg = `<div class="rank-badge-pension ${best>=1?'rank-pension-win':'rank-pension-none'}">${this.getPensionRankLabel(best)}</div>`;
                    if (b_val >= 1) {
                        resultMsg += `<div class="rank-badge-pension rank-pension-bonus-hit">보너스 ${b_val}점</div>`;
                    }
                    hitDetail = `티켓별: [${ensembleHits.join(', ')}]`;
                }

                tr.innerHTML = `<td>#${r.draw_no}</td><td>${r.draw_date}</td><td>${actualHtml}</td><td>${resultMsg}</td><td>${hitDetail}</td><td style="font-size:10px;">1~${r.draw_no-1}회 학습</td>`;
                
                const dTr = document.createElement('tr');
                dTr.className = 'method-details-row hidden';
                dTr.id = `detail-${r.draw_no}`;
                let grid = '<td colspan="6" class="method-details-content"><div class="method-detail-grid">';
                if (r.method_results) {
                    Object.entries(r.method_results).forEach(([m, s]) => {
                        const hitsArray = s.hits_1st || s.hits || [];
                        const tickets = (s.sets_generated || []).map((t, i) => {
                            const str = this.currentMode === 'LOTTO' ? t.join(', ') : `${t.group}조 ${t.digits.join('')}`;
                            const hitCount = hitsArray[i] !== undefined ? hitsArray[i] : 0;
                            return `<div style="font-size:10px; color:#666;">${str} (${hitCount}${this.currentMode==='LOTTO'?'개':'점'})</div>`;
                        }).join('');
                        grid += `<div class="method-stat-box"><div class="m-name">${m}</div>${tickets}</div>`;
                    });
                }
                dTr.innerHTML = grid + '</div></td>';
                tr.onclick = () => {
                    const target = document.getElementById(`detail-${r.draw_no}`);
                    const isHidden = target.classList.contains('hidden');
                    document.querySelectorAll('.method-details-row').forEach(row => row.classList.add('hidden'));
                    if (isHidden) target.classList.remove('hidden');
                };
                tbody.appendChild(tr);
                tbody.appendChild(dTr);
            });
        } catch (e) { console.error(e); }
    }

    updateLeaderboard(data) {
        const scores = {};
        data.forEach(r => {
            if (r.method_results) {
                Object.entries(r.method_results).forEach(([m, s]) => {
                    let p = 0, b = s.best_1st || s.best_hit || 0;
                    if (this.currentMode === 'LOTTO') { 
                        if(b>=3) p+=1; 
                        if(b>=4) p+=3; 
                        if(b>=5) p+=10; 
                    }
                    else { p = b + (s.best_bonus || 0); }
                    scores[m] = (scores[m] || 0) + p;
                });
            }
        });
        const sorted = Object.entries(scores).sort((a,b) => b[1]-a[1]);
        const container = document.getElementById('leaderboard-container');
        if (container && sorted.length > 0) {
            container.innerHTML = `<div class="leaderboard-header">✨ 추천 알고리즘 순위</div><div class="leaderboard-list">` + 
                sorted.slice(0, 3).map((m, i) => `<div class="leaderboard-row"><div class="lb-rank">#${i+1}</div><div class="lb-name">${m[0]}</div><div class="lb-metrics">${m[1]}점</div></div>`).join('') + `</div>`;
        }
    }

    async updateAccuracySummary(data) {
        const container = document.getElementById('accuracy-summary');
        if (!container) return;
        try {
            const res = await fetch(`/api/accuracy/coverage?mode=${this.currentMode}`);
            const coverage = await res.json();
            const bestCount = data.filter(r => (r.best_1st || 0) >= (this.currentMode === 'LOTTO' ? 3 : 1)).length;
            container.innerHTML = `
                <div class="summary-item"><span>데이터</span><strong>${coverage.source_min}~${coverage.source_max}회</strong></div>
                <div class="summary-item"><span>계산완료</span><strong>${coverage.computed_draws}/${coverage.total_draws}회</strong></div>
                <div class="summary-item"><span>커버리지</span><strong>${coverage.coverage_pct}%</strong></div>
                <div class="summary-item"><span>유효 성과</span><strong>${bestCount}건</strong></div>
                <div class="summary-actions">
                    <input id="backfill-count" class="summary-count-input" type="number" min="1" max="25" value="5" ${this.isAuthorized && coverage.missing_draws > 0 ? '' : 'disabled'}>
                    <button id="backfill-btn" class="summary-action-btn" ${this.isAuthorized && coverage.missing_draws > 0 ? '' : 'disabled'}>백필</button>
                    <span id="backfill-status">미계산 ${coverage.missing_draws}회</span>
                </div>
            `;
            document.getElementById('backfill-btn')?.addEventListener('click', () => this.startAccuracyBackfill());
            if (this.isAuthorized) await this.refreshBackfillStatus();
        } catch (e) {
            container.textContent = `정확도 기록 ${data.length}건`;
        }
    }

    async refreshBackfillStatus() {
        const statusEl = document.getElementById('backfill-status');
        if (!statusEl) return;
        try {
            const res = await this.apiFetch(`/api/accuracy/backfill/status?mode=${this.currentMode}`);
            const status = await res.json();
            statusEl.textContent = status.running
                ? `${status.current}/${status.total} 진행 중`
                : status.message;
        } catch (e) {
            if (e.message !== 'unauthorized') statusEl.textContent = '상태 확인 실패';
        }
    }

    async startAccuracyBackfill() {
        const btn = document.getElementById('backfill-btn');
        const statusEl = document.getElementById('backfill-status');
        if (btn) btn.disabled = true;
        if (statusEl) statusEl.textContent = '시작 중...';
        try {
            const countInput = document.getElementById('backfill-count');
            const count = Math.max(1, Math.min(parseInt(countInput?.value || '5', 10), 25));
            const res = await this.apiFetch('/api/accuracy/backfill/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: this.currentMode, count })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'backfill failed');
            if (statusEl) statusEl.textContent = data.status?.message || '백필 시작됨';
            this.pollBackfillStatus();
        } catch (e) {
            if (statusEl) statusEl.textContent = e.message === 'unauthorized' ? '' : `오류: ${e.message}`;
            if (btn) btn.disabled = false;
        }
    }

    pollBackfillStatus() {
        const tick = async () => {
            await this.refreshBackfillStatus();
            const statusEl = document.getElementById('backfill-status');
            if (statusEl && statusEl.textContent.includes('진행 중')) {
                setTimeout(tick, 5000);
            } else {
                await this.refreshAccuracy();
            }
        };
        setTimeout(tick, 5000);
    }

    getPensionRankLabel(h) { if(h>=7) return "1등!!"; if(h>=6) return "2등"; if(h>=1) return `${8-h}등`; return "낙첨"; }
    getLottoRankLabel(h) { if(h===6) return "1등"; if(h===5) return "3등"; if(h===4) return "4등"; if(h===3) return "5등"; return "낙첨"; }
    getBallColor(n) { if(n<=10) return '#f1c40f'; if(n<=20) return '#3498db'; if(n<=30) return '#e74c3c'; if(n<=40) return '#95a5a6'; return '#2ecc71'; }
    startStatusCheck() { setInterval(() => this.loadHistory(), 60000); }
}
window.onload = () => { window.app = new LotteryApp(); };
