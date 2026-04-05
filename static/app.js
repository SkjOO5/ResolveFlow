document.addEventListener('DOMContentLoaded', () => {
    const taskSelect = document.getElementById('task-select');
    const resetBtn = document.getElementById('reset-btn');
    const stepBtn = document.getElementById('step-btn');
    const actionType = document.getElementById('action-type');
    const actionPayload = document.getElementById('action-payload');
    const errorMsg = document.getElementById('action-error');

    // UI Elements
    const els = {
        diffBadge: document.getElementById('diff-badge'),
        stepBadge: document.getElementById('step-badge'),
        obsMessage: document.getElementById('obs-message'),
        obsMeta: document.getElementById('obs-meta'),
        contextContainer: document.getElementById('context-container'),
        obsContext: document.getElementById('obs-context'),
        totalReward: document.getElementById('total-reward'),
        lastReward: document.getElementById('last-reward'),
        actionTimeline: document.getElementById('action-timeline'),
        donePanel: document.getElementById('done-panel'),
        finalScore: document.getElementById('final-score-val'),
        finalSummary: document.getElementById('final-summary')
    };

    let isDone = false;

    // Helper: generate payload hints
    actionType.addEventListener('change', (e) => {
        const type = e.target.value;
        let hint = "{}";
        if (type === "classify_ticket") hint = '{\n  "label": "damaged_item"\n}';
        if (type === "set_priority") hint = '{\n  "priority": "medium"\n}';
        if (type === "draft_response") hint = '{\n  "message": "Apology and solution..."\n}';
        if (type === "issue_refund") hint = '{\n  "amount": 24.99,\n  "reason": "..."\n}';
        if (type === "escalate_to_human") hint = '{\n  "team": "billing_ops"\n}';
        actionPayload.value = hint;
    });

    async function initEnv() {
        const loading = document.createElement('div');
        loading.textContent = "Connecting to env...";
        els.actionTimeline.innerHTML = "";
        els.actionTimeline.appendChild(loading);
        isDone = false;
        errorMsg.textContent = "";
        els.donePanel.style.display = 'none';

        try {
            const res = await fetch(`/api/reset?task_id=${taskSelect.value}`, { method: 'POST' });
            const obs = await res.json();
            
            const stateRes = await fetch('/api/state');
            const state = await stateRes.json();
            
            renderState(obs, state);
            els.actionTimeline.innerHTML = '<div class="empty-state">Environment reset. Awaiting action.</div>';
        } catch (e) {
            errorMsg.textContent = "Failed to reach backend.";
        }
    }

    async function stepEnv() {
        if (isDone) return;
        errorMsg.textContent = "";
        stepBtn.disabled = true;
        
        let payload = {};
        try {
             payload = JSON.parse(actionPayload.value);
        } catch (e) {
             errorMsg.textContent = "Payload must be valid JSON.";
             stepBtn.disabled = false;
             return;
        }

        const action = {
            action_type: actionType.value,
            payload: payload
        };

        try {
            const res = await fetch('/api/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(action)
            });

            if (!res.ok) {
                const err = await res.json();
                errorMsg.textContent = `Error: ${err.detail || 'Bad request'}`;
                stepBtn.disabled = false;
                return;
            }

            const stepRes = await res.json();
            
            const stateRes = await fetch('/api/state');
            const state = await stateRes.json();
            
            renderState(stepRes.observation, state);
            appendTimeline(action, stepRes.reward);
            
        } catch(e) {
            errorMsg.textContent = "Network error executing step.";
        }
        stepBtn.disabled = false;
    }

    function renderState(obs, state) {
        els.diffBadge.textContent = `Difficulty: ${obs.difficulty}`;
        els.stepBadge.textContent = `Step: ${obs.step_count} / ${obs.max_steps}`;
        els.obsMessage.textContent = obs.customer_message || "-";
        els.obsMeta.textContent = JSON.stringify(obs.metadata, null, 2);
        
        if (Object.keys(obs.revealed_context).length > 0) {
            els.contextContainer.style.display = 'block';
            els.obsContext.textContent = JSON.stringify(obs.revealed_context, null, 2);
        } else {
            els.contextContainer.style.display = 'none';
        }

        els.totalReward.textContent = state.cumulative_reward.toFixed(2);
        
        if (state.last_reward) {
            const val = state.last_reward.value;
            els.lastReward.textContent = (val >= 0 ? '+' : '') + val.toFixed(2);
            els.lastReward.style.color = val >= 0 ? 'var(--success)' : 'var(--danger)';
        }

        isDone = state.done;
        
        if (isDone) {
            els.donePanel.style.display = 'block';
            els.finalScore.textContent = state.final_score !== null ? (state.final_score * 100).toFixed(0) + '%' : 'Crashed';
            els.finalSummary.textContent = state.terminal_summary || "Max steps reached or terminal action executed.";
            stepBtn.disabled = true;
            stepBtn.textContent = "Episode Done";
        } else {
            stepBtn.textContent = "Execute Action";
        }
    }

    function appendTimeline(action, reward) {
        if (els.actionTimeline.querySelector('.empty-state')) {
            els.actionTimeline.innerHTML = '';
        }
        
        const item = document.createElement('div');
        item.className = 'tl-item';
        
        item.innerHTML = `
            <div class="tl-header">
                <span class="tl-action">${action.action_type}</span>
                <span class="tl-reward" style="color: ${reward.value >= 0 ? 'var(--success)' : 'var(--danger)'}">
                    ${reward.value > 0 ? '+' : ''}${reward.value.toFixed(2)}
                </span>
            </div>
            <div class="tl-reason">${reward.reason}</div>
        `;
        
        // Insert at top
        els.actionTimeline.insertBefore(item, els.actionTimeline.firstChild);
    }

    resetBtn.addEventListener('click', initEnv);
    stepBtn.addEventListener('click', stepEnv);

    // Init on load
    initEnv();
});
