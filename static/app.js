document.addEventListener("DOMContentLoaded", () => {
    const btnReset = document.getElementById("btn-reset");
    const btnStep = document.getElementById("btn-step");
    const taskSelector = document.getElementById("task-selector");
    const actionPanel = document.getElementById("action-panel");
    const resultPanel = document.getElementById("result-panel");
    
    // UI Elements
    const uiTask = document.getElementById("ui-current-task");
    const uiStep = document.getElementById("ui-step-count");
    const uiReward = document.getElementById("ui-reward");
    const uiStatus = document.getElementById("ui-status");
    const uiMessage = document.getElementById("ui-customer-msg");
    const uiMetaGrid = document.getElementById("ui-metadata-grid");
    const uiContext = document.getElementById("ui-revealed-context");
    const uiTimeline = document.getElementById("ui-timeline");
    const uiScore = document.getElementById("ui-final-score");
    const uiTerminalSummary = document.getElementById("ui-terminal-summary");
    const actionFeedback = document.getElementById("action-feedback");
    
    const actionType = document.getElementById("action-type");
    const actionPayload = document.getElementById("action-payload");

    // Action templates for quick UX
    const payloads = {
        "classify_ticket": '{"label": "damaged_item"}',
        "set_priority": '{"priority": "high"}',
        "request_account_details": '{}',
        "request_order_history": '{}',
        "request_shipping_status": '{}',
        "request_return_policy": '{}',
        "draft_response": '{"message": "We apologize, we will issue a full refund."}',
        "issue_refund": '{"amount": 25.00, "reason": "damaged"}',
        "offer_replacement": '{}',
        "escalate_to_human": '{"team": "billing_ops"}',
        "close_ticket": '{}'
    };

    actionType.addEventListener("change", (e) => {
        actionPayload.value = payloads[e.target.value] || "{}";
    });

    btnReset.addEventListener("click", async () => {
        actionFeedback.textContent = "";
        const taskId = taskSelector.value;
        const url = taskId ? `/api/reset?task_id=${taskId}` : "/api/reset";
        
        try {
            const res = await fetch(url, { method: "POST" });
            const obs = await res.json();
            updateUIfromObservation(obs);
            
            // clear timeline
            uiTimeline.innerHTML = "";
            uiReward.textContent = "0.00";
            uiStatus.textContent = "In Progress";
            uiStatus.style.color = "var(--primary)";
            
            actionPanel.classList.remove("disabled");
            resultPanel.classList.add("hidden");
        } catch (e) {
            console.error(e);
            alert("Reset failed");
        }
    });

    btnStep.addEventListener("click", async () => {
        actionFeedback.textContent = "";
        actionFeedback.className = "feedback-msg";
        
        let payloadObj = {};
        try {
            payloadObj = JSON.parse(actionPayload.value);
        } catch(e) {
            actionFeedback.textContent = "Invalid JSON in payload.";
            actionFeedback.classList.add("error");
            return;
        }

        const body = {
            action_type: actionType.value,
            payload: payloadObj
        };

        try {
            const res = await fetch("/api/step", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body)
            });
            
            if(!res.ok) {
                const err = await res.json();
                actionFeedback.textContent = "Error: " + err.detail;
                actionFeedback.classList.add("error");
                return;
            }
            
            const stepResult = await res.json();
            updateUIfromStep(stepResult, body);
        } catch (e) {
            console.error(e);
        }
    });

    function updateUIfromObservation(obs) {
        uiTask.textContent = `${obs.task_id} (${obs.difficulty})`;
        uiStep.textContent = `${obs.step_count} / ${obs.max_steps}`;
        uiMessage.textContent = obs.customer_message;
        
        uiMetaGrid.innerHTML = "";
        for(const [k, v] of Object.entries(obs.metadata)) {
            const span = document.createElement("span");
            span.className = "meta-tag";
            span.textContent = `${k}: ${v}`;
            uiMetaGrid.appendChild(span);
        }
        
        if (Object.keys(obs.revealed_context).length > 0) {
            uiContext.textContent = JSON.stringify(obs.revealed_context, null, 2);
        } else {
            uiContext.textContent = "No internal data retrieved yet.";
        }
    }

    function updateUIfromStep(stepRes, actionTaken) {
        updateUIfromObservation(stepRes.observation);
        
        // Add timeline item
        const div = document.createElement("div");
        div.className = "timeline-item";
        div.innerHTML = `
            <p class="tl-action">${actionTaken.action_type}</p>
            <p class="tl-info">Reward: ${stepRes.reward.value > 0 ? '+' : ''}${stepRes.reward.value.toFixed(2)} | Reason: ${stepRes.reward.reason}</p>
        `;
        uiTimeline.prepend(div);
        
        // empty state remove
        const empty = uiTimeline.querySelector(".empty-state");
        if(empty) empty.remove();
        
        // update reward
        const currentReward = parseFloat(uiReward.textContent);
        uiReward.textContent = (currentReward + stepRes.reward.value).toFixed(2);
        
        if (stepRes.done) {
            uiStatus.textContent = "Completed";
            uiStatus.style.color = "var(--success)";
            actionPanel.classList.add("disabled");
            
            if (stepRes.info && stepRes.info.final_score !== undefined) {
                resultPanel.classList.remove("hidden");
                uiScore.textContent = `Score: ${(stepRes.info.final_score * 100).toFixed(0)}%`;
                uiTerminalSummary.textContent = stepRes.info.terminal_summary;
            }
        }
    }
});
