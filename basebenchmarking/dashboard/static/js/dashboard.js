/**
 * Dashboard Logic for Image Registration Benchmark Platform
 */

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    loadOverview();
    loadLeaderboard();
    loadMethodsStatus();
    loadPairsDropdown();
    loadISROComparison();
});

// Navigation tabs
function initNavigation() {
    const navButtons = document.querySelectorAll(".nav-btn");
    const sections = document.querySelectorAll(".tab-section");

    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            navButtons.forEach(b => b.classList.remove("active"));
            sections.forEach(s => s.classList.add("hidden"));

            btn.classList.add("active");
            const targetId = btn.getAttribute("data-target");
            const targetSection = document.getElementById(targetId);
            if (targetSection) {
                targetSection.classList.remove("hidden");
                targetSection.classList.add("active");
            }
        });
    });
}

// Load Overview KPIs
async function loadOverview() {
    try {
        const res = await fetch("/api/overview");
        const data = await res.json();

        document.getElementById("kpi-dataset-size").textContent = data.dataset_size || 0;
        document.getElementById("kpi-total-methods").textContent = data.total_methods || 6;

        const banner = document.getElementById("empty-state-banner");
        if (!data.has_data) {
            banner.classList.remove("hidden");
            document.getElementById("kpi-best-method").textContent = "N/A";
            document.getElementById("kpi-best-rmse").textContent = "N/A";
        } else {
            banner.classList.add("hidden");
            document.getElementById("kpi-best-method").textContent = data.best_method || "N/A";
            document.getElementById("kpi-best-rmse").textContent = data.best_rmse ? `${data.best_rmse} px` : "N/A";
            document.getElementById("kpi-best-detail").textContent = `Success Rate: ${(data.best_success_rate * 100).toFixed(1)}%`;
        }
    } catch (err) {
        console.error("Failed to load overview:", err);
    }
}

// Load Registered Methods Status
async function loadMethodsStatus() {
    try {
        const res = await fetch("/api/methods");
        const methods = await res.json();
        const grid = document.getElementById("methods-status-grid");
        grid.innerHTML = "";

        methods.forEach(m => {
            const card = document.createElement("div");
            card.className = "method-status-card";
            const tagClass = m.available ? "available" : "unavailable";
            const tagText = m.available ? "Available" : "Missing Dependency";

            card.innerHTML = `
                <div>
                    <strong>${m.name}</strong>
                    <div style="font-size:0.75rem; color:var(--text-muted); margin-top:2px;">${m.reason}</div>
                </div>
                <span class="status-tag ${tagClass}">${tagText}</span>
            `;
            grid.appendChild(card);
        });
    } catch (err) {
        console.error("Failed to load methods:", err);
    }
}

// Load Leaderboard Table and render Chart.js
async function loadLeaderboard() {
    try {
        const res = await fetch("/api/leaderboard");
        const data = await res.json();
        const tbody = document.getElementById("leaderboard-tbody");
        tbody.innerHTML = "";

        if (!data.has_data || !data.leaderboard || data.leaderboard.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">No benchmark results available. Add image pairs to data/pairs/ and run pipeline.runner.</td></tr>`;
            return;
        }

        const methodsList = [];
        const successRates = [];
        const runtimes = [];

        data.leaderboard.forEach(row => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${row.method}</strong></td>
                <td><span style="color:var(--primary-cyan); font-weight:600;">${(row.success_rate * 100).toFixed(1)}%</span></td>
                <td>${row.mean_rmse !== null ? row.mean_rmse.toFixed(4) : "N/A (No GT)"}</td>
                <td>${row.median_rmse !== null ? row.median_rmse.toFixed(4) : "N/A"}</td>
                <td>${row.mean_inlier_ratio !== null ? (row.mean_inlier_ratio * 100).toFixed(1) + "%" : "N/A"}</td>
                <td>${row.mean_runtime_ms !== null ? row.mean_runtime_ms.toFixed(1) : "N/A"}</td>
                <td>${row.failed_pairs}</td>
            `;
            tbody.appendChild(tr);

            methodsList.push(row.method);
            successRates.push((row.success_rate * 100).toFixed(1));
            runtimes.push(row.mean_runtime_ms ? row.mean_runtime_ms.toFixed(1) : 0);
        });

        renderCharts(methodsList, successRates, runtimes);

    } catch (err) {
        console.error("Failed to load leaderboard:", err);
    }
}

// Render Chart.js
function renderCharts(methods, successRates, runtimes) {
    const ctxSuccess = document.getElementById("chart-success-rate");
    const ctxRuntime = document.getElementById("chart-runtime");

    if (ctxSuccess) {
        new Chart(ctxSuccess, {
            type: "bar",
            data: {
                labels: methods,
                datasets: [{
                    label: "Success Rate (%)",
                    data: successRates,
                    backgroundColor: "rgba(0, 212, 255, 0.6)",
                    borderColor: "#00d4ff",
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true, max: 100 } }
            }
        });
    }

    if (ctxRuntime) {
        new Chart(ctxRuntime, {
            type: "bar",
            data: {
                labels: methods,
                datasets: [{
                    label: "Mean Runtime (ms)",
                    data: runtimes,
                    backgroundColor: "rgba(255, 0, 110, 0.6)",
                    borderColor: "#ff006e",
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true } }
            }
        });
    }
}

// Populate Pairs Dropdown for Inspector
async function loadPairsDropdown() {
    try {
        const res = await fetch("/api/pairs");
        const pairs = await res.json();
        const select = document.getElementById("pair-select");
        select.innerHTML = "";

        if (pairs.length === 0) {
            select.innerHTML = `<option value="">-- No Pairs Discovered --</option>`;
            return;
        }

        select.innerHTML = `<option value="">-- Select a Pair --</option>`;
        pairs.forEach(p => {
            const opt = document.createElement("option");
            opt.value = p.pair_id;
            opt.textContent = `${p.pair_id} ${p.has_gt ? '(Has Ground Truth)' : ''}`;
            select.appendChild(opt);
        });

        select.addEventListener("change", (e) => {
            const pairId = e.target.value;
            if (pairId) loadPairInspector(pairId);
            else document.getElementById("pair-details-container").classList.add("hidden");
        });
    } catch (err) {
        console.error("Failed to load pairs:", err);
    }
}

// Render Pair Inspector View
async function loadPairInspector(pairId) {
    const container = document.getElementById("pair-details-container");
    const title = document.getElementById("selected-pair-title");
    const grid = document.getElementById("pair-methods-grid");

    title.textContent = `Inspection for Pair: ${pairId}`;
    grid.innerHTML = "<p class='text-muted'>Loading results...</p>";
    container.classList.remove("hidden");

    try {
        const res = await fetch(`/api/pair/${pairId}`);
        const data = await res.json();
        grid.innerHTML = "";

        if (!data.results || data.results.length === 0) {
            grid.innerHTML = "<p class='text-muted'>No method results found for this pair.</p>";
            return;
        }

        data.results.forEach(r => {
            const card = document.createElement("div");
            card.className = "card";
            card.style.marginBottom = "0";

            const successTag = r.success 
                ? `<span class="status-tag available">Success</span>`
                : `<span class="status-tag unavailable">Failed</span>`;

            card.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <h4>${r.method}</h4>
                    ${successTag}
                </div>
                <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:10px;">
                    Inliers: <strong>${r.num_inliers || 0}</strong> | 
                    Inlier Ratio: <strong>${r.inlier_ratio !== null ? (r.inlier_ratio * 100).toFixed(1) + '%' : 'N/A'}</strong> | 
                    Runtime: <strong>${r.runtime_ms} ms</strong>
                </p>
                ${r.rmse !== null ? `<p style="font-size:0.85rem; color:var(--primary-cyan); margin-bottom:10px;">RMSE: <strong>${r.rmse} px</strong></p>` : ''}
                <div style="margin-top:10px;">
                    <div style="font-size:0.8rem; font-weight:600; margin-bottom:4px;">Matches Line Overlay:</div>
                    <img src="${r.matches_vis_url}" class="vis-image" alt="Matches Visual" onerror="this.style.display='none'">
                </div>
            `;
            grid.appendChild(card);
        });

    } catch (err) {
        console.error("Failed to load pair inspection:", err);
    }
}

// Load ISRO Paper Comparison Table
async function loadISROComparison() {
    try {
        const res = await fetch("/api/isro-comparison");
        const data = await res.json();
        const tbody = document.getElementById("isro-table-body");
        tbody.innerHTML = "";

        if (!data.table || data.table.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">No comparison data available.</td></tr>`;
            return;
        }

        data.table.forEach(r => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${r.dataset}</strong></td>
                <td>${r.method}</td>
                <td>${r.isro_status}</td>
                <td>${r.isro_rmse_x !== null ? r.isro_rmse_x : 'NA'}</td>
                <td>${r.our_rmse_x !== null ? r.our_rmse_x : 'N/A'}</td>
                <td>${r.isro_time_sec !== null ? r.isro_time_sec : 'NA'}</td>
                <td>${r.our_time_sec !== null ? r.our_time_sec : 'N/A'}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error("Failed to load ISRO comparison:", err);
    }
}
