const adminStatus = document.getElementById("adminStatus");
const adminTableBody = document.getElementById("adminTableBody");
const params = new URLSearchParams(window.location.search);
const adminToken = params.get("token") || "";

function renderRows(items) {
  adminTableBody.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${item.created_at || "-"}</td>
      <td>${item.question || "-"}</td>
      <td>${item.success ? "yes" : "no"}</td>
      <td>${item.selected_from || "-"}</td>
      <td>${item.fallback_triggered ? "yes" : "no"}</td>
      <td>${item.total_latency_ms ?? "-"} ms</td>
      <td>${item.session_id || "-"}</td>
      <td>${item.request_id || "-"}</td>
      <td>${item.feedback_rating || "-"}</td>
      <td>${item.feedback_comment || "-"}</td>
    `;
    adminTableBody.appendChild(row);
  });
}

async function loadAdminRequests() {
  if (!adminToken) {
    adminStatus.textContent = "missing token";
    return;
  }
  try {
    const response = await fetch(`/api/admin/requests?token=${encodeURIComponent(adminToken)}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "admin request failed");
    }
    renderRows(data.items || []);
    adminStatus.textContent = `${(data.items || []).length} rows`;
  } catch (error) {
    adminStatus.textContent = `error: ${error.message}`;
  }
}

loadAdminRequests();
