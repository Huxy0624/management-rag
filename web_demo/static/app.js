const questionInput = document.getElementById("questionInput");
const askButton = document.getElementById("askButton");
const requestStatus = document.getElementById("requestStatus");
const answerText = document.getElementById("answerText");
const sourcesList = document.getElementById("sourcesList");
const requestMeta = document.getElementById("requestMeta");
const selectorMeta = document.getElementById("selectorMeta");
const diagnosisMeta = document.getElementById("diagnosisMeta");
const contractMeta = document.getElementById("contractMeta");
const plannerPre = document.getElementById("plannerPre");
const rewriteTracePre = document.getElementById("rewriteTracePre");
const debugPre = document.getElementById("debugPre");
const timingsGrid = document.getElementById("timingsGrid");
const exampleButtons = document.getElementById("exampleButtons");
const healthBadge = document.getElementById("healthBadge");
const debugPanel = document.getElementById("debugPanel");
const demoTitle = document.getElementById("demoTitle");
const demoTagline = document.getElementById("demoTagline");
const demoDescription = document.getElementById("demoDescription");
const answerTitle = document.getElementById("answerTitle");
const feedbackPrompt = document.getElementById("feedbackPrompt");
const debugTitle = document.getElementById("debugTitle");
const debugDescription = document.getElementById("debugDescription");
const feedbackBox = document.getElementById("feedbackBox");
const thumbUpButton = document.getElementById("thumbUpButton");
const thumbDownButton = document.getElementById("thumbDownButton");
const feedbackComment = document.getElementById("feedbackComment");
const submitFeedbackButton = document.getElementById("submitFeedbackButton");
const feedbackStatus = document.getElementById("feedbackStatus");

const urlParams = new URLSearchParams(window.location.search);
const debugRequested = urlParams.get("debug") === "1";
const debugToken = urlParams.get("token") || "";
let lastRequestId = null;
let lastSessionId = null;
let selectedRating = "up";
let configPayload = null;

function renderExamples(examples) {
  exampleButtons.innerHTML = "";
  (examples || []).forEach((example) => {
    const button = document.createElement("button");
    button.className = "example-chip";
    button.textContent = example;
    button.addEventListener("click", () => {
      questionInput.value = example;
      questionInput.focus();
    });
    exampleButtons.appendChild(button);
  });
}

function setSelectedRating(rating) {
  selectedRating = rating;
  thumbUpButton.classList.toggle("muted", rating !== "up");
  thumbDownButton.classList.toggle("muted", rating !== "down");
}

function renderConfig(config) {
  configPayload = config;
  demoTitle.textContent = config.title;
  demoTagline.textContent = config.tagline;
  demoDescription.textContent = config.description;
  questionInput.placeholder = config.input_placeholder;
  answerTitle.textContent = config.answer_title;
  feedbackPrompt.textContent = config.feedback_prompt;
  debugTitle.textContent = config.debug_title;
  debugDescription.textContent = config.debug_description;
  renderExamples(config.examples);
  debugPanel.classList.toggle("hidden", !config.debug_mode_enabled);
}

function renderExamplesFallback() {
  renderExamples([
    "怎么临时推进跨部门合作？",
    "怎么通过机制解决跨部门协作？",
    "老板问进度时，向上汇报应该怎么压缩信息？",
    "战略传到一线时，向下解释应该怎么做？",
    "为什么跨部门协作经常低效？"
  ]);
}

function renderFeedbackVisible() {
  feedbackBox.classList.remove("hidden");
  feedbackStatus.textContent = "";
  feedbackComment.value = "";
  setSelectedRating("up");
}

function hideFeedback() {
  feedbackBox.classList.add("hidden");
  feedbackStatus.textContent = "";
}

function renderKeyValues(container, pairs) {
  container.innerHTML = "";
  pairs.forEach(([key, value]) => {
    const row = document.createElement("div");
    row.className = container.id === "timingsGrid" ? "timing-row" : "kv-row";
    row.innerHTML = `
      <span class="${container.id === "timingsGrid" ? "timing-key" : "kv-key"}">${key}</span>
      <span class="${container.id === "timingsGrid" ? "timing-value" : "kv-value"}">${value ?? "-"}</span>
    `;
    container.appendChild(row);
  });
}

function renderSources(sources) {
  sourcesList.innerHTML = "";
  if (!sources || sources.length === 0) {
    sourcesList.textContent = "No retrieved chunks.";
    return;
  }

  sources.forEach((source) => {
    const item = document.createElement("div");
    item.className = "source-item";
    item.innerHTML = `
      <div class="source-top">
        <span>${source.source} + ${source.chunk_id}</span>
        <span>rerank=${source.rerank_score}</span>
      </div>
      <div class="source-preview">${source.preview || ""}</div>
    `;
    sourcesList.appendChild(item);
  });
}

function traceOrFallback(data) {
  const debug = data.debug_info || {};
  const trace = data.generation_trace || {};
  const diagnosis = trace.diagnosis || debug.question_diagnosis || {};
  const queryRewrites = trace.query_rewrites || {};
  const retrieval = trace.retrieval || {};
  const planner = trace.planner || debug.planner_output_v2 || {};
  const controlLayer = trace.control_layer || {};
  const failureLogging = trace.failure_logging || {};
  const finalOutput = trace.final_output || {};

  return {
    debug,
    trace,
    diagnosis,
    queryRewrites,
    retrieval,
    planner,
    controlLayer,
    failureLogging,
    finalOutput
  };
}

function renderResponse(data) {
  answerText.textContent = data.answer || "";
  renderSources(data.sources || []);
  if (data.needs_clarification) {
    hideFeedback();
  } else {
    renderFeedbackVisible();
  }
  lastRequestId = data.request_id;
  lastSessionId = data.session_id;

  const {
    debug,
    trace,
    diagnosis,
    queryRewrites,
    retrieval,
    planner,
    controlLayer,
    failureLogging,
    finalOutput
  } = traceOrFallback(data);

  renderKeyValues(requestMeta, [
    ["request_id", data.request_id],
    ["session_id", data.session_id],
    ["generation_chain_v2", String(debug.generation_chain_v2_enabled)],
    ["query_type", diagnosis.question_type || debug.query_type],
    ["retrieval_query", retrieval.query || debug.retrieval_query || data.question],
    ["planning_query", debug.planning_query || "-"]
  ]);

  renderKeyValues(selectorMeta, [
    ["selected_from", debug.selected_from],
    ["needs_clarification", String(Boolean(data.needs_clarification))],
    ["rewrite_triggered", String(debug.rewrite_triggered ?? controlLayer.rewritten_or_regenerated)],
    ["fallback_triggered", String(debug.fallback_triggered)],
    ["failure_logged", String(Boolean(failureLogging.failure_logged))],
    ["failure_type", failureLogging.failure_type || debug.failure_type || "-"]
  ]);

  renderKeyValues(diagnosisMeta, [
    ["question_type", diagnosis.question_type],
    ["primary_intent", diagnosis.primary_intent],
    ["user_role", diagnosis.user_role],
    ["role_confidence", diagnosis.role_confidence],
    ["task_nature", diagnosis.task_nature],
    ["time_horizon", diagnosis.time_horizon],
    ["response_mode", diagnosis.response_mode],
    ["root_cause_mode", diagnosis.root_cause_mode],
    ["needs_clarification", diagnosis.needs_clarification],
    ["clarification_question", diagnosis.clarification_question || "-"]
  ]);

  renderKeyValues(contractMeta, [
    ["whether_blocked", String(Boolean(controlLayer.whether_blocked))],
    ["triggered_guardrails", (controlLayer.triggered_guardrails || []).join(", ") || "-"],
    ["rewritten_or_regenerated", String(Boolean(controlLayer.rewritten_or_regenerated))],
    ["diagnosis_mode_check_pass", String(debug.diagnosis_mode_check_pass ?? controlLayer.final_checks?.diagnosis_mode_check_pass)],
    ["mechanism_name_check_pass", String(debug.mechanism_name_check_pass ?? controlLayer.final_checks?.mechanism_name_check_pass)],
    ["action_steps_match_count", debug.action_steps_match_count ?? controlLayer.final_checks?.action_steps_match_count]
  ]);

  const timings = debug.timings_ms || {};
  renderKeyValues(timingsGrid, [
    ["retrieval", `${timings.retrieval ?? 0} ms`],
    ["planner", `${timings.planner ?? 0} ms`],
    ["generation", `${timings.generation ?? 0} ms`],
    ["rewrite", `${timings.rewrite ?? 0} ms`],
    ["total", `${timings.total ?? 0} ms`]
  ]);

  plannerPre.textContent = JSON.stringify(
    {
      diagnosis,
      query_rewrites: queryRewrites,
      router_decision: debug.router_decision,
      retrieval,
      selected_evidence: debug.selected_evidence,
      planner_output_v2: planner,
      planner_output_v21: debug.planner_output_v21
    },
    null,
    2
  );
  rewriteTracePre.textContent = JSON.stringify(
    {
      rewrite_for_retrieval: queryRewrites.rewrite_for_retrieval || [],
      rewrite_for_reasoning: queryRewrites.rewrite_for_reasoning || debug.reasoning_rewrite || [],
      answer_outline: planner.answer_outline || [],
      answer_guardrails: planner.answer_guardrails || [],
      final_output: finalOutput
    },
    null,
    2
  );
  debugPre.textContent = JSON.stringify(
    {
      generation_trace: trace,
      debug_info: debug
    },
    null,
    2
  );
}

async function fetchHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    healthBadge.textContent = `${data.status} | ${data.runtime_profile} | ${data.model_name}`;
  } catch (error) {
    healthBadge.textContent = "health unavailable";
  }
}

async function fetchConfig() {
  const query = new URLSearchParams();
  if (debugRequested) {
    query.set("debug", "1");
  }
  if (debugToken) {
    query.set("token", debugToken);
  }
  try {
    const response = await fetch(`/api/config?${query.toString()}`);
    const data = await response.json();
    renderConfig(data);
  } catch (error) {
    renderExamplesFallback();
  }
}

async function askQuestion() {
  if (!questionInput || !askButton) {
    return;
  }
  const question = questionInput.value.trim();
  if (!question) {
    requestStatus.textContent = "Please enter a question.";
    return;
  }

  askButton.disabled = true;
  requestStatus.textContent = "Running pipeline...";

  try {
    const askUrl = `${window.location.origin}/api/ask`;
    const payload = {
      question,
      debug: debugRequested,
      debug_token: debugToken
    };
    console.log("[Ask] url:", askUrl);
    console.log("[Ask] payload:", payload);
    const response = await window.fetch(askUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`HTTP ${response.status}: ${text}`);
    }
    const data = await response.json();
    renderResponse(data);
    requestStatus.textContent = "Done.";
  } catch (error) {
    console.log("[Ask] fetch error:", error?.message || error);
    answerText.textContent = `Error: ${error.message}`;
    requestStatus.textContent = "Failed.";
  } finally {
    askButton.disabled = false;
  }
}

async function submitFeedback() {
  if (!lastRequestId) {
    feedbackStatus.textContent = "Submit a question first.";
    return;
  }
  feedbackStatus.textContent = "Sending feedback...";
  try {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        request_id: lastRequestId,
        session_id: lastSessionId,
        rating: selectedRating,
        comment: feedbackComment.value.trim() || null
      })
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Feedback failed");
    }
    feedbackStatus.textContent = "Thanks for the feedback.";
  } catch (error) {
    feedbackStatus.textContent = `Error: ${error.message}`;
  }
}

function initializePage() {
  if (askButton) {
    askButton.addEventListener("click", (event) => {
      event.preventDefault();
      askQuestion();
    });
  }
  if (submitFeedbackButton) {
    submitFeedbackButton.addEventListener("click", (event) => {
      event.preventDefault();
      submitFeedback();
    });
  }
  if (thumbUpButton) {
    thumbUpButton.addEventListener("click", () => setSelectedRating("up"));
  }
  if (thumbDownButton) {
    thumbDownButton.addEventListener("click", () => setSelectedRating("down"));
  }
  if (questionInput) {
    questionInput.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        askQuestion();
      }
    });
  }

  renderExamplesFallback();
  fetchHealth();
  fetchConfig();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializePage);
} else {
  initializePage();
}
