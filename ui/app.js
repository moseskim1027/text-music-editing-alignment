const $ = (id) => document.getElementById(id)
const API = "http://localhost:8000"

fetch(`${API}/device`).then((response) => response.json()).then((device) => {
  $("device-name").textContent = `${device.recommended.toUpperCase()} ${device[device.recommended] ? "ready" : "fallback"}`
  $("device-detail").textContent = device.diagnostic
}).catch(() => {
  $("device-name").textContent = "API offline"
  $("device-detail").textContent = "Run make api"
})

$("run").addEventListener("click", () => {
  const instruction = $("instruction").value.trim()
  if (!instruction) { $("message").textContent = "Add an instruction before previewing the run."; return }
  const payload = {manifest: $("manifest").value, operation: $("operation").value, instruction, adapter_rank: Number($("rank").value), steps: Number($("steps").value), device: "mps", tracking_uri: "http://localhost:5000"}
  fetch(`${API}/experiments/preview`, {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(payload)})
    .then(async (response) => { const body = await response.json(); if (!response.ok) throw new Error(body.detail || "API validation failed"); return body })
    .then((body) => { $("payload").textContent = JSON.stringify(body.experiment, null, 2); $("preview").classList.remove("hidden"); $("message").textContent = "API contract validated. Preview is ready." })
    .catch((error) => { $("message").textContent = error.message })
})
