const $ = (id) => document.getElementById(id)
const setText = (id, value) => { const element = $(id); if (element) element.textContent = value }
const API = "http://localhost:8000"
const currentPayload = () => ({manifest: $("manifest").value, example_id: $("example-id").value.trim() || null, operation: $("operation").value, instruction: $("instruction").value.trim(), adapter_rank: Number($("rank").value), steps: Number($("steps").value), device: "mps", tracking_uri: "http://localhost:5000"})

fetch(`${API}/device`).then((response) => response.json()).then((device) => {
  setText("device-name", `${device.recommended.toUpperCase()} ${device[device.recommended] ? "ready" : "fallback"}`)
  setText("device-detail", device.diagnostic)
}).catch(() => {
  setText("device-name", "API offline")
  setText("device-detail", "Run make api")
})

$("run").addEventListener("click", () => {
  const instruction = $("instruction").value.trim()
  if (!instruction) { $("message").textContent = "Add an instruction before previewing the run."; return }
  const payload = currentPayload()
  fetch(`${API}/experiments/preview`, {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(payload)})
    .then(async (response) => { const body = await response.json(); if (!response.ok) throw new Error(body.detail || "API validation failed"); return body })
    .then((body) => { $("payload").textContent = JSON.stringify(body.experiment, null, 2); $("preview").classList.remove("hidden"); $("start").disabled = false; $("message").textContent = "API contract validated. Preview is ready." })
    .catch((error) => { $("message").textContent = error.message })
})

$("start").addEventListener("click", async () => {
  $("start").disabled = true
  try {
    const payload = currentPayload()
    if (!payload.instruction) throw new Error("Add an instruction before starting training")
    $("payload").textContent = JSON.stringify(payload, null, 2)
    const response = await fetch(`${API}/experiments/start`, {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(payload)})
    if (!response.ok) {
      const body = await response.json()
      throw new Error(body.detail || "Training could not be started")
    }
    $("run-state").textContent = "RUNNING"
    const timer = setInterval(async () => {
      const status = await fetch(`${API}/experiments/status`).then((r) => r.json())
      $("run-state").textContent = (status.phase || status.status).toUpperCase()
      $("step").textContent = `${status.step} / ${status.steps}`
      $("edit-loss").textContent = status.loss === undefined ? "—" : status.loss.toFixed(4)
      const result = status.result || {}
      if (status.status === "completed") {
        $("message").textContent = "Training completed. Audio and evaluation artifacts will appear when produced by the run."
        $("mlflow-link").classList.remove("hidden")
        $("evaluation-status").textContent = "COMPLETE"
        $("evaluation-status").classList.add("success")
        const evaluation = status.result?.evaluation || {}
        setText("adherence", evaluation.adherence_proxy === undefined ? "—" : evaluation.adherence_proxy.toFixed(3))
        setText("preservation", evaluation.preservation_proxy === undefined ? "—" : evaluation.preservation_proxy.toFixed(3))
        setText("quality", evaluation.quality_proxy === undefined ? "—" : evaluation.quality_proxy.toFixed(3))
        setText("preference-win", evaluation.preference_win_rate == null ? "N/A" : evaluation.preference_win_rate.toFixed(3))
        const output = status.result?.output || ""
        const match = output.match(/"artifacts"\s*:\s*\{[\s\S]*?"source_audio"\s*:\s*"([^"]+)"[\s\S]*?"generated_audio"\s*:\s*"([^"]+)"/)
        if (match) {
          const artifactUrl = (path) => `${API}/artifacts/${path.replace(/^outputs[\\/]/, "").replaceAll("\\\\", "/")}`
          $("source-audio").src = artifactUrl(match[1])
          $("generated-audio").src = artifactUrl(match[2])
          $("audio-status").textContent = "Generated artifacts are ready for playback."
        }
      }
      const progress = status.status === "completed" || status.phase === "generating" ? 100 : (status.steps ? 100 * status.step / status.steps : 0)
      $("progress-bar").style.width = `${progress}%`
      $("training-progress").textContent = `${Math.round(progress)}%`
      $("generation-phase").classList.toggle("hidden", status.phase !== "generating")
      if (status.status !== "running") { clearInterval(timer); $("start").disabled = false }
    }, 1000)
  } catch (error) { $("message").textContent = error.message; $("start").disabled = false }
})
