const $ = (id) => document.getElementById(id)
$("run").addEventListener("click", () => {
  const instruction = $("instruction").value.trim()
  if (!instruction) { $("message").textContent = "Add an instruction before previewing the run."; return }
  const payload = {manifest: $("manifest").value, operation: $("operation").value, instruction, adapter_rank: Number($("rank").value), steps: Number($("steps").value), device: "mps", tracking_uri: "http://localhost:5000"}
  $("payload").textContent = JSON.stringify(payload, null, 2)
  $("preview").classList.remove("hidden")
  $("message").textContent = "Contract validated. Preview is ready."
})
