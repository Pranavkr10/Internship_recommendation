async function showRoadmap(track) {
  const id = `roadmap-${track.replaceAll(" ", "-")}`;
  const box = document.getElementById(id);

  if (!box) return;

  if (!box.classList.contains("hidden")) {
    box.classList.add("hidden");
    return;
  }

  box.classList.remove("hidden");
  box.innerHTML = `<p class="text-sm text-slate-500">Loading roadmap...</p>`;

  const res = await fetch("/roadmap", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ track })
  });

  const data = await res.json();

  box.innerHTML = `
    <div class="bg-white dark:bg-slate-800 border border-sage/20 rounded-2xl p-4 space-y-4">
      <div>
        <h4 class="font-bold text-slate-900 dark:text-white">${data.title}</h4>
        <p class="text-xs text-slate-500 mt-1">Duration: ${data.duration}</p>
      </div>

      <div>
        <p class="font-semibold text-sm mb-2">Steps</p>
        <ol class="list-decimal ml-5 text-sm text-slate-600 dark:text-slate-300 space-y-1">
          ${data.steps.map(step => `<li>${step}</li>`).join("")}
        </ol>
      </div>

      <div>
        <p class="font-semibold text-sm mb-2">Resources</p>
        <div class="space-y-2">
          ${data.resources.map(r => `
            <a href="${r.url}" target="_blank"
               class="block p-3 rounded-xl border border-sage/20 hover:bg-sage/10 transition">
              <p class="text-sm font-semibold text-slate-900 dark:text-white">${r.name}</p>
              <p class="text-xs text-slate-500">${r.type || "Resource"}</p>
            </a>
          `).join("")}
        </div>
      </div>
    </div>
  `;
}
