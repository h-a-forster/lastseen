// "Questions": the written FAQ (static/faq.json) as expandable entries, and a box that finds the right written answer
// (app/faq.py): at once when the question is close to one, otherwise Muse Spark Open picks among them.

const F = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

async function showFaq() {
  const entries = await fetch("/faq.json").then((r) => r.json());
  F("faq-list").innerHTML = entries.map((e) =>
    `<details id="faq-${esc(e.id)}"><summary>${esc(e.q)}</summary><p>${esc(e.a)}</p></details>`).join("");
}

function answerHtml(e) {
  return `<div class="ask-answer"><strong>${esc(e.q)}</strong>${esc(e.a)}</div>`;
}

async function ask(event) {
  event.preventDefault();
  const question = F("ask-text").value.trim();
  if (!question) return;
  F("ask-result").innerHTML = `<p class="ask-note">Finding the answer...</p>`;
  try {
    const response = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }) });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || response.statusText);
    const how = body.matched_by === "model" ? "Muse Spark Open picked these written answers" : "Matched to a written answer";
    F("ask-result").innerHTML = body.answers.length
      ? `<p class="ask-note">${how}.</p>${body.answers.map(answerHtml).join("")}`
      : `<p class="ask-note">We don't have a written answer for that. Closest questions: ${body.suggestions.map((e) =>
          `<button type="button" class="ask-suggest" data-id="${esc(e.id)}">${esc(e.q)}</button>`).join(" · ") || "none"}</p>`;
    F("ask-result").querySelectorAll(".ask-suggest").forEach((b) => (b.onclick = () => openEntry(b.dataset.id)));
  } catch (err) {
    F("ask-result").innerHTML = `<p class="ask-note">Could not answer: ${esc(err.message)}. The written answers are below.</p>`;
  }
}

function openEntry(id) {
  const entry = F(`faq-${id}`);
  if (!entry) return;
  entry.open = true;
  entry.classList.add("flash");
  entry.scrollIntoView({ block: "center" });
}

F("ask").onsubmit = ask;
showFaq().catch(() => F("faq").classList.add("hidden"));
