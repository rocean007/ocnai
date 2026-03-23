// js/main.js — Shared JavaScript for VeggieWorld

// ── Floating Chat Bubble ──────────────────────────────────────────────────────

function toggleChat() {
  const frame  = document.getElementById("chat-frame");
  const bubble = document.getElementById("chat-bubble");
  if (!frame) return;

  const isOpen = frame.style.display === "block";
  frame.style.display = isOpen ? "none" : "block";
  bubble.title = isOpen ? "Open VeggieBot" : "Close VeggieBot";
}

// Close chat if clicking outside of it
document.addEventListener("click", function (e) {
  const frame  = document.getElementById("chat-frame");
  const bubble = document.getElementById("chat-bubble");
  if (!frame || !bubble) return;
  if (frame.style.display !== "block") return;

  const clickedInside = frame.contains(e.target) || bubble.contains(e.target);
  if (!clickedInside) {
    frame.style.display = "none";
  }
});
