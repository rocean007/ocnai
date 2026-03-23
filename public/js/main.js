// js/main.js

const toggle = document.getElementById("navToggle");
const links  = document.getElementById("navLinks");

if (toggle && links) {
  toggle.addEventListener("click", () => links.classList.toggle("open"));

  document.addEventListener("click", (e) => {
    if (links.classList.contains("open") &&
        !links.contains(e.target) &&
        !toggle.contains(e.target)) {
      links.classList.remove("open");
    }
  });
}
