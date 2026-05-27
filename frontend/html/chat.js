const API_URL = "/api/chat";

const messagesEl = document.getElementById("messages");
const form       = document.getElementById("chat-form");
const input      = document.getElementById("question-input");
const btn        = form.querySelector("button");

marked.use({ gfm: true, breaks: true });

function appendMessage(text, role, isMarkdown = false) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (isMarkdown) {
    div.innerHTML = marked.parse(text);
  } else {
    div.textContent = text;
  }
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  input.value = "";
  btn.disabled = true;
  appendMessage(question, "user");
  const thinking = appendMessage("…", "bot");

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const data = await res.json();
    thinking.innerHTML = marked.parse(data.answer);

    if (data.images && data.images.length > 0) {
      const gallery = document.createElement("div");
      gallery.className = "img-gallery";
      data.images.forEach(src => {
        const img = document.createElement("img");
        img.src = src;
        img.alt = "";
        img.loading = "lazy";
        gallery.appendChild(img);
      });
      thinking.appendChild(gallery);
    }
  } catch (err) {
    thinking.className = "msg error";
    thinking.textContent = `Error: ${err.message}`;
  } finally {
    btn.disabled = false;
    input.focus();
  }
});
