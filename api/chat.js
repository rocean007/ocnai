// api/chat.js — Vercel serverless function
// Receives a message from the website and calls your Hugging Face model

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin",  "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST")   return res.status(405).json({ error: "POST only" });

  const { message } = req.body ?? {};
  if (!message?.trim()) return res.status(400).json({ error: "No message provided" });

  const HF_TOKEN     = process.env.HF_TOKEN;
  const HF_MODEL_URL = process.env.HF_MODEL_URL;

  if (!HF_TOKEN || !HF_MODEL_URL) {
    return res.status(500).json({
      error: "Not configured. Add HF_TOKEN and HF_MODEL_URL in Vercel → Settings → Environment Variables, then redeploy."
    });
  }

  // Qwen instruction format
  const prompt =
    `<|im_start|>system\n` +
    `You are SagjiBot, a vegetable expert. Reply in Nepali if asked in Nepali, English if asked in English.<|im_end|>\n` +
    `<|im_start|>user\n${message.trim()}<|im_end|>\n` +
    `<|im_start|>assistant\n`;

  try {
    const hfRes = await fetch(HF_MODEL_URL, {
      method:  "POST",
      headers: {
        Authorization:      `Bearer ${HF_TOKEN}`,
        "Content-Type":     "application/json",
        "x-wait-for-model": "true",   // wait instead of 503 during cold start
      },
      body: JSON.stringify({
        inputs: prompt,
        parameters: {
          max_new_tokens:   350,
          temperature:      0.7,
          top_p:            0.9,
          return_full_text: false,
          do_sample:        true,
        },
      }),
    });

    // Model still loading (free HF cold start ~25s)
    if (hfRes.status === 503) {
      const body = await hfRes.json().catch(() => ({}));
      return res.status(503).json({
        error:       "loading",
        retry_after: Math.ceil(body.estimated_time ?? 25),
      });
    }

    if (!hfRes.ok) {
      const txt = await hfRes.text().catch(() => "");
      console.error("HF API error:", hfRes.status, txt);
      return res.status(502).json({ error: "AI service error. Please try again." });
    }

    const data  = await hfRes.json();
    const reply = (data[0]?.generated_text ?? "").trim();
    if (!reply) return res.status(502).json({ error: "Empty response from model." });

    return res.status(200).json({ reply });

  } catch (err) {
    console.error("Handler error:", err.message);
    return res.status(500).json({ error: "Server error: " + err.message });
  }
}
