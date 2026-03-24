// api/chat.js

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
      error: "Not configured. Add HF_TOKEN and HF_MODEL_URL in Vercel → Settings → Environment Variables, then Redeploy."
    });
  }

  // Qwen 2.5 chat format
  const prompt =
    `<|im_start|>system\n` +
    `You are SagjiBot, an expert vegetable assistant. ` +
    `You help with vegetable nutrition, growing, chemicals, pesticides, and food safety. ` +
    `Reply in English by default. ` +
    `Only reply in Nepali if the user writes in Nepali OR explicitly asks you to reply in Nepali. ` +
    `Never mix languages in one reply. ` +
    `Give accurate, practical answers. Keep replies under 250 words unless more detail is needed.<|im_end|>\n` +
    `<|im_start|>user\n${message.trim()}<|im_end|>\n` +
    `<|im_start|>assistant\n`;

  try {
    const hfRes = await fetch(HF_MODEL_URL, {
      method:  "POST",
      headers: {
        Authorization:      `Bearer ${HF_TOKEN}`,
        "Content-Type":     "application/json",
        "x-wait-for-model": "true",
      },
      body: JSON.stringify({
        inputs: prompt,
        parameters: {
          max_new_tokens:   400,
          temperature:      0.7,
          top_p:            0.9,
          repetition_penalty: 1.1,
          return_full_text: false,
          do_sample:        true,
        },
      }),
    });

    if (hfRes.status === 503) {
      const body = await hfRes.json().catch(() => ({}));
      return res.status(503).json({
        error:       "loading",
        retry_after: Math.ceil(body.estimated_time ?? 25),
      });
    }

    if (!hfRes.ok) {
      const txt = await hfRes.text().catch(() => "");
      console.error("HF error:", hfRes.status, txt.slice(0, 300));
      return res.status(502).json({ error: "AI service error. Please try again." });
    }

    const data  = await hfRes.json();
    let reply   = (data[0]?.generated_text ?? "").trim();

    // Clean up any leaked template tokens
    reply = reply
      .replace(/<\|im_end\|>/g, "")
      .replace(/<\|im_start\|>\w+\n/g, "")
      .trim();

    if (!reply) return res.status(502).json({ error: "Empty response. Please try again." });

    return res.status(200).json({ reply });

  } catch (err) {
    console.error("Handler error:", err.message);
    return res.status(500).json({ error: "Server error. Please try again." });
  }
}
