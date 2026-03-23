// api/chat.js
// Vercel serverless function — receives messages and calls your Hugging Face model

export default async function handler(req, res) {
  // CORS headers — allows your website to call this API
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  // Browser sends OPTIONS preflight before POST — just confirm it
  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Only POST requests are allowed" });
  }

  const { message } = req.body;

  if (!message || typeof message !== "string" || message.trim().length === 0) {
    return res.status(400).json({ error: "No message provided" });
  }

  const HF_TOKEN     = process.env.HF_TOKEN;
  const HF_MODEL_URL = process.env.HF_MODEL_URL;

  if (!HF_TOKEN || !HF_MODEL_URL) {
    console.error("Missing environment variables: HF_TOKEN or HF_MODEL_URL");
    return res.status(500).json({ error: "Server configuration error" });
  }

  // Format prompt in Mistral's instruction template
  const prompt =
    `<s>[INST] <<SYS>>\n` +
    `You are VeggieBot, a highly knowledgeable expert on vegetables, plant farming, ` +
    `agricultural chemicals, and plant diseases. ` +
    `Give specific, accurate, and practical answers. ` +
    `Keep responses under 250 words unless asked for more detail.\n` +
    `<</SYS>>\n\n` +
    `${message.trim()} [/INST]`;

  try {
    const response = await fetch(HF_MODEL_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json",
        "x-wait-for-model": "true",  // Wait if model is loading instead of returning 503
      },
      body: JSON.stringify({
        inputs: prompt,
        parameters: {
          max_new_tokens: 400,
          temperature: 0.7,
          top_p: 0.9,
          return_full_text: false,
          do_sample: true,
        },
      }),
    });

    // Model is still loading (cold start) — can take 20–30s on free tier
    if (response.status === 503) {
      const body = await response.json().catch(() => ({}));
      const wait  = body.estimated_time ? Math.ceil(body.estimated_time) : 30;
      return res.status(503).json({
        error: `Model is loading. Please wait ${wait} seconds and try again.`,
        retry_after: wait,
      });
    }

    if (!response.ok) {
      const errorBody = await response.text();
      console.error("HF API error:", response.status, errorBody);
      return res.status(502).json({ error: "AI service error. Please try again." });
    }

    const data = await response.json();

    if (!Array.isArray(data) || !data[0]?.generated_text) {
      console.error("Unexpected HF response:", JSON.stringify(data));
      return res.status(502).json({ error: "Unexpected response from AI." });
    }

    const reply = data[0].generated_text.trim();

    return res.status(200).json({ reply });

  } catch (error) {
    console.error("Fetch error:", error.message);
    return res.status(500).json({ error: "Server error: " + error.message });
  }
}
