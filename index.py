import os
import json
import base64
import secrets
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# ── API Key Security ─────────────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def verify_api_key(key: str = Security(_api_key_header)) -> None:
    """Validate the X-API-Key header against the API_KEY environment variable."""
    expected = os.environ.get("API_KEY", "")
    if not expected:
        raise HTTPException(status_code=500, detail="API_KEY environment variable is not configured.")
    if not secrets.compare_digest(key, expected):
        raise HTTPException(status_code=403, detail="Invalid API key.")


app = FastAPI(
    title="Retail Shelf Audit API",
    description="Analyze supermarket shelf images and extract ABC Foods merchandising compliance data using Gemini AI.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are an expert Retail Execution Compliance Agent powered by Agentforce. Your task is to analyze the provided image of a modern trade supermarket shelf and extract merchandising details with high precision.

Follow these instructions perfectly:

Scan the display from Top to Bottom, shelf by shelf.

Identify "ABC Foods" products (e.g., "ABC Premium Sweets" in gold boxes, "ABC Spicy Mix 500g" in red pouches). Note: Account for slight typos or blurry text in the image (e.g., treating "SPICY INIX" as "SPICY MIX").

Identify non-ABC products, categorizing them as either "Competitor Snacks" or "Other Categories" (e.g., Kellogg's Oats, Quaker, Saffola, Aashirvaad Atta, India Gate Rice).

Count the visible "facings" (front-facing items) for each product on each shelf.

Extract any visible pricing tags associated with the shelves.

Calculate the approximate Shelf Share percentage of ABC Foods products on this specific display rack.

Generate actionable compliance insights and recommended tasks based on the audit (e.g., restock, correct planogram).

Output the extracted data strictly in the following JSON format without any markdown wrappers, conversational text, or explanations.

{
"audit_summary": {
"total_shelves_detected": <integer>,
"abc_shelf_share_percentage": <float>,
"is_planogram_compliant": <boolean>,
"overall_store_lighting_and_visibility": "<string: Good, Fair, Poor>"
},
"shelf_details":[
{
"shelf_index": <integer: 1 for top shelf, 2 for next, etc.>,
"shelf_level_type": "<string: Top, Eye-level, Middle, Bottom>",
"products_detected":[
{
"brand_name": "<string>",
"product_name": "<string>",
"packaging_type": "<string: Box, Pouch, Bag, etc.>",
"is_abc_product": <boolean>,
"facings_count": <integer>,
"visible_price_tag": "<string or null>"
}
]
}
],
"compliance_issues":[
{
"issue_type": "<string: Missing SKU, Wrong Placement, Low Stock, Competitor Intrusion>",
"description": "<string: Detailed description of the issue>"
}
],
"recommended_action_tasks":[
{
"task_type": "<string: Restock Order, Merchandising Correction, Competitor Flag>",
"urgency": "<string: High, Medium, Low>",
"action_description": "<string: e.g., 'Replenish ABC Spicy Mix on Shelf 4', 'Move Premium Sweets to Eye-Level'>"
}
]
}"""


def _get_llm() -> ChatGoogleGenerativeAI:
    """Initialize the Gemini LLM. Lazy-loaded to avoid cold-start issues."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY environment variable is not set.",
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.1,
    )


def _parse_json_response(raw_text: str) -> dict:
    """Strip markdown code fences if present, then parse JSON."""
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        raw_text = parts[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    return json.loads(raw_text.strip())


# ── Health Check ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health_check():
    """Returns the health status of the API."""
    return {"status": "ok", "service": "Retail Shelf Audit API", "version": "1.0.0"}


# ── Landing Page ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing_page():
    """Serve the HTML landing page."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ShelfScan AI — Retail Shelf Auditor</title>
<style>
  :root{--brand:#2563eb;--brand-dark:#1d4ed8;--accent:#16a34a;--bg:#f8fafc;--card:#ffffff;--text:#1e293b;--muted:#64748b;--border:#e2e8f0;--radius:12px}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
  /* Nav */
  nav{background:var(--card);border-bottom:1px solid var(--border);padding:0 2rem;display:flex;align-items:center;justify-content:space-between;height:60px;position:sticky;top:0;z-index:100}
  .nav-logo{font-size:1.2rem;font-weight:700;color:var(--brand);text-decoration:none}
  .nav-links a{color:var(--muted);text-decoration:none;margin-left:1.5rem;font-size:.9rem}
  .nav-links a:hover{color:var(--brand)}
  /* Hero */
  .hero{background:linear-gradient(135deg,#1e3a8a 0%,#2563eb 60%,#0ea5e9 100%);color:#fff;padding:5rem 2rem 4rem;text-align:center}
  .hero h1{font-size:2.8rem;font-weight:800;line-height:1.15;margin-bottom:1rem}
  .hero p{font-size:1.15rem;opacity:.88;max-width:600px;margin:0 auto 2rem}
  .hero-badge{display:inline-block;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.3);border-radius:20px;padding:.3rem 1rem;font-size:.8rem;margin-bottom:1.2rem;letter-spacing:.05em}
  /* Features */
  .section{padding:4rem 2rem;max-width:1100px;margin:0 auto}
  .section-title{font-size:1.6rem;font-weight:700;margin-bottom:.5rem}
  .section-sub{color:var(--muted);margin-bottom:2.5rem}
  .features{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1.5rem}
  .feature-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:1.5rem}
  .feature-icon{font-size:1.8rem;margin-bottom:.8rem}
  .feature-card h3{font-size:1rem;font-weight:600;margin-bottom:.4rem}
  .feature-card p{font-size:.875rem;color:var(--muted);line-height:1.55}
  /* Upload Form */
  .upload-section{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:2rem;max-width:700px;margin:0 auto}
  .form-group{margin-bottom:1.2rem}
  label{display:block;font-size:.875rem;font-weight:600;margin-bottom:.4rem}
  input[type=text],input[type=file]{width:100%;padding:.65rem 1rem;border:1px solid var(--border);border-radius:8px;font-size:.9rem;outline:none;transition:border .2s}
  input[type=text]:focus{border-color:var(--brand)}
  .btn{display:inline-flex;align-items:center;gap:.5rem;background:var(--brand);color:#fff;border:none;padding:.75rem 2rem;border-radius:8px;font-size:1rem;font-weight:600;cursor:pointer;transition:background .2s}
  .btn:hover{background:var(--brand-dark)}
  .btn:disabled{background:#94a3b8;cursor:not-allowed}
  .spinner{width:18px;height:18px;border:2px solid rgba(255,255,255,.4);border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite;display:none}
  @keyframes spin{to{transform:rotate(360deg)}}
  /* Result */
  #result-box{display:none;margin-top:1.5rem}
  .result-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:.5rem}
  .result-header span{font-size:.875rem;font-weight:600}
  .copy-btn{font-size:.78rem;padding:.3rem .8rem;background:#f1f5f9;border:1px solid var(--border);border-radius:6px;cursor:pointer}
  pre{background:#0f172a;color:#e2e8f0;padding:1.2rem;border-radius:8px;overflow:auto;max-height:420px;font-size:.8rem;line-height:1.6;white-space:pre-wrap;word-break:break-word}
  .error-msg{background:#fef2f2;border:1px solid #fecaca;color:#b91c1c;padding:.75rem 1rem;border-radius:8px;font-size:.875rem;margin-top:1rem;display:none}
  /* Footer */
  footer{background:var(--card);border-top:1px solid var(--border);text-align:center;padding:1.5rem;color:var(--muted);font-size:.85rem;margin-top:3rem}
  footer a{color:var(--brand);text-decoration:none}
</style>
</head>
<body>

<nav>
  <a class="nav-logo" href="/">&#x1F4CA; ShelfScan AI</a>
  <div class="nav-links">
    <a href="#features">Features</a>
    <a href="#try-it">Try it</a>
    <a href="/docs">API Docs</a>
    <a href="/health">Health</a>
  </div>
</nav>

<section class="hero">
  <div class="hero-badge">&#x26A1; Powered by Gemini 2.0 Flash</div>
  <h1>Retail Shelf Audit,<br/>Powered by AI</h1>
  <p>Upload a supermarket shelf photo and instantly receive a full merchandising compliance report — shelf share, facings, pricing, and recommended actions.</p>
</section>

<div class="section" id="features">
  <div class="section-title">What ShelfScan AI does</div>
  <div class="section-sub">One image. Seconds. A complete audit.</div>
  <div class="features">
    <div class="feature-card">
      <div class="feature-icon">&#x1F4CA;</div>
      <h3>Shelf Share Analysis</h3>
      <p>Calculates the exact percentage of shelf space owned by ABC Foods vs competitors.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon">&#x1F50D;</div>
      <h3>Product Detection</h3>
      <p>Identifies every brand, SKU, and facing on every shelf — top to bottom, left to right.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon">&#x1F4B0;</div>
      <h3>Price Tag Extraction</h3>
      <p>Reads visible price tags and associates them with the correct product row.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon">&#x2705;</div>
      <h3>Compliance Insights</h3>
      <p>Flags planogram violations, low stock, competitor intrusions, and wrong placements.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon">&#x1F4CB;</div>
      <h3>Action Tasks</h3>
      <p>Generates prioritised restocking and merchandising correction tasks ready for field teams.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon">&#x1F512;</div>
      <h3>API Key Protected</h3>
      <p>All analysis endpoints are secured with an API key — only authorised clients can query.</p>
    </div>
  </div>
</div>

<div class="section" id="try-it">
  <div class="section-title">Try the Audit API</div>
  <div class="section-sub">Upload a shelf image to get a live JSON compliance report.</div>
  <div class="upload-section">
    <div class="form-group">
      <label for="api-key">X-API-Key <span style="color:#ef4444">*</span></label>
      <input type="text" id="api-key" placeholder="Enter your API key" autocomplete="off" spellcheck="false"/>
    </div>
    <div class="form-group">
      <label for="shelf-image">Shelf Image (JPG / PNG / WEBP)</label>
      <input type="file" id="shelf-image" accept="image/jpeg,image/png,image/webp"/>
    </div>
    <button class="btn" id="analyze-btn" onclick="analyzeShelf()">
      <span class="spinner" id="spinner"></span>
      <span id="btn-text">&#x1F50E; Analyze Shelf</span>
    </button>
    <div class="error-msg" id="error-msg"></div>
    <div id="result-box">
      <div class="result-header">
        <span>Audit Result (JSON)</span>
        <button class="copy-btn" onclick="copyResult()">Copy</button>
      </div>
      <pre id="result-pre"></pre>
    </div>
  </div>
</div>

<footer>
  &copy; 2026 ShelfScan AI &mdash; ABC Foods Retail Execution &mdash;
  <a href="/docs">Swagger Docs</a> &middot; <a href="/health">Health Check</a>
</footer>

<script>
async function analyzeShelf() {
  const key = document.getElementById('api-key').value.trim();
  const fileInput = document.getElementById('shelf-image');
  const errEl = document.getElementById('error-msg');
  const resultBox = document.getElementById('result-box');
  const resultPre = document.getElementById('result-pre');
  const btn = document.getElementById('analyze-btn');
  const spinner = document.getElementById('spinner');
  const btnText = document.getElementById('btn-text');

  errEl.style.display = 'none';
  resultBox.style.display = 'none';

  if (!key) { showError('Please enter your API key.'); return; }
  if (!fileInput.files.length) { showError('Please select a shelf image.'); return; }

  btn.disabled = true;
  spinner.style.display = 'block';
  btnText.textContent = 'Analyzing…';

  const formData = new FormData();
  formData.append('image', fileInput.files[0]);

  try {
    const resp = await fetch('/analyze', {
      method: 'POST',
      headers: { 'X-API-Key': key },
      body: formData
    });
    const data = await resp.json();
    if (!resp.ok) {
      showError('Error ' + resp.status + ': ' + (data.detail || JSON.stringify(data)));
    } else {
      resultPre.textContent = JSON.stringify(data, null, 2);
      resultBox.style.display = 'block';
    }
  } catch(e) {
    showError('Network error: ' + e.message);
  } finally {
    btn.disabled = false;
    spinner.style.display = 'none';
    btnText.textContent = '&#x1F50E; Analyze Shelf';
  }
}

function showError(msg) {
  const el = document.getElementById('error-msg');
  el.textContent = msg;
  el.style.display = 'block';
}

function copyResult() {
  navigator.clipboard.writeText(document.getElementById('result-pre').textContent);
}
</script>
</body>
</html>"""
    return HTMLResponse(content=html, status_code=200)




@app.post("/analyze", tags=["Analysis"], dependencies=[Depends(verify_api_key)])
async def analyze_shelf(image: UploadFile = File(..., description="Shelf image (JPG, PNG, or WEBP)")):
    """
    Upload a retail shelf image to receive a structured JSON audit.

    Returns a full compliance report including:
    - Shelf share percentage for ABC Foods
    - Per-shelf product breakdown with facings and price tags
    - Compliance issues detected
    - Recommended action tasks
    """
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    content_type = image.content_type or ""
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{content_type}'. Allowed: {', '.join(allowed_types)}",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    encoded = base64.b64encode(image_bytes).decode("utf-8")

    llm = _get_llm()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Analyze this retail shelf image and extract all merchandising data exactly as specified in the instructions.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{content_type};base64,{encoded}"},
                },
            ]
        ),
    ]

    response = llm.invoke(messages)

    # response.content can be a string or list depending on the model response
    raw_text = response.content
    if isinstance(raw_text, list):
        raw_text = raw_text[0].get("text", "") if raw_text else ""

    try:
        data = _parse_json_response(raw_text)
    except (json.JSONDecodeError, IndexError, KeyError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to parse model response as JSON: {exc}. Raw response: {raw_text[:500]}",
        )

    return data
