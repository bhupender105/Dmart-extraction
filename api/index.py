import os
import json
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

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


# ── Shelf Image Analysis ─────────────────────────────────────────────────────────

@app.post("/analyze", tags=["Analysis"])
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
