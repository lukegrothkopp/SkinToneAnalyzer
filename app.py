import os
import io
import json
import base64
from typing import Optional, Dict, Any, List

import streamlit as st
from PIL import Image
from openai import OpenAI

SKIN_TONES: List[str] = [
    "FAIR",
    "LIGHT",
    "LIGHT_MEDIUM",
    "MEDIUM",
    "MEDIUM_TAN",
    "TAN",
    "DARK",
    "DEEP",
]

MODEL = "gpt-4.1-mini"  # vision-capable model example shown in OpenAI docs


def _get_openai_client() -> OpenAI:
    # Prefer Streamlit secrets; fall back to env var
    api_key = None
    if hasattr(st, "secrets"):
        api_key = st.secrets.get("OPENAI_API_KEY")
    api_key = api_key or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in Streamlit Secrets or as an environment variable."
        )
    return OpenAI(api_key=api_key)


def _preprocess_to_jpeg_bytes(image_bytes: bytes, max_side: int = 1024) -> bytes:
    """
    Downscale + convert to JPEG to reduce token/cost and improve consistency.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    scale = min(max_side / w, max_side / h, 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


def _to_data_url(image_jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(image_jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@st.cache_data(show_spinner=False)
def suggest_skin_tone(
    selfie_bytes: bytes,
    reference_swatches_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "skin_tone": one of SKIN_TONES or "unknown",
        "confidence": 0..1,
        "needs_better_photo": bool,
        "notes": str,
        "warnings": [str, ...]
      }
    """
    client = _get_openai_client()

    selfie_jpeg = _preprocess_to_jpeg_bytes(selfie_bytes)
    selfie_url = _to_data_url(selfie_jpeg)

    content = [
        {
            "type": "input_text",
            "text": (
                "You are assisting a cosmetics subscription quiz.\n"
                "Task: choose the best matching skin tone label from this fixed set:\n"
                f"{', '.join(SKIN_TONES)}.\n\n"
                "Rules:\n"
                "- Focus on the person's natural skin tone (cheek/jaw/neck if visible).\n"
                "- Ignore background, hair, clothing, and temporary redness.\n"
                "- Lighting can skew color; if the lighting is very tinted (strong warm/cool cast), "
                "or the face/skin is not clearly visible, return skin_tone='unknown' and set "
                "needs_better_photo=true.\n"
                "- Return ONLY valid JSON that matches the provided schema."
            ),
        }
    ]

    # Optional: include your swatch-reference image to anchor the labels
    # (e.g., export a clean image of the 8 swatches from your design system)
    if reference_swatches_bytes:
        ref_jpeg = _preprocess_to_jpeg_bytes(reference_swatches_bytes)
        content.append({"type": "input_image", "image_url": _to_data_url(ref_jpeg)})

    # User selfie last
    content.append({"type": "input_image", "image_url": selfie_url})

    schema = {
        "type": "object",
        "properties": {
            "skin_tone": {"type": "string", "enum": SKIN_TONES + ["unknown"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "needs_better_photo": {"type": "boolean"},
            "notes": {"type": "string"},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["skin_tone", "confidence", "needs_better_photo", "notes", "warnings"],
        "additionalProperties": False,
    }

    resp = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": content}],
        max_output_tokens=220,
        text={
            "format": {
                "type": "json_schema",
                "name": "skin_tone_suggestion",
                "schema": schema,
                "strict": True,
            }
        },
    )

    # With strict JSON schema, output_text should be a valid JSON string
    return json.loads(resp.output_text)


# -------------------------
# Streamlit UI example
# -------------------------
st.title("Skin Tone Auto-Suggestion (8-bucket)")

st.caption(
    "Tip: best results come from a makeup-free selfie in natural daylight (no filters), "
    "with face + neck visible."
)

selfie = st.file_uploader("Upload a selfie", type=["jpg", "jpeg", "png"])

# Optional: include a reference image containing your 8 tone swatches
# You can remove this if you don’t want/need it.
ref = st.file_uploader(
    "Optional: Upload your 8-swatch reference image (exported from your UI)",
    type=["jpg", "jpeg", "png"],
)

if selfie:
    st.image(selfie, caption="Uploaded selfie", use_container_width=True)

    if st.button("Suggest my skin tone"):
        with st.spinner("Analyzing..."):
            try:
                result = suggest_skin_tone(
                    selfie_bytes=selfie.getvalue(),
                    reference_swatches_bytes=ref.getvalue() if ref else None,
                )
            except Exception as e:
                st.error(f"Failed to analyze image: {e}")
                st.stop()

        st.subheader("Result")
        st.json(result)

        # Auto-select in UI (and allow manual override)
        suggested = result.get("skin_tone", "unknown")
        if suggested in SKIN_TONES:
            idx = SKIN_TONES.index(suggested)
            st.success(f"Suggested skin tone: **{suggested}** (confidence {result.get('confidence', 0):.2f})")
        else:
            idx = 0
            st.warning("Could not confidently determine tone. Try a clearer photo.")

        final_choice = st.selectbox(
            "Use this selection (you can override):",
            options=SKIN_TONES,
            index=idx,
        )

        st.write("Final selection:", final_choice)
