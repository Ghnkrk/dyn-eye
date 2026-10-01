"""
DYN-EYE Architecture Document Generator

Generates a detailed PDF describing the full system architecture,
every pipeline node, data flow, and retraining workflow.
All descriptions are derived directly from the source code.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import (
    HexColor, white, black, Color,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.graphics import renderPDF
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "DYN-EYE_Architecture_Document.pdf"

# ── Color Palette ────────────────────────────────────────────
PRIMARY     = HexColor("#1a1a2e")
ACCENT      = HexColor("#16213e")
BLUE        = HexColor("#0f3460")
TEAL        = HexColor("#1b998b")
ORANGE      = HexColor("#e77f43")
LIGHT_GRAY  = HexColor("#f0f0f5")
MEDIUM_GRAY = HexColor("#888888")
DARK_TEXT    = HexColor("#1a1a2e")
BORDER      = HexColor("#cccccc")

# Node colors for the flowchart
NODE_COLORS = [
    HexColor("#2196F3"),  # YOLO Inference - blue
    HexColor("#00897B"),  # Dataset Context - teal
    HexColor("#7B1FA2"),  # VLM Annotation - purple
    HexColor("#F57F17"),  # Crop Extraction - amber
    HexColor("#C62828"),  # DINOv2 Features - red
    HexColor("#1565C0"),  # FAISS Search - dark blue
    HexColor("#2E7D32"),  # HDBSCAN - green
    HexColor("#455A64"),  # Manifest Save - gray-blue
    HexColor("#E65100"),  # Dashboard - deep orange
    HexColor("#AD1457"),  # Retraining - pink
]


def _styles():
    ss = getSampleStyleSheet()

    ss.add(ParagraphStyle(
        "DocTitle", parent=ss["Title"],
        fontSize=26, leading=32, textColor=PRIMARY,
        spaceAfter=6, alignment=TA_CENTER,
    ))
    ss.add(ParagraphStyle(
        "DocSubtitle", parent=ss["Normal"],
        fontSize=12, leading=16, textColor=MEDIUM_GRAY,
        spaceAfter=20, alignment=TA_CENTER,
    ))
    ss.add(ParagraphStyle(
        "SectionHead", parent=ss["Heading1"],
        fontSize=16, leading=20, textColor=PRIMARY,
        spaceBefore=18, spaceAfter=8,
        borderWidth=0, borderPadding=0,
    ))
    ss.add(ParagraphStyle(
        "SubHead", parent=ss["Heading2"],
        fontSize=13, leading=17, textColor=BLUE,
        spaceBefore=12, spaceAfter=4,
    ))
    ss.add(ParagraphStyle(
        "BodyText2", parent=ss["Normal"],
        fontSize=10, leading=14, textColor=DARK_TEXT,
        alignment=TA_JUSTIFY, spaceAfter=6,
    ))
    ss.add(ParagraphStyle(
        "BulletText", parent=ss["Normal"],
        fontSize=10, leading=14, textColor=DARK_TEXT,
        leftIndent=18, spaceAfter=3,
    ))
    ss.add(ParagraphStyle(
        "SmallCode", parent=ss["Normal"],
        fontSize=8.5, leading=11, textColor=HexColor("#333333"),
        fontName="Courier", leftIndent=12, spaceAfter=4,
    ))
    ss.add(ParagraphStyle(
        "Caption", parent=ss["Normal"],
        fontSize=9, leading=12, textColor=MEDIUM_GRAY,
        alignment=TA_CENTER, spaceAfter=10,
    ))
    ss.add(ParagraphStyle(
        "NodeTitle", parent=ss["Heading3"],
        fontSize=12, leading=16, textColor=TEAL,
        spaceBefore=10, spaceAfter=4,
    ))
    return ss


# ── Flowchart Diagram ────────────────────────────────────────

def _draw_pipeline_diagram():
    """
    Draw the 10-step pipeline flowchart as a vector Drawing.
    Two columns, 5 rows, with arrows connecting them in sequence.
    """
    w, h = 500, 420
    d = Drawing(w, h)

    # Background
    d.add(Rect(0, 0, w, h, fillColor=HexColor("#fafbfc"), strokeColor=None))

    # Title
    d.add(String(w / 2, h - 18, "DYN-EYE Pipeline Flow",
                 fontSize=14, fillColor=PRIMARY, textAnchor="middle",
                 fontName="Helvetica-Bold"))

    nodes = [
        ("1. YOLO Inference\n(Known/Unknown Split)", NODE_COLORS[0]),
        ("2. Dataset Context\n(Groq Dynamic Prompt)", NODE_COLORS[1]),
        ("3. VLM Annotation\n(Gemma 4-31b-it)", NODE_COLORS[2]),
        ("4. Crop Extraction\n(Defect Region Cut)", NODE_COLORS[3]),
        ("5. DINOv2 Features\n(384-d Embeddings)", NODE_COLORS[4]),
        ("6. FAISS Search\n(Novelty Filter)", NODE_COLORS[5]),
        ("7. HDBSCAN Cluster\n(Density Grouping)", NODE_COLORS[6]),
        ("8. Manifest + ICC\n(Save & Metrics)", NODE_COLORS[7]),
        ("9. Dashboard UI\n(Human Label Review)", NODE_COLORS[8]),
        ("10. Retraining Agent\n(LLM-Advised Training)", NODE_COLORS[9]),
    ]

    box_w, box_h = 195, 48
    col_x = [40, 265]
    start_y = h - 60
    row_gap = 62

    positions = []
    for i, (label, color) in enumerate(nodes):
        row = i // 2
        col = i % 2
        x = col_x[col]
        y = start_y - row * row_gap

        # Rounded box
        d.add(Rect(x, y - box_h, box_w, box_h,
                    fillColor=color, strokeColor=None,
                    rx=8, ry=8))

        # Label text (split into two lines)
        lines = label.split("\n")
        d.add(String(x + box_w / 2, y - 18, lines[0],
                     fontSize=9.5, fillColor=white, textAnchor="middle",
                     fontName="Helvetica-Bold"))
        if len(lines) > 1:
            d.add(String(x + box_w / 2, y - 32, lines[1],
                         fontSize=8, fillColor=HexColor("#dddddd"),
                         textAnchor="middle", fontName="Helvetica"))

        positions.append((x, y, box_w, box_h))

    # Draw arrows connecting sequential nodes
    arrow_color = HexColor("#555555")
    for i in range(len(nodes) - 1):
        x1, y1, w1, h1 = positions[i]
        x2, y2, w2, h2 = positions[i + 1]

        row_i = i // 2
        col_i = i % 2
        row_next = (i + 1) // 2
        col_next = (i + 1) % 2

        if row_i == row_next:
            # Same row: horizontal arrow from right of left box to left of right box
            ax1 = x1 + w1
            ay1 = y1 - h1 / 2
            ax2 = x2
            ay2 = y2 - h2 / 2
            d.add(Line(ax1, ay1, ax2, ay2, strokeColor=arrow_color, strokeWidth=1.5))
            # Arrowhead pointing right
            d.add(Polygon(
                points=[ax2, ay2, ax2 - 7, ay2 + 4, ax2 - 7, ay2 - 4],
                fillColor=arrow_color, strokeColor=None
            ))
        else:
            # Different row: arrow goes down from right box to left box of next row
            # Down from right column box bottom center
            ax1 = x1 + w1 / 2
            ay1 = y1 - h1
            ax2 = x2 + w2 / 2
            ay2 = y2

            if col_i == 1 and col_next == 0:
                # Right col → left col (next row): diagonal
                d.add(Line(ax1, ay1, ax2, ay2, strokeColor=arrow_color, strokeWidth=1.5))
                d.add(Polygon(
                    points=[ax2, ay2, ax2 - 4, ay2 + 7, ax2 + 4, ay2 + 7],
                    fillColor=arrow_color, strokeColor=None
                ))

    return d


# ── Retraining Diagram ───────────────────────────────────────

def _draw_retraining_diagram():
    """Draw the retraining agent pipeline as a linear flowchart."""
    w, h = 500, 160
    d = Drawing(w, h)

    d.add(Rect(0, 0, w, h, fillColor=HexColor("#fafbfc"), strokeColor=None))
    d.add(String(w / 2, h - 16, "Retraining Agent Flow",
                 fontSize=12, fillColor=PRIMARY, textAnchor="middle",
                 fontName="Helvetica-Bold"))

    nodes = [
        ("Export\nAnnotations", HexColor("#F57F17")),
        ("Validate\nDataset", HexColor("#1565C0")),
        ("DVC\nVersion", HexColor("#2E7D32")),
        ("LLM\nAdvisor", HexColor("#7B1FA2")),
        ("YOLO\nTrain", HexColor("#C62828")),
        ("MLflow\nDeploy", HexColor("#00897B")),
        ("Sync\nRegistry", HexColor("#455A64")),
    ]

    box_w, box_h = 58, 42
    gap = 8
    total_w = len(nodes) * box_w + (len(nodes) - 1) * gap
    start_x = (w - total_w) / 2
    y = h / 2 - 8

    for i, (label, color) in enumerate(nodes):
        x = start_x + i * (box_w + gap)
        d.add(Rect(x, y - box_h / 2, box_w, box_h,
                    fillColor=color, strokeColor=None, rx=5, ry=5))
        lines = label.split("\n")
        d.add(String(x + box_w / 2, y + 2, lines[0],
                     fontSize=7.5, fillColor=white, textAnchor="middle",
                     fontName="Helvetica-Bold"))
        if len(lines) > 1:
            d.add(String(x + box_w / 2, y - 10, lines[1],
                         fontSize=7, fillColor=HexColor("#dddddd"),
                         textAnchor="middle", fontName="Helvetica"))

        if i < len(nodes) - 1:
            ax = x + box_w
            ay = y
            d.add(Line(ax + 1, ay, ax + gap - 1, ay,
                        strokeColor=HexColor("#555555"), strokeWidth=1.2))
            d.add(Polygon(
                points=[ax + gap - 1, ay, ax + gap - 5, ay + 3, ax + gap - 5, ay - 3],
                fillColor=HexColor("#555555"), strokeColor=None
            ))

    return d


# ── Content Builders ─────────────────────────────────────────

def _bullet(text, styles):
    return Paragraph(f"• {text}", styles["BulletText"])


def _node_section(title, subtitle, body_paragraphs, io_table_data, styles):
    """Build a keep-together section for one pipeline node."""
    elements = []
    elements.append(Paragraph(f"<b>{title}</b> — <i>{subtitle}</i>", styles["NodeTitle"]))

    for p in body_paragraphs:
        elements.append(Paragraph(p, styles["BodyText2"]))

    if io_table_data:
        t = Table(io_table_data, colWidths=[55 * mm, 100 * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("LEADING", (0, 0), (-1, -1), 11),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 1), (-1, -1), LIGHT_GRAY),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(Spacer(1, 4))
        elements.append(t)

    elements.append(Spacer(1, 6))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    return KeepTogether(elements)


def build_document():
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    styles = _styles()
    story = []

    # ══════════════════════════════════════════════════════════
    # PAGE 1 — Title & Overview
    # ══════════════════════════════════════════════════════════
    story.append(Spacer(1, 30))
    story.append(Paragraph("DYN-EYE", styles["DocTitle"]))
    story.append(Paragraph("System Architecture Document", styles["DocSubtitle"]))
    story.append(Spacer(1, 10))

    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. System Overview", styles["SectionHead"]))
    story.append(Paragraph(
        "DYN-EYE is an unknown defect discovery and iterative model improvement system "
        "for industrial visual inspection. It takes a set of input images, uses an existing "
        "YOLOv8 object detection model to separate images into known and unknown categories, "
        "then uses a Vision Language Model (VLM) to annotate the unknown images with bounding "
        "boxes around potential defects. Those bounding boxes are cropped out, embedded into "
        "a vector space using DINOv2, filtered for novelty against a FAISS index of known "
        "defects, and then clustered using HDBSCAN. The resulting clusters are presented in "
        "a web dashboard where a human operator names them. Once named, a retraining agent "
        "exports the labels into YOLO format, gets training hyperparameter recommendations "
        "from an LLM, fine-tunes the YOLO model, and deploys the updated model.",
        styles["BodyText2"],
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("1.1 Technology Stack", styles["SubHead"]))
    stack_data = [
        ["Component", "Technology"],
        ["Object Detection", "YOLOv8 (Ultralytics)"],
        ["Vision Language Model", "Gemma 4-31b-it (Google GenAI API)"],
        ["Dynamic Prompt Generation", "Llama-3.3-70b-versatile (Groq API)"],
        ["Feature Embeddings", "DINOv2 ViT-S/14 (Facebook, 384-dim)"],
        ["Nearest-Neighbor Index", "FAISS IndexFlatL2 (Facebook)"],
        ["Clustering", "HDBSCAN + UMAP/PCA + KMeans fallback"],
        ["Pipeline Orchestration", "LangGraph (StateGraph)"],
        ["LLM Training Advisor", "Groq Llama-3.3-70b / Gemini 2.0 Flash"],
        ["Experiment Tracking", "MLflow (local file store)"],
        ["Dataset Versioning", "DVC (Data Version Control)"],
        ["Dashboard", "FastAPI + HTML/CSS/JS"],
        ["Model Registry", "Custom JSON-based local registry"],
    ]
    t = Table(stack_data, colWidths=[55 * mm, 100 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    story.append(Paragraph("1.2 Directory Structure", styles["SubHead"]))
    story.append(Paragraph(
        "<font name='Courier' size='8.5'>"
        "dyn-eye/<br/>"
        "├── config.py                  # All paths, thresholds, API keys<br/>"
        "├── models/best.pt             # Active YOLO model weights<br/>"
        "├── models/versions/           # Versioned model checkpoints<br/>"
        "├── data/<br/>"
        "│   ├── input_images/          # Raw inspection images<br/>"
        "│   ├── crops/                 # VLM-extracted defect crops<br/>"
        "│   ├── clusters/             # HDBSCAN cluster folders + manifest<br/>"
        "│   ├── faiss_index/           # FAISS index + label files<br/>"
        "│   ├── yolo_dataset/          # Exported YOLO train/val split<br/>"
        "│   ├── known_defect_crops/    # Reference crops for FAISS setup<br/>"
        "│   └── known_defects.json     # Known defect class registry<br/>"
        "├── src/pipeline/nodes/        # 8 LangGraph discovery nodes<br/>"
        "├── src/pipeline/graph.py      # Discovery pipeline graph definition<br/>"
        "├── src/retraining/agent.py    # Retraining agent graph definition<br/>"
        "├── src/retraining/llm_advisor.py  # LLM hyperparameter advisor<br/>"
        "├── src/features/             # DINOv2 extractor, FAISS manager<br/>"
        "└── dashboard/                # FastAPI dashboard app<br/>"
        "</font>",
        styles["BodyText2"],
    ))

    # ══════════════════════════════════════════════════════════
    # PAGE 2 — Pipeline Diagram
    # ══════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("2. Discovery Pipeline Diagram", styles["SectionHead"]))
    story.append(Paragraph(
        "The discovery pipeline is a LangGraph StateGraph with 8 nodes executed in a "
        "linear sequence. Each node reads from and writes to a shared state dictionary. "
        "The pipeline is defined in <font name='Courier'>src/pipeline/graph.py</font> "
        "and invoked via <font name='Courier'>run_discovery_pipeline()</font>.",
        styles["BodyText2"],
    ))
    story.append(Spacer(1, 8))
    story.append(_draw_pipeline_diagram())
    story.append(Paragraph(
        "Figure 1: Discovery pipeline flow. Nodes 1–8 run automatically. "
        "Node 9 (Dashboard) requires human interaction to name clusters. "
        "Node 10 (Retraining) is triggered from the dashboard after labelling.",
        styles["Caption"],
    ))

    story.append(Spacer(1, 12))
    story.append(Paragraph("2.1 Retraining Agent Diagram", styles["SubHead"]))
    story.append(Paragraph(
        "The retraining agent is a separate LangGraph StateGraph with 7 nodes. "
        "It contains conditional edges: if dataset validation fails, the pipeline "
        "stops. If training fails, deployment is skipped.",
        styles["BodyText2"],
    ))
    story.append(Spacer(1, 6))
    story.append(_draw_retraining_diagram())
    story.append(Paragraph(
        "Figure 2: Retraining agent flow. Conditional gates after Validate and Train "
        "nodes can halt the pipeline early if validation fails or training produces "
        "no usable model.",
        styles["Caption"],
    ))

    # ══════════════════════════════════════════════════════════
    # PAGE 3+ — Detailed Node Descriptions
    # ══════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("3. Discovery Pipeline — Detailed Node Descriptions", styles["SectionHead"]))
    story.append(Paragraph(
        "Each node below is described exactly as implemented in the source code. "
        "The Input/Output table shows the state keys that each node reads and writes.",
        styles["BodyText2"],
    ))
    story.append(Spacer(1, 6))

    # ── Node 1: YOLO Inference ──
    story.append(_node_section(
        "Node 1: YOLO Inference",
        "src/pipeline/nodes/yolo_inference.py",
        [
            "This node runs the current YOLOv8 model (<font name='Courier'>models/best.pt</font>) "
            "over every image in the input directory. It classifies each image as either "
            "<b>known</b> (the model detected a defect class it was trained on, with confidence "
            "above the threshold) or <b>unknown</b> (no confident known-class detection).",

            "The model is loaded using the Ultralytics YOLO API. Images are processed in "
            "batches of 16. For each image, every detection box is checked: if the predicted "
            "class name (lowercased) matches any entry in the known defects registry and the "
            "confidence score meets the threshold (default 0.30), the image is marked as known.",

            "There is a <b>fallback stage</b>: if YOLO does not confidently detect a known class, "
            "the node checks if any crop file in <font name='Courier'>data/known_defect_crops/</font> "
            "has a filename prefix matching the current image stem. This bootstraps filtering "
            "for known defects before the model is fully fine-tuned on new classes.",

            "The known defect names list is read dynamically from the known defects registry "
            "(<font name='Courier'>data/known_defects.json</font>) at the start of each run, "
            "so newly deployed classes are picked up automatically.",

            "All unknown image paths are saved to <font name='Courier'>data/unknown_defects.json</font> "
            "for downstream use.",
        ],
        [
            ["Direction", "State Keys"],
            ["Reads", "input_images_dir, use_cache"],
            ["Writes", "all_image_paths, known_image_paths, unknown_image_paths, "
                       "yolo_raw_results, known_defect_names"],
        ],
        styles,
    ))

    # ── Node 2: Dataset Context ──
    story.append(_node_section(
        "Node 2: Dataset Context (Dynamic VLM Prompt)",
        "src/pipeline/nodes/dataset_context.py",
        [
            "This node runs after YOLO inference and before VLM annotation. Its purpose "
            "is to generate a <b>domain-specific detection prompt</b> for the VLM, so that "
            "the VLM annotation step uses vocabulary and sensitivity appropriate to the "
            "current inspection domain rather than a generic static prompt.",

            "It first builds a <b>pre-annotation context</b> dictionary containing: the "
            "inspection domain (from config, e.g. 'steel_casting'), the list of known defect "
            "class names, the count of unknown images found by YOLO, the novelty ratio "
            "(unknown count / total count), and <b>closed-loop VARS feedback metrics</b> "
            "from the previous pipeline run (CDS, BQS, DRS). A flag <font name='Courier'>high_novelty</font> "
            "is set to True if the novelty ratio exceeds 0.5.",

            "If previous VARS sub-scores indicate quality bottlenecks (e.g. low BQS for overzoomed "
            "boxes, low CDS for background texture hallucination, or low DRS for imbalanced detection rates), "
            "specific <b>vars_corrective_directives</b> are dynamically injected into the context.",

            "This context is sent to the <b>Groq API</b> (Llama-3.3-70b-versatile model) "
            "along with a meta-prompt that instructs the LLM to write a Gemini-compatible "
            "system prompt. The meta-prompt tells the LLM to include the inspection domain, "
            "list all known classes so the VLM can distinguish new defects from near-misses, "
            "apply any VARS corrective directives to optimize prompt quality, and if high_novelty is true, "
            "instruct the VLM to coin new descriptive labels.",

            "The generated prompt is cached to disk (keyed by a SHA-256 hash of the context) "
            "so repeated runs with identical contexts do not re-call Groq. If Groq is "
            "unavailable (no API key, network error, or the groq package is not installed), "
            "the node silently falls back to the static prompt defined in vlm_annotation.py.",
        ],
        [
            ["Direction", "State Keys"],
            ["Reads", "known_defect_names, unknown_image_paths, all_image_paths"],
            ["Writes", "dataset_context, vlm_system_prompt (optional)"],
        ],
        styles,
    ))

    # ── Node 3: VLM Annotation ──
    story.append(_node_section(
        "Node 3: VLM Annotation",
        "src/pipeline/nodes/vlm_annotation.py",
        [
            "This node sends each unknown image to the <b>Gemma 4-31b-it</b> vision language "
            "model through the Google GenAI API for bounding-box defect detection. Images are "
            "processed <b>one at a time</b> (sequentially, not batched) with a configurable "
            "sleep interval between calls (default 4.5 seconds) to respect API rate limits.",

            "The VLM is called with <font name='Courier'>response_mime_type='application/json'</font> "
            "and a Pydantic response schema (<font name='Courier'>InspectionReport</font>) that "
            "enforces the output format. The expected JSON has: "
            "<font name='Courier'>anomalies_found</font> (bool) and "
            "<font name='Courier'>findings</font> (list of objects, each with "
            "<font name='Courier'>box_2d</font> as [ymin, xmin, ymax, xmax] in 0–1000 scale, "
            "and <font name='Courier'>physical_traits</font> as a text description).",

            "If the dynamic prompt was generated by the Dataset Context node, it is used. "
            "Otherwise the node falls back to a hardcoded static system prompt that instructs "
            "the VLM to detect structural/geometric violations, surface/texture violations, "
            "and tonal/material violations, with tight bounding boxes (5–8% margin).",

            "Each image call has retry logic with exponential backoff (up to 5 retries). "
            "On 503 errors or network failures, the backoff is extended. If all retries fail "
            "for an image, it is recorded with an error flag and empty findings.",

            "<b>Post-Processing Box Merging:</b> Raw bounding boxes undergo an automated "
            "overlap consolidation filter. If two predicted boxes overlap with an Intersection-over-Union "
            "(IoU) ≥ 0.30 or Intersection-over-Smaller-Area (IoS) ≥ 0.50 (nested boxes), they are "
            "merged into a unified bounding envelope <font name='Courier'>[min_y, min_x, max_y, max_x]</font> "
            "with combined physical traits. This eliminates redundant overlapping crops prior to feature extraction.",

            "The VLM temperature is set to 0.1 for maximum determinism. All annotations are "
            "automatically cached to <font name='Courier'>data/vlm_cache.json</font> after "
            "processing completes, so a subsequent run with <font name='Courier'>use_cache=True</font> "
            "can skip the VLM entirely.",
        ],
        [
            ["Direction", "State Keys"],
            ["Reads", "unknown_image_paths, use_cache, vlm_system_prompt"],
            ["Writes", "vlm_annotations (list of dicts with image_path, findings, etc.)"],
        ],
        styles,
    ))

    # ── Node 4: Crop Extraction ──
    story.append(_node_section(
        "Node 4: Crop Extraction",
        "src/pipeline/nodes/crop_extraction.py",
        [
            "This node takes the bounding box annotations from the VLM and physically "
            "crops those regions out of the original images using OpenCV. Each crop is "
            "saved as a JPEG file in <font name='Courier'>data/crops/</font> with a naming "
            "convention of <font name='Courier'>{image_stem}_crop_{index:04d}.jpg</font>.",

            "The VLM returns box coordinates in a 0–1000 scale relative to image dimensions. "
            "This node converts them to pixel coordinates by multiplying by the actual image "
            "height and width, then dividing by 1000. Coordinates are clamped to image boundaries.",

            "Three quality filters are applied to each crop before saving:<br/>"
            "<b>1. Minimum size guard</b>: Crops smaller than 30×30 pixels (or area < 900 px²) "
            "are discarded as too small to be meaningful.<br/>"
            "<b>2. Overzoom guard</b>: If a crop covers more than 85% of the original image in "
            "both dimensions, it is discarded. This catches a common VLM failure mode where the "
            "model draws a box around the entire image.<br/>"
            "<b>3. Blur guard</b>: The Laplacian variance of the grayscale crop is computed. "
            "If it falls below 25.0, the crop is considered too blurry or uninformative and is "
            "discarded.",

            "Metadata for each saved crop (source image path, bounding box in both raw and pixel "
            "coordinates, physical traits description, crop dimensions) is collected and passed "
            "downstream for manifest generation.",
        ],
        [
            ["Direction", "State Keys"],
            ["Reads", "vlm_annotations, use_cache"],
            ["Writes", "crop_paths (list of file paths), crop_metadata (list of dicts)"],
        ],
        styles,
    ))

    # ── Node 5: DINOv2 Feature Extraction ──
    story.append(_node_section(
        "Node 5: DINOv2 Feature Extraction",
        "src/pipeline/nodes/feature_extraction.py",
        [
            "This node encodes each crop image into a <b>384-dimensional feature vector</b> "
            "using the DINOv2 ViT-S/14 model (loaded from PyTorch Hub, Facebook Research). "
            "DINOv2 is a self-supervised vision transformer that produces dense, semantically "
            "meaningful embeddings without requiring task-specific fine-tuning.",

            "Each image is preprocessed by resizing to 224×224, converting to a tensor, and "
            "normalizing with ImageNet statistics (mean=[0.485, 0.456, 0.406], "
            "std=[0.229, 0.224, 0.225]). Images are processed in batches (default batch size "
            "32) for efficiency.",

            "The output vectors are <b>L2-normalized</b> using "
            "<font name='Courier'>torch.nn.functional.normalize(features, p=2, dim=-1)</font> "
            "so that cosine similarity between any two vectors equals their dot product. This "
            "normalization is critical for the FAISS distance computation and HDBSCAN clustering "
            "that follow.",

            "The model runs on GPU if CUDA is available, otherwise CPU. All inference is done "
            "under <font name='Courier'>@torch.no_grad()</font> to avoid gradient computation.",

            "The final output is a NumPy array of shape (N, 384) where N is the number of crops.",
        ],
        [
            ["Direction", "State Keys"],
            ["Reads", "crop_paths"],
            ["Writes", "feature_vectors (N×384 numpy array), feature_crop_paths"],
        ],
        styles,
    ))

    # ── Node 6: FAISS Search ──
    story.append(_node_section(
        "Node 6: FAISS Novelty Search",
        "src/pipeline/nodes/faiss_search.py",
        [
            "This node queries every crop's 384-d embedding against a pre-built <b>FAISS "
            "IndexFlatL2</b> index of known defect embeddings. The index contains embeddings "
            "of reference crops stored in <font name='Courier'>data/known_defect_crops/</font>, "
            "organized by class subdirectory (e.g. <font name='Courier'>scratch/</font>, "
            "<font name='Courier'>dent/</font>).",

            "For each query vector, FAISS returns the L2 distance to its single nearest neighbor "
            "(k=1 search). If this distance exceeds the novelty threshold (default 0.35 in L2 "
            "space of L2-normalized vectors), the crop is flagged as <b>novel</b> — meaning it "
            "does not closely resemble any known defect type.",

            "The threshold of 0.35 on L2-normalized vectors corresponds roughly to a cosine "
            "similarity of 0.825. This was tuned to balance between catching truly new defect "
            "types and not flooding the system with minor variations of known defects.",

            "If no FAISS index file exists on disk (first run before FAISS setup), all crops "
            "are treated as novel. This is a safe default that ensures the system works out of "
            "the box even without a pre-built reference database.",

            "The FAISS index is rebuilt automatically after every successful model deployment "
            "or rollback, ensuring it stays synchronized with the deployed model's class set.",
        ],
        [
            ["Direction", "State Keys"],
            ["Reads", "feature_vectors, feature_crop_paths"],
            ["Writes", "faiss_distances, faiss_is_novel, novel_indices, total_detected_crops"],
        ],
        styles,
    ))

    # ── Node 7: HDBSCAN Clustering ──
    story.append(_node_section(
        "Node 7: HDBSCAN Clustering",
        "src/pipeline/nodes/hdbscan_cluster.py",
        [
            "This node groups the novel crop embeddings into clusters using density-based "
            "clustering. It is the most complex node in the pipeline with several stages:",

            "<b>Stage 1 — L2 Normalization</b>: Novel feature vectors are L2-normalized "
            "(vectors with zero norm are left as-is to avoid division by zero).",

            "<b>Stage 2 — Dimensionality Reduction</b>: If there are ≥30 crops, UMAP is used "
            "to project the 384-d vectors into a 10-dimensional space "
            "(cosine metric, fixed random_state=42, n_jobs=1 for determinism). "
            "For fewer than 30 crops, PCA is used as a deterministic fallback. "
            "For ≤10 crops, no reduction is applied.",

            "<b>Stage 3 — Adaptive Parameter Tuning</b>: A grid search over "
            "<font name='Courier'>(min_cluster_size, min_samples)</font> is performed. "
            "For each parameter combination, HDBSCAN is run and the resulting clustering is "
            "scored by silhouette coefficient (computed on L2-normalized high-dimensional features, "
            "not the reduced space). The combination with the best silhouette score wins, with "
            "ties broken in favor of more clusters. Candidate values are scaled to the dataset size.",

            "<b>Stage 4 — Clustering Execution</b>: HDBSCAN is run with the tuned parameters "
            "(EOM cluster selection, euclidean metric). If the noise ratio exceeds 85% or no "
            "clusters are found, AgglomerativeClustering is used as a fallback "
            "(distance_threshold=0.65, average linkage). If only a single cluster results and "
            "there are ≥6 crops, a silhouette-optimized KMeans sweep over k=2..8 is run.",

            "<b>Stage 5 — Noise Reassignment</b>: Points labeled as noise (label=-1) are "
            "reassigned to the nearest cluster centroid if their cosine distance is below the "
            "reassignment threshold (default 0.50). Remaining unassigned points are placed in "
            "an 'unassigned' folder.",

            "<b>Stage 6 — Cluster Fingerprint Registry</b>: Each cluster's L2-normalized "
            "centroid is compared against a persistent fingerprint registry "
            "(<font name='Courier'>data/clusters/cluster_registry.json</font>). If a centroid "
            "matches a stored fingerprint within the match threshold (cosine distance < 0.30), "
            "the match count is incremented. Otherwise, a new fingerprint is registered. This "
            "gives clusters persistent identity across pipeline runs.",

            "Crop files are physically copied into cluster subdirectories "
            "(<font name='Courier'>data/clusters/cluster_000/</font>, etc.) for the dashboard.",
        ],
        [
            ["Direction", "State Keys"],
            ["Reads", "feature_vectors, feature_crop_paths, novel_indices"],
            ["Writes", "cluster_labels, cluster_folders, num_clusters, cluster_registry, "
                       "cluster_tuned_params, unassigned_crop_paths, dbcv_score, "
                       "registry_hits, registry_total"],
        ],
        styles,
    ))

    # ── Node 8: Manifest Save + ICC ──
    story.append(_node_section(
        "Node 8: Manifest Save + ICC Metrics",
        "src/pipeline/nodes/manifest_save.py",
        [
            "This is the final node in the discovery pipeline. It produces the "
            "<font name='Courier'>cluster_manifest.json</font> file that the dashboard reads, "
            "and computes cluster quality metrics.",

            "<b>Cohesion Score</b>: For each cluster, the mean cosine similarity between every "
            "member vector and the cluster centroid is computed. This gives a 0–1 score where "
            "higher means the cluster is more visually consistent.",

            "<b>Global ICC (Intra-Class Correlation)</b>: Computed via a one-way ANOVA "
            "decomposition on the DINOv2 embeddings. The between-group sum of squares (SS_between) "
            "and within-group sum of squares (SS_within) are computed, then "
            "<font name='Courier'>ICC = (MS_between - MS_within) / (MS_between + (k₀-1)·MS_within)</font> "
            "where k₀ adjusts for unequal group sizes. This measures how well the clusters "
            "separate distinct defect types.",

            "<b>Global Silhouette Score</b>: Computed on L2-normalized features with cosine "
            "metric (sklearn), excluding noise points. Requires at least 4 non-noise points "
            "across at least 2 clusters.",

            "The manifest JSON maps each cluster folder name to its crops, including: crop file "
            "path, source image path, bounding box coordinates (both raw 0–1000 and pixel), "
            "physical traits text, and crop dimensions. It preserves any previously assigned "
            "defect names from the old manifest.",

            "A separate <font name='Courier'>crop_to_source.json</font> mapping is saved, "
            "which records the normalized YOLO-format bounding box (center x, center y, width, "
            "height) for each crop relative to its source image. This is used during the export "
            "step of the retraining agent.",
        ],
        [
            ["Direction", "State Keys"],
            ["Reads", "cluster_folders, crop_metadata, run_id, vlm_system_prompt, "
                      "feature_vectors, cluster_labels, novel_indices"],
            ["Writes", "vlm_cluster_results"],
        ],
        styles,
    ))

    # ══════════════════════════════════════════════════════════
    # PAGE — Dashboard
    # ══════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("4. Dashboard (Human-in-the-Loop)", styles["SectionHead"]))
    story.append(Paragraph(
        "The dashboard is a FastAPI web application that provides a visual interface for "
        "the human operator. It serves the following functions:",
        styles["BodyText2"],
    ))

    story.append(_bullet(
        "<b>Cluster Review</b>: Displays each cluster as a grid of crop thumbnails. "
        "The operator can visually inspect whether the crops in a cluster actually represent "
        "the same defect type.", styles))
    story.append(_bullet(
        "<b>Cluster Naming</b>: The operator assigns a defect class name to each cluster "
        "(e.g. 'edge_crack', 'surface_pit'). This name is written into the "
        "<font name='Courier'>cluster_manifest.json</font>.", styles))
    story.append(_bullet(
        "<b>Crop Reassignment</b>: Individual crops can be moved between clusters or "
        "removed entirely if they are incorrect detections.", styles))
    story.append(_bullet(
        "<b>Pipeline Controls</b>: Start/stop the discovery pipeline (fresh run or cache mode), "
        "trigger the retraining agent, view live log streaming via Server-Sent Events (SSE).",
        styles))
    story.append(_bullet(
        "<b>Model Registry</b>: View all registered model versions with their metrics "
        "(mAP50, precision, recall), deployment status, and deployment history. "
        "Deploy or rollback to any version with a single click. Deployment automatically "
        "rebuilds the FAISS index and syncs the known defects registry.", styles))
    story.append(_bullet(
        "<b>Run Metrics</b>: Displays per-cluster cohesion scores, global ICC, global "
        "silhouette score, and per-node execution timing.", styles))

    # ══════════════════════════════════════════════════════════
    # PAGE — Retraining Agent
    # ══════════════════════════════════════════════════════════
    story.append(Spacer(1, 10))
    story.append(Paragraph("5. Retraining Agent — Detailed Node Descriptions", styles["SectionHead"]))
    story.append(Paragraph(
        "The retraining agent is a LangGraph StateGraph defined in "
        "<font name='Courier'>src/retraining/agent.py</font>. It is triggered from the "
        "dashboard after the human has named at least one cluster. It has 7 nodes with "
        "conditional edges.",
        styles["BodyText2"],
    ))
    story.append(Spacer(1, 6))

    # ── Export Node ──
    story.append(_node_section(
        "Node R1: Export Annotations",
        "src/retraining/agent.py → export_node",
        [
            "Reads the <font name='Courier'>cluster_manifest.json</font> and identifies all "
            "clusters that have been given a defect name by the human operator. For each named "
            "cluster, it maps the crops back to their original source images using the "
            "<font name='Courier'>crop_to_source.json</font> mapping.",

            "The mapping produces YOLO-format label files: each source image gets a "
            "<font name='Courier'>.txt</font> file with lines of "
            "<font name='Courier'>class_id center_x center_y width height</font> (normalized "
            "0–1 coordinates). Source images are copied into "
            "<font name='Courier'>data/yolo_dataset/images/{train|val}/</font> and label files "
            "into <font name='Courier'>data/yolo_dataset/labels/{train|val}/</font> with an "
            "80/20 train/val split.",

            "A <font name='Courier'>data.yaml</font> file is generated listing the class names "
            "and paths. If no named clusters are found, the node returns an error and the pipeline "
            "terminates.",
        ],
        [
            ["Direction", "State Keys"],
            ["Reads", "(manifest file on disk)"],
            ["Writes", "export_result"],
        ],
        styles,
    ))

    # ── Validate Node ──
    story.append(_node_section(
        "Node R2: Validate Dataset",
        "src/retraining/tools/dataset_validator.py",
        [
            "Validates the exported YOLO dataset for structural correctness: checks that "
            "<font name='Courier'>data.yaml</font> exists and is parseable, that at least one "
            "class is listed, that image files referenced in the dataset exist, and that label "
            "files contain valid YOLO-format annotations (5 space-separated values per line, "
            "class IDs within range, coordinates in 0–1).",

            "An additional safeguard checks that the training split contains at least 1 image. "
            "If the dataset has zero training images (e.g. due to stale absolute paths from a "
            "previous workspace), validation fails immediately with a descriptive error.",

            "If validation fails, the conditional edge routes the pipeline to END, skipping "
            "all subsequent nodes.",
        ],
        [
            ["Direction", "State Keys"],
            ["Reads", "export_result"],
            ["Writes", "validation_result"],
        ],
        styles,
    ))

    # ── DVC Version Node ──
    story.append(_node_section(
        "Node R3: DVC Dataset Versioning",
        "src/retraining/tools/dvc_version.py",
        [
            "Versions the <font name='Courier'>data/yolo_dataset/</font> directory using DVC "
            "(Data Version Control). It runs four shell commands: "
            "<font name='Courier'>dvc add</font> (track the dataset), "
            "<font name='Courier'>git add</font> (stage the .dvc file and .gitignore), "
            "<font name='Courier'>git commit</font> (commit with a timestamped message), and "
            "<font name='Courier'>git tag</font> (create an annotated tag like "
            "<font name='Courier'>dataset-v20260701_120000</font>).",

            "If DVC is not initialized in the project, it runs <font name='Courier'>dvc init</font> "
            "first. Version metadata (tag, timestamp, paths) is saved to a JSON file under "
            "<font name='Courier'>logs/dataset_versions/</font>.",

            "This step is skipped if dataset validation failed in the previous node.",
        ],
        [
            ["Direction", "State Keys"],
            ["Reads", "validation_result"],
            ["Writes", "dvc_result"],
        ],
        styles,
    ))

    # ── LLM Advisor Node ──
    story.append(_node_section(
        "Node R4: LLM Training Advisor",
        "src/retraining/llm_advisor.py",
        [
            "This node queries an LLM to analyze the dataset metadata and recommend training "
            "hyperparameters. It first collects metadata by scanning the named clusters: number "
            "of classes, crops per class, total crops, min/max/avg crops per class, whether the "
            "base model exists, existing known defect names, and the total number of YOLO backbone "
            "layers (detected by loading the model checkpoint).",

            "A detailed prompt is constructed that explains YOLO fine-tuning best practices and "
            "asks the LLM to output a JSON object with: <font name='Courier'>should_train</font> "
            "(bool), <font name='Courier'>reason</font> (str), and a <font name='Courier'>config</font> "
            "object containing: epochs, batch size, image size, learning rate (lr0, lrf), momentum, "
            "weight decay, warmup epochs, patience, optimizer type, cosine LR flag, backbone freeze "
            "depth, and 11 augmentation parameters (mosaic, mixup, degrees, translate, scale, "
            "flipud, fliplr, hsv_h, hsv_s, hsv_v, augment toggle).",

            "The prompt includes a heuristic suggestion for the freeze depth: "
            "<font name='Courier'>freeze = max(0, total_layers - max(1, total_crops // 15))</font>, "
            "and scaling guidelines (e.g. <30 crops → head-only training, >200 crops → allow most "
            "layers to adapt).",

            "The node tries <b>Groq first</b> (Llama-3.3-70b-versatile, JSON mode), then "
            "<b>Gemini</b> (gemini-2.0-flash, JSON response MIME type), then a "
            "<b>heuristic fallback</b> that scales epochs, batch size, learning rate, freeze depth, "
            "and augmentation intensity based on dataset size thresholds (< 50, 50–200, 200–500, "
            "> 500 crops).",

            "The LLM response is validated and merged with defaults. If the LLM returns the "
            "generic default freeze value of 10, a smart override is applied using the heuristic "
            "formula. If the LLM advises against training (e.g. too few samples), the pipeline "
            "skips training.",
        ],
        [
            ["Direction", "State Keys"],
            ["Reads", "(manifest + dataset files on disk)"],
            ["Writes", "llm_recommendation (dict with should_train, reason, config)"],
        ],
        styles,
    ))

    # ── Train Node ──
    story.append(_node_section(
        "Node R5: YOLO Fine-Tuning",
        "src/retraining/tools/train_yolo.py",
        [
            "This node fine-tunes the YOLOv8 model using the Ultralytics training API. "
            "It merges hyperparameters from three sources with this priority: "
            "<b>user-specified overrides</b> (from the dashboard) > "
            "<b>LLM-recommended config</b> > <b>config.py defaults</b>.",

            "Before training starts, Python garbage collection is triggered and the PyTorch "
            "CUDA cache is cleared to free memory. The training is configured with "
            "<font name='Courier'>workers=0</font> (main-thread data loading to avoid Windows "
            "multiprocess memory duplication) and <font name='Courier'>cache=False</font> "
            "(no image caching to RAM).",

            "Custom callbacks are registered on the YOLO model to stream training progress "
            "to the dashboard via the LogStream: <font name='Courier'>on_train_start</font>, "
            "<font name='Courier'>on_train_epoch_start</font>, "
            "<font name='Courier'>on_fit_epoch_end</font> (logs loss values), and "
            "<font name='Courier'>on_train_end</font>.",

            "After training, the best model weights (<font name='Courier'>best.pt</font>) are "
            "copied to both the model versions directory and the active model path. Metrics "
            "(mAP50, mAP50-95, precision, recall) are extracted from the Ultralytics results "
            "object.",

            "The trained model is also registered in the local Model Registry with full metadata "
            "(metrics, training config, source, class names, dataset stats).",
        ],
        [
            ["Direction", "State Keys"],
            ["Reads", "llm_recommendation, epochs, imgsz, batch_size, freeze"],
            ["Writes", "training_result (dict with success, model_path, metrics, training_config)"],
        ],
        styles,
    ))

    # ── Deploy Node ──
    story.append(_node_section(
        "Node R6: MLflow Deployment",
        "src/retraining/tools/mlflow_deploy.py",
        [
            "Registers the fine-tuned model in <b>MLflow</b> (local file-backed tracking store). "
            "It creates a new MLflow run, logs the training metrics, logs the model file as an "
            "artifact, creates a registered model version, and transitions it to the 'Production' "
            "stage (archiving any previous production versions).",

            "The model is also copied to the active model path "
            "(<font name='Courier'>models/best.pt</font>) and to the versions directory. "
            "Deployment metadata (model name, version, run ID, timestamp, metrics) is saved "
            "to <font name='Courier'>logs/deployments/</font>.",

            "The Model Registry's deployment status and deployment history are updated. "
            "If MLflow fails (e.g. missing dependency), the model is still copied locally as "
            "a fallback.",

            "This node is skipped (via conditional edge) if training failed.",
        ],
        [
            ["Direction", "State Keys"],
            ["Reads", "training_result"],
            ["Writes", "deploy_result"],
        ],
        styles,
    ))

    # ── Sync Registry Node ──
    story.append(_node_section(
        "Node R7: Sync Registry + Rebuild FAISS",
        "src/retraining/agent.py → sync_registry_node",
        [
            "This is the final node of the retraining pipeline. It performs three tasks:",

            "<b>1. Register classes from the YOLO model</b>: Loads the newly deployed "
            "<font name='Courier'>best.pt</font> model and reads its class names dictionary. "
            "Any class names not already in the known defects registry are added.",

            "<b>2. Register classes from data.yaml</b>: Reads the "
            "<font name='Courier'>data/yolo_dataset/data.yaml</font> file and merges its class "
            "names into the registry. This catches any classes that might be in the training "
            "data but not in the model (edge case).",

            "<b>3. Rebuild FAISS index</b>: Re-extracts DINOv2 embeddings from all crops in "
            "<font name='Courier'>data/known_defect_crops/</font> and rebuilds the FAISS "
            "IndexFlatL2 index. This ensures the next discovery pipeline run will correctly "
            "identify crops that match the newly learned classes as 'known' rather than 'novel'.",

            "The known defects registry (<font name='Courier'>data/known_defects.json</font>) "
            "is a persistent JSON file that tracks: the sorted list of defect class names, "
            "a version counter, a last-updated timestamp, and a history array of all additions "
            "with their sources. It is thread-safe (uses a threading Lock for writes).",
        ],
        [
            ["Direction", "State Keys"],
            ["Reads", "deploy_result, training_result"],
            ["Writes", "sync_result"],
        ],
        styles,
    ))

    # ══════════════════════════════════════════════════════════
    # PAGE — Key Design Decisions
    # ══════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("6. Key Design Decisions", styles["SectionHead"]))

    story.append(Paragraph("6.1 Why DINOv2 for Embeddings?", styles["SubHead"]))
    story.append(Paragraph(
        "DINOv2 ViT-S/14 was chosen because it produces high-quality visual embeddings "
        "without any task-specific fine-tuning. Since the system is designed to discover "
        "<i>unknown</i> defect types (which by definition have no training labels), a "
        "self-supervised model that learns general visual features is more appropriate than "
        "a supervised model trained on a fixed set of classes. The 384-dimensional output "
        "is compact enough for efficient FAISS indexing while still being discriminative.",
        styles["BodyText2"],
    ))

    story.append(Paragraph("6.2 Why HDBSCAN over KMeans?", styles["SubHead"]))
    story.append(Paragraph(
        "HDBSCAN does not require specifying the number of clusters in advance. Since the "
        "system discovers unknown defect types, the number of distinct defect classes in a "
        "batch of images is not known ahead of time. HDBSCAN also naturally identifies noise "
        "points (outliers that do not belong to any cluster), which is useful for filtering "
        "out VLM false positives. However, the system includes KMeans and Agglomerative "
        "Clustering as fallbacks for edge cases (single cluster, very small datasets, high "
        "noise ratio).",
        styles["BodyText2"],
    ))

    story.append(Paragraph("6.3 Why LLM-Advised Hyperparameters?", styles["SubHead"]))
    story.append(Paragraph(
        "The training hyperparameters (learning rate, freeze depth, augmentation intensity, "
        "epoch count) should adapt to the dataset size and composition. A fixed configuration "
        "would either overfit small datasets or undertrain large ones. By providing the dataset "
        "metadata to an LLM and asking it to reason about the optimal configuration, the system "
        "can adapt without hardcoding rules for every possible scenario. The heuristic fallback "
        "ensures the system works even without LLM access.",
        styles["BodyText2"],
    ))

    story.append(Paragraph("6.4 Why Two Separate LangGraph Pipelines?", styles["SubHead"]))
    story.append(Paragraph(
        "The discovery pipeline and the retraining agent are separate graphs because they have "
        "different trigger conditions: discovery runs over new input images, while retraining "
        "runs only after a human has labeled clusters. Separating them also means a retraining "
        "failure does not block future discovery runs, and the human labeling step naturally "
        "forms the boundary between the two workflows.",
        styles["BodyText2"],
    ))

    story.append(Paragraph("6.5 Known Defects Registry as Single Source of Truth", styles["SubHead"]))
    story.append(Paragraph(
        "Rather than hardcoding defect class names in config, the system maintains a persistent "
        "JSON registry that grows over time. Every YOLO inference node reads from this registry "
        "at run time, so a newly deployed model's classes are immediately recognized as 'known' "
        "in the next discovery run. The registry is updated from three sources: the deployed "
        "YOLO model's class dictionary, the training data.yaml, and manual additions via the "
        "dashboard.",
        styles["BodyText2"],
    ))

    # ══════════════════════════════════════════════════════════
    # PAGE — Configuration Reference
    # ══════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("7. Configuration Reference", styles["SectionHead"]))
    story.append(Paragraph(
        "Key configuration values defined in <font name='Courier'>config.py</font>:",
        styles["BodyText2"],
    ))

    config_data = [
        ["Parameter", "Default Value", "Purpose"],
        ["YOLO_CONFIDENCE_THRESHOLD", "0.30", "Min confidence for YOLO known-class detection"],
        ["VLM_MODEL_ID", "gemma-4-31b-it", "Vision Language Model for annotation"],
        ["VLM_TEMPERATURE", "0.1", "VLM generation temperature (low = deterministic)"],
        ["VLM_SLEEP_BETWEEN", "4.5s", "Delay between VLM API calls (rate limiting)"],
        ["VLM_MAX_RETRIES", "5", "Max retry attempts per VLM call"],
        ["FEATURE_DIM", "384", "DINOv2 ViT-S/14 embedding dimension"],
        ["FAISS_NOVELTY_THRESHOLD", "0.35", "L2 distance above which = novel"],
        ["HDBSCAN_MIN_CLUSTER_SIZE", "3", "Minimum cluster size for HDBSCAN"],
        ["HDBSCAN_METRIC", "euclidean", "Distance metric for HDBSCAN"],
        ["CLUSTER_MATCH_THRESHOLD", "0.30", "Cosine dist for fingerprint matching"],
        ["CLUSTER_NOISE_REASSIGN", "0.50", "Cosine dist for noise reassignment"],
        ["UMAP_N_COMPONENTS", "10", "Target dimensions after UMAP"],
        ["UMAP_BATCH_THRESHOLD", "30", "Min crops to activate UMAP over PCA"],
        ["LLM_MODEL_ID", "qwen/qwen3.8-27b", "Groq LLM: advisor, retrain decision, optional dynamic prompt"],
        ["LLM_MIN_CROPS_PER_CLASS", "10", "Min crops before LLM considers training"],
    ]
    t = Table(config_data, colWidths=[48 * mm, 35 * mm, 72 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)

    # ── Build ────────────────────────────────────────────────
    doc.build(story)
    print(f"\nPDF generated: {OUTPUT_PATH}")
    print(f"  Pages: ~10")
    print(f"  Sections: 7 (Overview, Diagrams, Discovery Nodes, Dashboard, Retraining Nodes, Design Decisions, Config)")


if __name__ == "__main__":
    build_document()
