import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import os

# ---------- helpers ----------

GREEN = "#318f36"
WHITE = "#ffffff"
LINE = "#444444"
BG = "#f5f5f5"
MINT_BG = "#d4f1e8"

def draw_box(ax, xy, w, h, text, facecolor, edgecolor="#333333",
             fontsize=13.5, bold=False, textcolor=None):
    x, y = xy
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="square,pad=0.02",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2
    )
    ax.add_patch(rect)

    # Default text color: white for green boxes, black for others
    if textcolor is None:
        textcolor = WHITE if facecolor == GREEN else "#000000"

    ax.text(
        x + w/2,
        y + h/2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=textcolor
    )

    return rect

def draw_image_box(ax, xy, w, h, image_path, label_text):
    """Draw a box with an image and label text below it."""
    x, y = xy
    
    # Draw border box
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="square,pad=0.02",
        linewidth=1.5,
        edgecolor="#333333",
        facecolor=WHITE,
        zorder=2
    )
    ax.add_patch(rect)
    
    # Load and add image if file exists
    if os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            # Calculate image position (centered in upper portion of box)
            img_height_ratio = 0.7  # Use 70% of box height for image
            imagebox = OffsetImage(img, zoom=0.08)
            ab = AnnotationBbox(imagebox, (x + w/2, y + h*0.6),
                                frameon=False, zorder=3)
            ax.add_artist(ab)
            
            # Add label text below image
            ax.text(
                x + w/2,
                y + h*0.2,
                label_text,
                ha="center",
                va="center",
                fontsize=13.5,
                fontweight="bold"
            )
        except Exception as e:
            # If image fails to load, just show text
            ax.text(
                x + w/2,
                y + h/2,
                label_text,
                ha="center",
                va="center",
                fontsize=13.5,
                fontweight="bold"
            )
    else:
        # If image doesn't exist, just show text
        ax.text(
            x + w/2,
            y + h/2,
            label_text,
            ha="center",
            va="center",
            fontsize=13.5,
            fontweight="bold"
        )
    
    return rect

def arrow_right(ax, start_box, end_box):
    """Draw a horizontal arrow from the right side of start_box to left of end_box."""
    sx = start_box.get_x() + start_box.get_width()
    ex = end_box.get_x()
    y = start_box.get_y() + start_box.get_height()/2
    
    # Add small gaps so arrow doesn't touch boxes
    gap = 0.03
    arrow = FancyArrowPatch(
        (sx + gap, y),
        (ex - gap, y),
        arrowstyle="->",
        mutation_scale=22,
        linewidth=3.0,
        color=LINE,
        zorder=3,
    )
    ax.add_patch(arrow)

def arrow_down(ax, start_box, end_box):
    """Draw an arrow with 90-degree turns from bottom of start_box to top of end_box."""
    # Start point: bottom center of start box
    sx = start_box.get_x() + start_box.get_width()/2
    sy = start_box.get_y()
    
    # End point: top center of end box
    ex = end_box.get_x() + end_box.get_width()/2
    ey = end_box.get_y() + end_box.get_height()
    
    # Create path with 90-degree turns
    gap = 0.03
    mid_y = (sy + ey) / 2
    
    # Vertical down from start box
    arrow1 = FancyArrowPatch(
        (sx, sy - gap),
        (sx, mid_y),
        arrowstyle="-",
        linewidth=2.5,
        color=LINE,
        zorder=3,
    )
    ax.add_patch(arrow1)
    
    # Horizontal segment
    arrow2 = FancyArrowPatch(
        (sx, mid_y),
        (ex, mid_y),
        arrowstyle="-",
        linewidth=2.5,
        color=LINE,
        zorder=3,
    )
    ax.add_patch(arrow2)
    
    # Vertical down to end box with arrowhead
    arrow3 = FancyArrowPatch(
        (ex, mid_y),
        (ex, ey + gap),
        arrowstyle="->",
        mutation_scale=20,
        linewidth=2.5,
        color=LINE,
        zorder=3,
    )
    ax.add_patch(arrow3)

def arrow_down_direct(ax, start_box, end_box):
    """Draw a direct vertical arrow from bottom of start_box to top of end_box."""
    # Start point: bottom center of start box
    sx = start_box.get_x() + start_box.get_width()/2
    sy = start_box.get_y()
    
    # End point: top center of end box
    ex = end_box.get_x() + end_box.get_width()/2
    ey = end_box.get_y() + end_box.get_height()
    
    gap = 0.03
    
    # Direct vertical arrow
    arrow = FancyArrowPatch(
        (sx, sy - gap),
        (ex, ey + gap),
        arrowstyle="->",
        mutation_scale=20,
        linewidth=2.5,
        color=LINE,
        zorder=3,
    )
    ax.add_patch(arrow)

def arrow_up_then_across(ax, start_box, end_box):
    """Draw an arrow from top of start_box with 90-degree turns to bottom of end_box."""
    # Start point: top center of start box
    sx = start_box.get_x() + start_box.get_width()/2
    sy = start_box.get_y() + start_box.get_height()
    
    # End point: bottom center of end box
    ex = end_box.get_x() + end_box.get_width()/2
    ey = end_box.get_y()
    
    gap = 0.03
    mid_y = (sy + ey) / 2
    
    # Vertical up from start box
    arrow1 = FancyArrowPatch(
        (sx, sy + gap),
        (sx, mid_y),
        arrowstyle="-",
        linewidth=2.5,
        color=LINE,
        zorder=3,
    )
    ax.add_patch(arrow1)
    
    # Horizontal segment
    arrow2 = FancyArrowPatch(
        (sx, mid_y),
        (ex, mid_y),
        arrowstyle="-",
        linewidth=2.5,
        color=LINE,
        zorder=3,
    )
    ax.add_patch(arrow2)
    
    # Vertical down to end box with arrowhead
    arrow3 = FancyArrowPatch(
        (ex, mid_y),
        (ex, ey - gap),
        arrowstyle="->",
        mutation_scale=20,
        linewidth=2.5,
        color=LINE,
        zorder=3,
    )
    ax.add_patch(arrow3)

# ---------- figure setup ----------

fig, ax = plt.subplots(figsize=(24, 14))
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 24)
ax.set_ylim(0, 16)
ax.axis("off")

box_w = 2.2
box_h = 1.6
x0 = 1.2
y1 = 10.5     # row for Approach 1 flow
y2 = 4.0     # row for Approach 2 flow
x_gap = 0.8  # gap between boxes

# ---------- Mint green background rectangles ----------

# Background for Approach 1
bg1 = FancyBboxPatch(
    (0.8, 8.2), 22.4, 6.9,
    boxstyle="round,pad=0.1",
    linewidth=1.5,
    edgecolor="#cccccc",
    facecolor="#f8f8f8",
    zorder=0
)
ax.add_patch(bg1)

# Background for Approach 2
bg2 = FancyBboxPatch(
    (0.8, 0.8), 22.4, 6.9,
    boxstyle="round,pad=0.1",
    linewidth=1.5,
    edgecolor="#cccccc",
    facecolor="#f8f8f8",
    zorder=0
)
ax.add_patch(bg2)

# ---------- Approach 1 ----------

ax.text(
    12, 14.8,
    "Approach 1: Structured Input to Question Generation",
    ha="center", va="center", fontsize=13.5, fontweight="bold"
)

# boxes left-to-right
a1_boxes = []

# Use image for videos if available, otherwise text box
bodycam_img = '/Users/alexli/MQP Stuff/bodycam_frame.png'
a1_video = draw_image_box(
    ax, (x0, y1), box_w, box_h,
    bodycam_img,
    "Videos\n(1 min clips)"
)
a1_boxes.append(a1_video)

a1_boxes.append(
    draw_box(
        ax, (x0 + (box_w + x_gap), y1), box_w, box_h,
        "Script +\nGemini 2.5 Pro", WHITE, bold=False
    )
)

a1_structured = draw_box(
    ax, (x0 + 2*(box_w + x_gap), y1), box_w, box_h,
    "Structured Key\nDetails Format", GREEN, WHITE, bold=True
)
a1_boxes.append(a1_structured)

# Gold questions box (training data creation)
a1_gold_train = draw_box(
    ax, (x0 + 4.5*(box_w + x_gap), y1 + 2.0), box_w, box_h,
    "Gold Questions\n(Gemini 2.5 Pro)", WHITE, bold=False
)

a1_boxes.append(
    draw_box(
        ax, (x0 + 3*(box_w + x_gap), y1), box_w, box_h,
        "Train Question\nGeneration LLM", GREEN, WHITE, bold=True
    )
)

a1_generated = draw_box(
    ax, (x0 + 4*(box_w + x_gap), y1), box_w, box_h,
    "Generated\nQuestions", GREEN, WHITE, bold=True
)
a1_boxes.append(a1_generated)

# evaluation metrics box for approach 1
eval1 = draw_box(
    ax,
    (x0 + 4.5*(box_w + x_gap), y1 - 2.2),
    2.6, box_h,
    "Evaluation\n(F1, BERT Score)",
    WHITE,
    bold=False
)

# arrows along the main chain
for b1, b2 in zip(a1_boxes[:-1], a1_boxes[1:]):
    arrow_right(ax, b1, b2)

# Arrow from structured format to gold questions (training pair creation) - starts from TOP
arrow_up_then_across(ax, a1_structured, a1_gold_train)

# Arrow from generated questions to eval metrics - 90-degree routing to center
arrow_down(ax, a1_generated, eval1)

# Arrow from gold questions (training) to eval - straight down
sx = a1_gold_train.get_x() + a1_gold_train.get_width()/2
sy = a1_gold_train.get_y()
ex = eval1.get_x() + eval1.get_width()/2
ey = eval1.get_y() + eval1.get_height()
gap = 0.03
arrow = FancyArrowPatch(
    (sx, sy - gap),
    (sx, ey + gap),
    arrowstyle="->",
    mutation_scale=20,
    linewidth=2.5,
    color=LINE,
    zorder=3,
)
ax.add_patch(arrow)



# ---------- Approach 2 ----------

ax.text(
    12, 7.6,
    "Approach 2: Masked VLM Summary to Question Generation",
    ha="center", va="center", fontsize=13.5, fontweight="bold"
)

a2_boxes = []

# Use image for videos if available, otherwise text box
bodycam_img = '/Users/alexli/MQP Stuff/bodycam_frame.png'
a2_video = draw_image_box(
    ax, (x0, y2), box_w, box_h,
    bodycam_img,
    "Videos\n(1 min clips)"
)
a2_boxes.append(a2_video)

a2_boxes.append(
    draw_box(
        ax, (x0 + (box_w + x_gap), y2), box_w, box_h,
        "VLM\nSummary\nScript", WHITE, bold=False
    )
)

a2_boxes.append(
    draw_box(
        ax, (x0 + 2*(box_w + x_gap), y2), box_w, box_h,
        "VLM\nSummary", GREEN, WHITE, bold=True
    )
)

a2_boxes.append(
    draw_box(
        ax, (x0 + 3*(box_w + x_gap), y2), box_w, box_h,
        "Mask Important\nDetails", WHITE, bold=False
    )
)

a2_masked = draw_box(
    ax, (x0 + 4*(box_w + x_gap), y2), box_w, box_h,
    "Masked VLM\nSummary", GREEN, WHITE, bold=True
)
a2_boxes.append(a2_masked)

# Gold questions for training (per masked section)
a2_gold_train = draw_box(
    ax, (x0 + 6.5*(box_w + x_gap), y2 + 2.0), box_w, box_h,
    "Gold Questions\n(Gemini 2.5 Pro)\nPer\n Masked Section", WHITE, bold=False
)

a2_boxes.append(
    draw_box(
        ax, (x0 + 5*(box_w + x_gap), y2), box_w, box_h,
        "Train Question\nGeneration LLM", GREEN, WHITE, bold=True
    )
)

a2_generated = draw_box(
    ax, (x0 + 6*(box_w + x_gap), y2), box_w, box_h,
    "Generated\nQuestions", GREEN, WHITE, bold=True
)
a2_boxes.append(a2_generated)

# eval metrics for approach 2
eval2 = draw_box(
    ax,
    (x0 + 6.5*(box_w + x_gap), y2 - 2.2),
    2.6, box_h,
    "Evaluation\n(F1, BERT Score)",
    WHITE,
    bold=False
)

# arrows along the main chain
for b1, b2 in zip(a2_boxes[:-1], a2_boxes[1:]):
    arrow_right(ax, b1, b2)

# Arrow from masked summary to gold questions (training pair creation) - starts from TOP
arrow_up_then_across(ax, a2_masked, a2_gold_train)

# Arrow from generated questions to eval metrics - 90-degree routing to center
arrow_down(ax, a2_generated, eval2)

# Arrow from gold questions (training) to eval - straight down
sx = a2_gold_train.get_x() + a2_gold_train.get_width()/2
sy = a2_gold_train.get_y()
ex = eval2.get_x() + eval2.get_width()/2
ey = eval2.get_y() + eval2.get_height()
gap = 0.03
arrow = FancyArrowPatch(
    (sx, sy - gap),
    (sx, ey + gap),
    arrowstyle="->",
    mutation_scale=20,
    linewidth=2.5,
    color=LINE,
    zorder=3,
)
ax.add_patch(arrow)

plt.tight_layout(pad=2.0)
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig('diagram.png', dpi=300, bbox_inches='tight', facecolor=BG)
plt.show()
