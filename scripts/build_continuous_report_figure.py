from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/continuous_online_seed43/continuous_online_class_diffusion_alternative_1.gif"
OUTPUT = ROOT / "deliverables/GestureGraph_Complete_Report_2026-09-02/figures/continuous_online_recovery_plate.png"
FRAMES = (
    (15, "a", "Frame 16: early error (Tap)"),
    (36, "b", "Frame 37: persistent error (CCW rotation)"),
    (37, "c", "Frame 38: recovered (Swipe right)"),
    (44, "d", "Frame 45: stable early decision"),
)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def main() -> None:
    gif = Image.open(SOURCE)
    canvas = Image.new("RGB", (1620, 1050), "#F5F7F9")
    draw = ImageDraw.Draw(canvas)
    positions = ((20, 70), (820, 70), (20, 570), (820, 570))
    for (frame_index, panel, caption), (left, top) in zip(FRAMES, positions):
        gif.seek(frame_index)
        frame = gif.convert("RGB").resize((780, 457), Image.Resampling.LANCZOS)
        canvas.paste(frame, (left, top))
        draw.rounded_rectangle((left + 12, top + 12, left + 52, top + 52), radius=20, fill="#16324F")
        draw.text((left + 27, top + 18), panel, anchor="ma", font=load_font(23, True), fill="white")
        draw.text((left + 4, top - 40), caption, font=load_font(23, True), fill="#1F2933")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, dpi=(220, 220))
    print(OUTPUT)


if __name__ == "__main__":
    main()
