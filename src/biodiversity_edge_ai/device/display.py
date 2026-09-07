"""Small adapter around the Waveshare ST7789 display driver."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from biodiversity_edge_ai.pipeline import Prediction


def render_result(
    image: Image.Image,
    prediction: Prediction,
    *,
    inference_ms: float | None = None,
    location_used: bool = False,
) -> Image.Image:
    canvas = image.convert("RGB")
    width, height = canvas.size
    side = min(width, height)
    canvas = canvas.crop(((width - side) // 2, (height - side) // 2, (width + side) // 2, (height + side) // 2))
    canvas = canvas.resize((240, 240), Image.Resampling.BILINEAR).convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((0, 188, 240, 240), fill=(0, 0, 0, 220))
    label = prediction.name[:28]
    draw.text((7, 192), label, font=font, fill=(255, 255, 255, 255))
    context = "vision + geo" if location_used else "vision only"
    timing = f"  {inference_ms:.0f} ms" if inference_ms is not None else ""
    draw.text(
        (7, 216),
        f"{prediction.score * 100:.1f}%  {context}{timing}",
        font=font,
        fill=(160, 255, 180, 255),
    )
    return Image.alpha_composite(canvas, overlay).convert("RGB")


class WaveshareST7789Display:
    """Hardware display wrapper. Importing it does not initialize GPIO."""

    def __init__(self, brightness: int = 50) -> None:
        from .waveshare.ST7789 import ST7789

        self.driver = ST7789()
        self.driver.Init()
        self.driver.clear()
        self.driver.bl_DutyCycle(brightness)

    def show(self, image: Image.Image) -> None:
        self.driver.ShowImage(image.resize((240, 240)).rotate(270))

    def close(self) -> None:
        self.driver.clear()
        self.driver.module_exit()
