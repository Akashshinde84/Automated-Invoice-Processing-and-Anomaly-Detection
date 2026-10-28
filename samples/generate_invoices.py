"""Generate matching sample PNG invoices and a multipage PDF, without OCR.

Run: .venv/Scripts/python.exe samples/generate_invoices.py
Outputs live beside the tracked, manually authored transcripts in phase3/.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    root = Path(__file__).resolve().parent / "phase3"
    candidates = [Path("C:/Windows/Fonts/consola.ttf"), Path("C:/Windows/Fonts/arial.ttf")]
    font_path = next((path for path in candidates if path.is_file()), None)
    font = ImageFont.truetype(str(font_path), 32) if font_path else ImageFont.load_default(size=32)
    for transcript in sorted(root.glob("*.txt")):
        image = Image.new("RGB", (2100, 1500), "white")
        draw = ImageDraw.Draw(image)
        for index, line in enumerate(transcript.read_text(encoding="utf-8").splitlines()):
            draw.text((100, 120 + index * 85), line, fill="black", font=font)
        target = transcript.with_suffix(".png")
        if not target.exists():
            image.save(target)
            print(f"Created {target}")
        if transcript.stem == "acme":
            # Two pages from one invoice: metadata first, table and totals second.
            pages = []
            lines = transcript.read_text(encoding="utf-8").splitlines()
            for chunk in (lines[:4], lines[4:]):
                page = Image.new("RGB", (2100, 1500), "white")
                page_draw = ImageDraw.Draw(page)
                for index, line in enumerate(chunk):
                    page_draw.text((100, 120 + index * 85), line, fill="black", font=font)
                pages.append(page)
            pdf = root / "acme_multipage.pdf"
            if not pdf.exists():
                pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=150)
                print(f"Created {pdf}")


if __name__ == "__main__":
    main()
