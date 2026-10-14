"""Regenerate sample_docs/aero_x2_spec_sheet.pdf (needs `pip install fpdf2`; not a runtime dependency)."""

from pathlib import Path

from fpdf import FPDF

PAGES = [
    ("Velora Aero X2 - Product Specification", [
        ("Overview", "The Aero X2 is Velora's lightweight urban e-bike, designed for daily commuting "
         "and weekend rides. It launched in January 2026 and replaces the Aero X1."),
        ("Motor", "The Aero X2 uses a 250 W mid-drive motor with 80 Nm of peak torque. Pedal assistance "
         "is provided up to 25 km/h, in line with EU regulations. Four assist modes are available: "
         "Eco, Tour, Sport and Turbo."),
        ("Battery", "The integrated battery has a capacity of 540 Wh and can be removed with a key for "
         "charging indoors. With the included 4 A charger, a full charge from empty takes 4.5 hours; "
         "charging from empty to 80% takes 2.5 hours."),
        ("Range", "Range is up to 95 km in Eco mode, about 70 km in Tour mode and about 45 km in Turbo "
         "mode, measured with a 75 kg rider on flat terrain at 20 degrees C. Cold weather, hills and "
         "headwind reduce range."),
    ]),
    ("Velora Aero X2 - Frame, Weight and Pricing", [
        ("Frame and weight", "The frame is made of hydroformed 6061 aluminium. The complete bike weighs "
         "21.4 kg including the battery. The maximum permitted total weight (rider plus luggage) is "
         "120 kg. The rear rack carries up to 25 kg."),
        ("Weather resistance", "The motor, battery and display are rated IPX5: they are protected "
         "against water jets and riding in heavy rain, but must not be submerged or cleaned with a "
         "pressure washer."),
        ("Sizes and colours", "Available in frame sizes S, M and L, and in three colours: Graphite, "
         "Fjord Blue and Sand."),
        ("Price", "The recommended retail price is EUR 3,290 including VAT in the Netherlands. "
         "Prices in other countries may differ."),
    ]),
]


def build(out: Path) -> None:
    pdf = FPDF(format="A4")
    pdf.set_margins(22, 22, 22)
    for title, sections in PAGES:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.multi_cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        for heading, body in sections:
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 7, heading, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, body, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)
    pdf.output(str(out))


if __name__ == "__main__":
    build(Path(__file__).resolve().parent.parent / "sample_docs" / "aero_x2_spec_sheet.pdf")
