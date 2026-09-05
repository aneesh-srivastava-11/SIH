"""
Dump every unique tag in a PDS4 XML label, with one example value each.

This does NOT print the whole file (which can be long and repetitive) —
just one line per unique tag name, so you can quickly spot which tags
hold sun angle, incidence angle, corners, resolution, etc.

Usage:
    python inspect_label.py path/to/product.xml
    python inspect_label.py path/to/product.xml --filter angle
    python inspect_label.py path/to/product.xml --filter corner
"""

import sys
from lxml import etree


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_label.py path/to/product.xml [--filter TEXT]")
        sys.exit(1)

    xml_path = sys.argv[1]
    filter_text = None
    if "--filter" in sys.argv:
        idx = sys.argv.index("--filter")
        filter_text = sys.argv[idx + 1].lower()

    tree = etree.parse(xml_path)
    root = tree.getroot()

    seen = {}
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]  # strip namespace
        text = (elem.text or "").strip()
        if tag not in seen and text:
            seen[tag] = text

    print(f"{'TAG':45s} EXAMPLE VALUE")
    print("-" * 90)
    for tag, value in sorted(seen.items()):
        if filter_text and filter_text not in tag.lower():
            continue
        display_val = value[:60] + "..." if len(value) > 60 else value
        print(f"{tag:45s} {display_val}")

    print(f"\n{len(seen)} unique tags with text content found.")
    if filter_text:
        matched = sum(1 for t in seen if filter_text in t.lower())
        print(f"({matched} match filter '{filter_text}')")


if __name__ == "__main__":
    main()