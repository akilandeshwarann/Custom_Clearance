"""
step2_chunk.py

Preprocesses raw HTS data (hts_full.json) into search-optimized text chunks.

Solves the "context loss" problem in flat HTS JSON records by:
  1. Maintaining a hierarchy stack keyed by indent level, so every record
     knows its full parent breadcrumb (e.g. "Live horses, asses, mules >
     Horses > Purebred breeding animals").
  2. Inheriting duty rates (general / special / other / additionalDuties)
     from the nearest ancestor when the current record doesn't define its
     own rate.
  3. Tracking the current HTS chapter as records stream past, since chapter
     headers often carry the chapter number but leaf records don't repeat it.

Input : hts_full.json  (flat list of raw USITC HTS records)
Output: hts_chunks.json (list of {id, text, metadata} chunks)
"""

import sys
import json


def _clean(value):
    """Normalize a possibly-missing/empty field to a stripped string or None."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v).strip() for v in value if str(v).strip())
    value = str(value).strip()
    return value if value else None


def _format_or_na(value):
    cleaned = _clean(value)
    return cleaned if cleaned else "N/A"


def _get_indent(record):
    try:
        return int(record.get("indent", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _derive_chapter(htsno, current_chapter):
    """
    Chapter is the first two digits of the HTS number (e.g. '0101.21.00.10'
    -> '01'). Chapter-header / grouping records sometimes carry no usable
    htsno, so we fall back to whatever chapter is currently active.
    """
    if htsno:
        digits = "".join(ch for ch in htsno if ch.isdigit())
        if len(digits) >= 2:
            return digits[:2]
    return current_chapter


def process_hts_data(input_file="hts_full.json", output_file="hts_chunks.json"):
    print(f"Loading raw HTS records from '{input_file}'...")
    with open(input_file, "r", encoding="utf-8") as f:
        raw_records = json.load(f)

    print(f"Loaded {len(raw_records)} raw records. Building hierarchy + chunks...")

    # hierarchy_stack[indent_level] = {
    #     "description": str,
    #     "general": str or None,
    #     "special": str or None,
    #     "other": str or None,
    #     "additionalDuties": str or None,
    # }
    hierarchy_stack = {}
    current_chapter = None
    chunks = []
    chunk_counter = 0
    skipped = 0

    for record in raw_records:
        description = _clean(record.get("description"))
        if description is None:
            # Nothing meaningful to index or use as a breadcrumb node.
            skipped += 1
            continue

        indent = _get_indent(record)
        htsno = _clean(record.get("htsno"))

        current_chapter = _derive_chapter(htsno, current_chapter)

        # --- Duty rate inheritance -------------------------------------
        # Check the record's own values first; if missing, walk the
        # hierarchy stack from the deepest existing ancestor upward.
        own_general = _clean(record.get("general"))
        own_special = _clean(record.get("special"))
        own_other = _clean(record.get("other"))
        own_additional = _clean(record.get("additionalDuties"))

        def inherit(field_name, own_value):
            if own_value is not None:
                return own_value
            for lvl in sorted((l for l in hierarchy_stack if l < indent), reverse=True):
                ancestor_value = hierarchy_stack[lvl].get(field_name)
                if ancestor_value is not None:
                    return ancestor_value
            return None

        general_rate = inherit("general", own_general)
        special_rate = inherit("special", own_special)
        other_rate = inherit("other", own_other)
        additional_duties = inherit("additionalDuties", own_additional)

        # --- Breadcrumb construction -------------------------------------
        # Pop any stack levels at or deeper than the current indent so the
        # stack always reflects the true ancestor path.
        for lvl in [l for l in hierarchy_stack if l >= indent]:
            del hierarchy_stack[lvl]

        ancestor_descriptions = [
            hierarchy_stack[lvl]["description"]
            for lvl in sorted(hierarchy_stack.keys())
        ]
        hierarchy_breadcrumb = " > ".join(ancestor_descriptions) if ancestor_descriptions else description
        full_context_description = (
            " > ".join(ancestor_descriptions + [description])
            if ancestor_descriptions
            else description
        )

        # Push this record onto the stack as a potential parent for deeper
        # indent levels that follow it.
        hierarchy_stack[indent] = {
            "description": description,
            "general": general_rate,
            "special": special_rate,
            "other": other_rate,
            "additionalDuties": additional_duties,
        }

        # Only emit an actual searchable chunk for records that carry a
        # real HTS code. Pure category headers (no htsno) still serve
        # their purpose by being pushed onto the hierarchy stack above,
        # but they don't need their own chunk.
        if not htsno:
            continue

        units_str = _format_or_na(record.get("units"))
        footnotes_str = _format_or_na(record.get("footnotes"))
        quota_str = _format_or_na(record.get("quotaQuantity"))

        chunk_counter += 1
        chunk_id = f"hts_chunk_{chunk_counter}"

        text = (
            f"HTS Code: {htsno}\n"
            f"Chapter: {current_chapter or 'N/A'}\n"
            f"Hierarchy: {hierarchy_breadcrumb}\n"
            f"Description: {description}\n"
            f"Full Context Description: {full_context_description}\n"
            f"General Duty Rate: {general_rate or 'N/A'}\n"
            f"Special Duty Rate: {special_rate or 'N/A'}\n"
            f"Other Duty Rate: {other_rate or 'N/A'}\n"
            f"Unit of Quantity: {units_str}\n"
            f"Footnotes: {footnotes_str}\n"
            f"Quota Quantity: {quota_str}\n"
            f"Additional Duties: {additional_duties or 'N/A'}"
        )

        chunks.append({
            "id": chunk_id,
            "text": text,
            "metadata": {
                "htsno": htsno,
                "description": description,
                "chapter": current_chapter or "N/A",
                "indent": str(indent),
            },
        })

    print(f"Built {len(chunks)} chunks ({skipped} records skipped for missing description).")

    print(f"Writing chunks to '{output_file}'...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Done. '{output_file}' contains {len(chunks)} chunks.")
    return chunks


def main():
    try:
        process_hts_data(input_file="hts_full.json", output_file="hts_chunks.json")
    except FileNotFoundError as e:
        print(f"Error: {e}. Run step1_collect.py first to generate hts_full.json.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: hts_full.json is not valid JSON ({e}).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
