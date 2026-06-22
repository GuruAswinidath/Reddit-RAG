def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[str]:
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    separators = ["\n\n", "\n", ". ", " "]
    return _recursive_split(
        text, separators, chunk_size,
        chunk_overlap,
    )


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    sep = separators[0] if separators else " "
    remaining_seps = (
        separators[1:] if len(separators) > 1
        else []
    )

    parts = text.split(sep)

    chunks = []
    current = ""

    for part in parts:
        candidate = (
            f"{current}{sep}{part}"
            if current else part
        )

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())

            if len(part) > chunk_size:
                if remaining_seps:
                    sub = _recursive_split(
                        part, remaining_seps,
                        chunk_size, chunk_overlap,
                    )
                    chunks.extend(sub)
                    current = ""
                else:
                    for i in range(
                        0, len(part), chunk_size
                    ):
                        chunks.append(
                            part[i:i + chunk_size]
                                .strip()
                        )
                    current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current.strip())

    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _add_overlap(
            chunks, chunk_overlap
        )

    return [c for c in chunks if c]


def _add_overlap(
    chunks: list[str],
    overlap: int,
) -> list[str]:
    result = [chunks[0]]

    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        overlap_text = prev[-overlap:]

        space_idx = overlap_text.find(" ")
        if space_idx != -1:
            overlap_text = (
                overlap_text[space_idx + 1:]
            )

        result.append(
            f"{overlap_text} {chunks[i]}"
        )

    return result
