"""Document parsing + structure-aware chunking."""
import io
import re
from pypdf import PdfReader
from docx import Document

CATEGORY_KEYWORDS = {
    "vpn": ["vpn", "tunnel", "anyconnect", "openvpn", "wireguard"],
    "network": ["wi-fi", "wifi", "dns", "dhcp", "router", "ethernet", "ip address", "gateway"],
    "server": ["server", "cpu", "disk", "ssh", "nginx", "systemd", "uptime"],
    "laptop": ["laptop", "battery", "bios", "driver", "boot", "screen", "keyboard"],
    "software": ["install", "application", "crash", "update", "license", "windows", "macos"],
}


def extract_text(filename: str, data: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    if ext == "docx":
        doc = Document(io.BytesIO(data))
        out = []
        for p in doc.paragraphs:
            if p.style.name.startswith("Heading"):
                out.append(f"## {p.text}")
            elif p.text.strip():
                out.append(p.text)
        return "\n\n".join(out)
    if ext in ("txt", "md"):
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: .{ext}")


def infer_category(text: str, filename: str) -> str:
    blob = (filename + " " + text[:3000]).lower()
    scores = {c: sum(blob.count(k) for k in kws) for c, kws in CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def chunk_text(text: str, max_chars: int = 900, overlap_paras: int = 1):
    """Split on headings, then pack paragraphs up to max_chars.

    Each chunk is prefixed with its section heading so the embedding keeps context.
    Returns list of (section, chunk_text).
    """
    sections, heading, buf = [], "Introduction", []
    for line in text.splitlines():
        m = re.match(r"^\s{0,3}#{1,4}\s+(.*)", line)
        if m:
            if buf:
                sections.append((heading, "\n".join(buf)))
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf)))

    chunks = []
    for heading, body in sections:
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        cur = []
        for p in paras:
            if cur and sum(len(x) for x in cur) + len(p) > max_chars:
                chunks.append((heading, f"{heading}\n" + "\n\n".join(cur)))
                cur = cur[-overlap_paras:] if overlap_paras else []
            cur.append(p)
        if cur:
            chunks.append((heading, f"{heading}\n" + "\n\n".join(cur)))
    return chunks
