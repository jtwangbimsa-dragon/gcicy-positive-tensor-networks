#!/usr/bin/env python3
"""Generate the REVTeX/PRD submission source from the maintained paper."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "gcicy paper" / "gcicy_tn_paper.tex"
TARGET = ROOT / "gcicy paper" / "gcicy_tn_prd.tex"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one occurrence, found {count}: {old!r}")
    return text.replace(old, new, 1)


text = SOURCE.read_text(encoding="utf-8")
text = replace_once(
    text,
    "\\documentclass[a4paper,11pt]{article}\n"
    "\\pdfoutput=1\n"
    "\\usepackage{jheppub}\n"
    "\\usepackage[a4paper,left=2.05cm,right=2.05cm,top=2.35cm,bottom=2.35cm]{geometry}\n"
    "\\usepackage{amsthm}",
    "\\documentclass[aps,prd,12pt,onecolumn,notitlepage,showkeys,superscriptaddress,"
    "nofootinbib,longbibliography]{revtex4-2}\n"
    "\\pdfoutput=1\n"
    "\\usepackage{amsmath}\n"
    "\\usepackage{amssymb}\n"
    "\\usepackage{amsthm}",
)
text = replace_once(
    text,
    "\\usepackage{url}\n"
    "\\setlength{\\hoffset}{0pt}\n"
    "\\setlength{\\voffset}{0pt}",
    "\\usepackage{url}\n"
    "\\usepackage[colorlinks=true,allcolors=blue]{hyperref}",
)
text = replace_once(
    text,
    "\\begin{document}\n\\maketitle\n\\flushbottom",
    "\\maketitle",
)
text = replace_once(
    text,
    "\\title{\\boldmath Positive Tensor-Network",
    "\\begin{document}\n\n\\title{Positive Tensor-Network",
)
text = replace_once(
    text,
    "K\\\"ahler Metrics\\\\\non Sequential gCICY Threefolds}",
    "K\\\"ahler Metrics on Sequential gCICY Threefolds}",
)
text = replace_once(text, "\\emailAdd{", "\\email{")
text = replace_once(text, "\\abstract{", "\\begin{abstract}\n")
keywords_marker = "\n\n\\keywords"
keywords_index = text.find(keywords_marker)
if keywords_index < 1 or text[keywords_index - 1] != "}":
    raise RuntimeError("could not locate the closing abstract brace")
text = (
    text[: keywords_index - 1]
    + "\n\\end{abstract}"
    + text[keywords_index:]
)
text = replace_once(
    text,
    "\\bibliographystyle{unsrtnat}",
    "\\bibliographystyle{apsrev4-2}",
)

TARGET.write_text(text, encoding="utf-8")
print(TARGET)
