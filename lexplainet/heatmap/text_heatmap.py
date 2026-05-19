"""Text heatmap utilities for language-model relevance scores.

The module provides three complementary renderers:

- ``latex_heatmap`` / ``pdf_heatmap`` for publication-style PDF output.
- ``plot_text_heatmap`` for static Matplotlib figures that can be saved as PNG.
- ``interactive_text_heatmap`` for optional Plotly hover labels.
"""

from __future__ import annotations

import html
import shutil
import subprocess
from pathlib import Path

_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_SENTENCE_END = (".", "!", "?", ":", ";")
_NO_SPACE_BEFORE = set(",.!?:;%)]}\u201d\u2019")
_NO_SPACE_AFTER = set("([{\u201c\u2018")
_NUMERIC_JOINERS = set(",.:/")
_UNIT_SUFFIXES = {
    "%",
    "k",
    "m",
    "b",
    "t",
    "mm",
    "cm",
    "km",
    "kg",
    "g",
    "mg",
    "lb",
    "lbs",
    "oz",
    "s",
    "ms",
    "hz",
    "khz",
    "mhz",
    "ghz",
    "w",
    "kw",
    "mw",
    "kb",
    "mb",
    "gb",
    "tb",
    "°",
    "°c",
    "°f",
}


def _flatten(values):
    if isinstance(values, (list, tuple)):
        flat = []
        for value in values:
            flat.extend(_flatten(value))
        return flat
    return [values]


def _to_float_list(values):
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()

    if isinstance(values, (int, float)):
        return [float(values)]

    return [float(value) for value in _flatten(values)]


def _prepare_inputs(words, relevances, normalize=False):
    words = [str(word) for word in words]
    relevances = _to_float_list(relevances)

    if len(words) != len(relevances):
        raise ValueError("words and relevances must have the same length")
    if len(words) == 0:
        raise ValueError("at least one word/token is required")

    if normalize:
        scale = max(abs(value) for value in relevances)
        if scale > 0:
            relevances = [value / scale for value in relevances]

    if any(value < -1 or value > 1 for value in relevances):
        raise ValueError(
            "relevances must be in [-1, 1]; pass normalize=True to scale them"
        )

    return words, relevances


def _fallback_bwr(relevance):
    relevance = max(-1.0, min(1.0, float(relevance)))
    if relevance < 0:
        mix = relevance + 1.0
        return (mix, mix, 1.0, 1.0)
    mix = 1.0 - relevance
    return (1.0, mix, mix, 1.0)


def _apply_colormap(relevance, cmap):
    try:
        import matplotlib.cm as cm
        import matplotlib.colors as colors
    except ImportError as exc:
        if cmap in {"bwr", "seismic", "coolwarm"}:
            return _fallback_bwr(relevance)
        raise ImportError(
            "custom colormaps require matplotlib; use cmap='bwr' without matplotlib"
        ) from exc

    colormap = cm.get_cmap(cmap)
    return colormap(colors.Normalize(vmin=-1, vmax=1)(float(relevance)))


def _rgb255(relevance, cmap):
    rgba = _apply_colormap(relevance, cmap)
    return tuple(int(channel * 255) for channel in rgba[:3])


def _rgba(relevance, cmap, alpha=1.0):
    rgba = list(_apply_colormap(relevance, cmap))
    rgba[3] = alpha
    return tuple(rgba)


def _latex_escape(text):
    return "".join(_LATEX_SPECIALS.get(char, char) for char in text)


def _looks_numeric(text):
    return bool(text) and any(char.isdigit() for char in text)


def _is_unit_suffix(text):
    return text.lower() in _UNIT_SUFFIXES


def _attach_to_previous(previous, current, had_leading_space):
    if not previous or not current:
        return False

    previous = previous.rstrip()
    current = current.lstrip()
    if not previous or not current:
        return False

    previous_last = previous[-1]
    current_first = current[0]

    if current_first in _NO_SPACE_BEFORE:
        return True
    if previous_last in _NO_SPACE_AFTER:
        return True
    if (
        previous_last in _NUMERIC_JOINERS
        and current_first.isdigit()
        and _looks_numeric(previous)
    ):
        return True
    if previous_last.isdigit() and current_first in _NUMERIC_JOINERS:
        return True
    if previous_last.isdigit() and _is_unit_suffix(current):
        return True
    if not had_leading_space and previous_last.isdigit() and current_first.isalpha():
        return len(current) <= 4
    if not had_leading_space and previous_last.isalpha() and current_first.isdigit():
        return True
    if not had_leading_space and current_first in "-–—+":
        return True
    if not had_leading_space and previous_last in "-–—+" and current_first.isalnum():
        return True

    return False


def _token_spacing(previous, current, had_leading_space=False):
    if not previous:
        return ""
    if _attach_to_previous(previous, current, had_leading_space):
        return ""
    if had_leading_space:
        return " "
    return " "


def _clean_raw_token(raw):
    token = str(raw)
    token = token.replace("Ċ", "\n").replace("<0x0A>", "\n")
    token = token.replace("▁", " ").replace("Ġ", " ")
    token = token.replace("##", "")
    return token


def _display_tokens(words):
    """Return readable token strings while preserving tokenizer spacing hints."""

    display = []
    previous = ""

    for raw in words:
        token = _clean_raw_token(raw)
        if token == "":
            continue

        newline_prefix = token[: len(token) - len(token.lstrip("\n"))]
        body = token.lstrip("\n")
        had_leading_space = len(body) > len(body.lstrip(" "))
        body = body.lstrip(" ")

        if newline_prefix:
            if body:
                visible = newline_prefix + body
            else:
                visible = newline_prefix
        elif not display:
            visible = body
        else:
            visible = _token_spacing(previous, body, had_leading_space) + body

        if visible == "":
            continue

        display.append(visible)
        if body:
            if _attach_to_previous(previous, body, had_leading_space):
                previous = previous.rstrip() + body
            else:
                previous = body

    return display


def _display_spans(words, relevances):
    spans = []
    for token, relevance in zip(_display_tokens(words), relevances):
        newline_count = len(token) - len(token.lstrip("\n"))
        without_newline = token.lstrip("\n")
        leading_space = len(without_newline) > len(without_newline.lstrip(" "))
        text = without_newline.strip()
        if not text and newline_count == 0:
            continue
        spans.append(
            {
                "text": text,
                "relevance": float(relevance),
                "leading_space": leading_space,
                "newline_count": newline_count,
            }
        )
    return spans


def _latex_space(mode, is_first, token_gap):
    if is_first:
        return ""
    if mode == "text":
        return " "
    return rf"\hspace{{{token_gap}}}"


def _latex_fragment(words, relevances, cmap, spacing="text", token_gap="0.12em"):
    if spacing not in {"text", "tokens"}:
        raise ValueError("spacing must be either 'text' or 'tokens'")

    pieces = []
    has_text = False
    for span in _display_spans(words, relevances):
        if span["newline_count"]:
            pieces.append(r"\par\noindent " + "\n")
            has_text = False
        if not span["text"]:
            continue

        r, g, b = _rgb255(span["relevance"], cmap)
        prefix = (
            _latex_space(spacing, not has_text, token_gap)
            if span["leading_space"]
            else ""
        )
        token = _latex_escape(span["text"])
        pieces.append(prefix + rf"\colorbox[RGB]{{{r},{g},{b}}}{{\strut {token}}}")
        has_text = True
    return "".join(pieces)


def _generate_latex(
    words,
    relevances,
    cmap="bwr",
    page_width="160mm",
    font_size="10pt",
    line_height="13pt",
    spacing="text",
    token_gap="0.12em",
    box_padding="0.45pt",
    page_margin="2pt",
    border=False,
    border_padding="3pt",
    border_width="0.4pt",
    border_color="black",
    justify=True,
    justify_last=True,
):
    body = _latex_fragment(
        words, relevances, cmap, spacing=spacing, token_gap=token_gap
    )
    alignment = r"\justifying" if justify and spacing == "text" else r"\raggedright"
    final_line = (
        r"\setlength{\parfillskip}{0pt}"
        if justify and justify_last and spacing == "text"
        else ""
    )
    content = (
        r"\begin{minipage}{\linewidth}"
        + alignment
        + final_line
        + r"\setlength{\parindent}{0pt}"
        + body
        + r"\par\end{minipage}"
    )
    if border:
        content = (
            r"{\setlength\fboxsep{"
            + border_padding
            + r"}\setlength\fboxrule{"
            + border_width
            + r"}\fcolorbox{"
            + border_color
            + r"}{white}{"
            + content
            + r"}}"
        )

    return rf"""
\documentclass[varwidth={page_width}, border={page_margin}]{{standalone}}
\usepackage[table]{{xcolor}}
\usepackage{{ragged2e}}
\setlength\fboxsep{{{box_padding}}}
\begin{{document}}
\fontsize{{{font_size}}}{{{line_height}}}\selectfont
\noindent {content}
\end{{document}}
""".strip()


def _run_latex(tex_path, pdf_path, backend):
    if backend not in {"xelatex", "pdflatex", "lualatex"}:
        raise ValueError("backend must be one of: xelatex, pdflatex, lualatex")
    if shutil.which(backend) is None:
        raise RuntimeError(f"LaTeX backend '{backend}' was not found on PATH")

    subprocess.run(
        [
            backend,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(pdf_path.parent),
            str(tex_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _convert_pdf_to_png(pdf_path, png_path, dpi=200):
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("pdftoppm") is not None:
        output_base = png_path.with_suffix("")
        subprocess.run(
            [
                "pdftoppm",
                "-singlefile",
                "-png",
                "-r",
                str(dpi),
                str(pdf_path),
                str(output_base),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        generated = output_base.with_suffix(".png")
        if generated != png_path:
            generated.replace(png_path)
        return png_path

    for command in ("magick", "convert"):
        if shutil.which(command) is not None:
            args = [
                command,
                "-density",
                str(dpi),
                str(pdf_path),
                "-quality",
                "95",
                str(png_path),
            ]
            subprocess.run(
                args,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return png_path

    raise RuntimeError(
        "Could not convert PDF to PNG; install poppler-utils or ImageMagick"
    )


def latex_heatmap(
    words,
    relevances,
    cmap="bwr",
    path="heatmap.pdf",
    png_path=None,
    delete_aux_files=True,
    backend="xelatex",
    normalize=False,
    dpi=200,
    page_width="160mm",
    font_size="10pt",
    line_height="13pt",
    spacing="text",
    token_gap="0.12em",
    box_padding="0.45pt",
    page_margin="2pt",
    border=False,
    border_padding="3pt",
    border_width="0.4pt",
    border_color="black",
    justify=True,
    justify_last=True,
):
    """Render a token relevance heatmap through LaTeX.

    Returns a dictionary with generated file paths. When ``png_path`` is set, the
    compiled PDF is also rasterized to PNG using ``pdftoppm`` or ImageMagick.
    """

    words, relevances = _prepare_inputs(words, relevances, normalize=normalize)
    pdf_path = Path(path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path = pdf_path.with_suffix(".tex")
    tex_path.write_text(
        _generate_latex(
            words,
            relevances,
            cmap=cmap,
            page_width=page_width,
            font_size=font_size,
            line_height=line_height,
            spacing=spacing,
            token_gap=token_gap,
            box_padding=box_padding,
            page_margin=page_margin,
            border=border,
            border_padding=border_padding,
            border_width=border_width,
            border_color=border_color,
            justify=justify,
            justify_last=justify_last,
        ),
        encoding="utf-8",
    )

    _run_latex(tex_path, pdf_path, backend)

    outputs = {"pdf": pdf_path}
    if png_path is not None:
        outputs["png"] = _convert_pdf_to_png(pdf_path, png_path, dpi=dpi)

    if delete_aux_files:
        for suffix in (".aux", ".log", ".tex"):
            aux_path = pdf_path.with_suffix(suffix)
            if aux_path.exists():
                aux_path.unlink()

    return outputs


def pdf_heatmap(*args, **kwargs):
    """Backward-compatible alias for ``latex_heatmap``."""

    return latex_heatmap(*args, **kwargs)


def _measure_text_width(ax, text, fontsize, font_family):
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    probe = ax.text(
        0,
        0,
        text,
        fontsize=fontsize,
        fontfamily=font_family,
        alpha=0,
    )
    width = probe.get_window_extent(renderer=renderer).width
    probe.remove()
    return width


def _layout_justified_lines(
    ax, spans, fontsize, font_family, left, right, line_px, token_pad_x
):
    lines = []
    current = []
    current_width = 0.0
    current_gaps = 0

    for span in spans:
        if span["newline_count"]:
            if current:
                lines.append(
                    {
                        "items": current,
                        "width": current_width,
                        "gaps": current_gaps,
                        "forced": True,
                    }
                )
                current = []
                current_width = 0.0
                current_gaps = 0
            for _ in range(max(0, span["newline_count"] - 1)):
                lines.append({"items": [], "width": 0.0, "gaps": 0, "forced": True})
        if not span["text"]:
            continue

        text_width = _measure_text_width(ax, span["text"], fontsize, font_family)
        box_width = text_width + 2 * token_pad_x
        starts_with_space = bool(current and span["leading_space"])
        gap_count = 1 if starts_with_space else 0
        next_width = current_width + box_width + (0 if not starts_with_space else 0)

        if current and next_width > right - left:
            lines.append(
                {
                    "items": current,
                    "width": current_width,
                    "gaps": current_gaps,
                    "forced": False,
                }
            )
            current = []
            current_width = 0.0
            current_gaps = 0
            starts_with_space = False
            gap_count = 0

        current.append(
            {
                "span": span,
                "text_width": text_width,
                "box_width": box_width,
                "starts_with_space": starts_with_space,
            }
        )
        current_width += box_width
        current_gaps += gap_count

    if current:
        lines.append(
            {
                "items": current,
                "width": current_width,
                "gaps": current_gaps,
                "forced": False,
            }
        )
    return lines


def plot_text_heatmap(
    words,
    relevances,
    cmap="bwr",
    ax=None,
    max_width=0.96,
    line_height=1.55,
    fontsize=12,
    text_color="black",
    save_path=None,
    show=True,
    normalize=False,
    dpi=200,
    figsize=None,
    justify=True,
    justify_last=True,
    max_justify_gap=None,
    font_family="DejaVu Serif",
    box_alpha=0.9,
    pad_px=4,
):
    """Draw a readable static text heatmap with Matplotlib.

    The layout wraps tokens into lines and, by default, justifies every complete
    line by spreading the remaining width over word gaps. Set ``justify=False``
    for ragged-right text.
    """

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise ImportError("plot_text_heatmap requires matplotlib") from exc

    plt.rcParams.setdefault("font.family", font_family)
    plt.rcParams.setdefault("mathtext.fontset", "cm")

    words, relevances = _prepare_inputs(words, relevances, normalize=normalize)
    spans = _display_spans(words, relevances)

    if ax is None:
        if figsize is None:
            rows_guess = max(1, len(spans) // 12 + 1)
            figsize = (12, max(2.0, rows_guess * 0.62 + 0.8))
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    fig.canvas.draw()
    ax_bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())

    left = pad_px
    right = max(left + 1, max_width * ax_bbox.width - pad_px)
    top = pad_px
    token_pad_x = 0.16 * fontsize
    token_pad_y = 0.12 * fontsize
    line_px = line_height * fontsize
    if max_justify_gap is None:
        max_justify_gap = 1.8 * _measure_text_width(ax, " ", fontsize, font_family)

    lines = _layout_justified_lines(
        ax, spans, fontsize, font_family, left, right, line_px, token_pad_x
    )
    total_height = top * 2 + max(1, len(lines)) * line_px
    if total_height > ax_bbox.height:
        width_in, height_in = fig.get_size_inches()
        fig.set_size_inches(
            width_in, height_in * total_height / ax_bbox.height, forward=True
        )
        fig.canvas.draw()
        ax_bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
        left = pad_px
        right = max(left + 1, max_width * ax_bbox.width - pad_px)
        top = pad_px
        lines = _layout_justified_lines(
            ax, spans, fontsize, font_family, left, right, line_px, token_pad_x
        )
        total_height = top * 2 + max(1, len(lines)) * line_px

    ax.set_xlim(0, ax_bbox.width)
    ax.set_ylim(total_height, 0)
    max_draw_x = left

    for line_index, line in enumerate(lines):
        y = top + line_index * line_px
        items = line["items"]
        if not items:
            continue
        is_last_line = line_index == len(lines) - 1 or line["forced"]
        gap_px = 0.0
        if justify and line["gaps"] and (justify_last or not is_last_line):
            requested_gap = max(0.0, (right - left - line["width"]) / line["gaps"])
            gap_px = min(requested_gap, max_justify_gap)

        x = left
        for item in items:
            if item["starts_with_space"]:
                x += gap_px
            span = item["span"]
            rect = Rectangle(
                (x, y + token_pad_y * 0.2),
                item["box_width"],
                line_px * 0.78,
                facecolor=_rgba(span["relevance"], cmap, box_alpha),
                edgecolor="none",
                linewidth=0,
            )
            ax.add_patch(rect)
            artist = ax.text(
                x + token_pad_x,
                y + line_px * 0.41,
                span["text"],
                fontsize=fontsize,
                fontfamily=font_family,
                color=text_color,
                va="center",
                ha="left",
            )
            artist.set_gid(f"relevance={span['relevance']:.6f}")
            x += item["box_width"]
            max_draw_x = max(max_draw_x, x)

    ax.set_xlim(0, min(ax_bbox.width, max_draw_x + pad_px))

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    if show:
        plt.show()

    return fig, ax


def _rgba_css(relevance, cmap, alpha=0.65):
    r, g, b = _rgb255(relevance, cmap)
    return f"rgba({r},{g},{b},{alpha})"


def _layout_plotly_lines(spans, max_line_units):
    lines = []
    current = []
    current_width = 0.0
    current_gaps = 0

    for span in spans:
        if span["newline_count"]:
            if current:
                lines.append(
                    {
                        "items": current,
                        "width": current_width,
                        "gaps": current_gaps,
                        "forced": True,
                    }
                )
                current = []
                current_width = 0.0
                current_gaps = 0
            for _ in range(max(0, span["newline_count"] - 1)):
                lines.append({"items": [], "width": 0.0, "gaps": 0, "forced": True})
        if not span["text"]:
            continue

        box_width = max(1.4, len(span["text"]) * 0.62 + 0.75)
        starts_with_space = bool(current and span["leading_space"])
        next_width = current_width + box_width
        if current and next_width > max_line_units:
            lines.append(
                {
                    "items": current,
                    "width": current_width,
                    "gaps": current_gaps,
                    "forced": False,
                }
            )
            current = []
            current_width = 0.0
            current_gaps = 0
            starts_with_space = False

        current.append(
            {
                "span": span,
                "box_width": box_width,
                "starts_with_space": starts_with_space,
            }
        )
        current_width += box_width
        if starts_with_space:
            current_gaps += 1

    if current:
        lines.append(
            {
                "items": current,
                "width": current_width,
                "gaps": current_gaps,
                "forced": False,
            }
        )
    return lines


def interactive_text_heatmap(
    words,
    relevances,
    cmap="bwr",
    normalize=False,
    width=1000,
    fontsize=15,
    max_line_units=68,
    justify=True,
    justify_last=True,
    max_justify_gap=1.25,
    font_family="Latin Modern Roman, Computer Modern, Times New Roman, Times, serif",
    box_alpha=0.65,
):
    """Create a Plotly text heatmap with colored hoverable token boxes."""

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("interactive_text_heatmap requires plotly") from exc

    words, relevances = _prepare_inputs(words, relevances, normalize=normalize)
    spans = _display_spans(words, relevances)
    lines = _layout_plotly_lines(spans, max_line_units=max_line_units)

    fig = go.Figure()
    annotations = []
    x_extent = 0.0
    row_height = 1.25
    box_height = 0.82

    for line_index, line in enumerate(lines):
        y_center = -line_index * row_height
        items = line["items"]
        if not items:
            continue
        is_last_line = line_index == len(lines) - 1 or line["forced"]
        gap = 0.0
        if justify and line["gaps"] and (justify_last or not is_last_line):
            requested_gap = max(0.0, (max_line_units - line["width"]) / line["gaps"])
            gap = (
                requested_gap
                if max_justify_gap is None
                else min(requested_gap, max_justify_gap)
            )

        x = 0.0
        for item in items:
            if item["starts_with_space"]:
                x += gap
            span = item["span"]
            box_width = item["box_width"]
            x0 = x
            x1 = x + box_width
            y0 = y_center - box_height / 2
            y1 = y_center + box_height / 2
            hover = f"token: {html.escape(span['text'])}<br>relevance: {span['relevance']:.6f}"
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, x1, x0, x0],
                    y=[y0, y0, y1, y1, y0],
                    mode="lines",
                    fill="toself",
                    fillcolor=_rgba_css(span["relevance"], cmap, box_alpha),
                    line={"color": "rgba(0,0,0,0)", "width": 0},
                    hoveron="fills",
                    hoverinfo="text",
                    text=hover,
                    showlegend=False,
                )
            )
            annotations.append(
                {
                    "x": (x0 + x1) / 2,
                    "y": y_center,
                    "text": html.escape(span["text"]),
                    "showarrow": False,
                    "font": {"size": fontsize, "color": "black", "family": font_family},
                    "xanchor": "center",
                    "yanchor": "middle",
                }
            )
            x = x1
            x_extent = max(x_extent, x1)

    x_range_end = max(1.0, min(max_line_units, x_extent))

    fig.update_layout(
        width=width,
        height=max(80, 18 + max(1, len(lines)) * 34),
        margin={"l": 2, "r": 2, "t": 2, "b": 2},
        plot_bgcolor="white",
        paper_bgcolor="white",
        annotations=annotations,
        xaxis={"visible": False, "range": [0, x_range_end], "fixedrange": True},
        yaxis={
            "visible": False,
            "range": [-max(1, len(lines)) * row_height + 0.45, 0.45],
            "fixedrange": True,
        },
        hoverlabel={"bgcolor": "white", "font_size": 13},
        showlegend=False,
    )
    return fig


def clean_tokens(words, escape_latex=False):
    """Clean common wordpiece tokens into display-oriented text tokens.

    Set ``escape_latex=True`` only when you need the returned tokens for custom
    LaTeX code. The renderers in this module escape LaTeX internally.
    """

    cleaned = _display_tokens(words)
    if escape_latex:
        return [_latex_escape(token) for token in cleaned]
    return cleaned
