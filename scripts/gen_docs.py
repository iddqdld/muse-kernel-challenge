#!/usr/bin/env python3
"""Generate a static documentation site for this repository under docs/.

The generator is deterministic and reads the working tree: no list of pages,
modules or functions is written by hand. It runs from any directory (the repo
root is found with git). Standard library only.
"""

import hashlib
import html
import os
import re
import subprocess
import sys
import urllib.parse
from datetime import datetime, timezone

# --------------------------------------------------------------------------
# repository / git helpers
# --------------------------------------------------------------------------

SKIP_DIRS = {"target", "node_modules", "__pycache__", "profiles"}
GEN_SPAN_RE = re.compile(r'<span class="gen">.*?</span>', re.S)


def git(*args):
    """Run git in the repository and return stdout as text."""
    proc = subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), proc.stderr.strip()))
    return proc.stdout


def find_root():
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    )
    if out.returncode != 0 or not out.stdout.strip():
        sys.exit("gen_docs.py: not inside a git repository")
    return os.path.abspath(out.stdout.strip())


ROOT = find_root()
DOCS = os.path.join(ROOT, "docs")
GEN_TIME = ""


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def rel_posix(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def walk_files():
    """Yield (rel_posix, abspath) for every non-hidden file."""
    for dirpath, dirnames, filenames in os.walk(ROOT, followlinks=False):
        rel_dir = rel_posix(dirpath) if dirpath != ROOT else "."
        dirnames[:] = sorted(
            d for d in dirnames
            if not d.startswith(".") and d not in SKIP_DIRS
        )
        for name in sorted(filenames):
            rel = name if rel_dir == "." else rel_dir + "/" + name
            yield rel, os.path.join(dirpath, name)


def ignored_paths(paths):
    if not paths:
        return set()
    proc = subprocess.run(
        ["git", "-c", "core.quotePath=false", "check-ignore", "--stdin"],
        input="\n".join(paths), capture_output=True, text=True,
    )
    return set(proc.stdout.splitlines())


# --------------------------------------------------------------------------
# slugs for rendered markdown pages
# --------------------------------------------------------------------------

RU = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}
SLUG_OK = re.compile(r"^[a-z0-9._-]+$")


def slugify(rel):
    base = rel[:-3] if rel.endswith(".md") else rel
    chars = []
    for ch in base.lower():
        if ch in RU:
            chars.append(RU[ch])
        elif re.match(r"[a-z0-9._/-]", ch):
            chars.append(ch)
        elif ch == "/":
            chars.append("-")
        else:
            chars.append("-")
    slug = re.sub(r"-+", "-", "".join(chars).replace("/", "-")).strip("-.")
    if not slug or not SLUG_OK.match(slug):
        h = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
        slug = (slug + "-" if slug else "") + h
    return slug


# --------------------------------------------------------------------------
# markdown rendering
# --------------------------------------------------------------------------

def first_sentence(text, limit=240):
    text = " ".join(text.split())
    if not text:
        return ""
    m = re.search(r"\.(?:\s|$)", text)
    if m and m.end() <= len(text):
        text = text[:m.end()]
    if len(text) > limit:
        text = text[:limit].rstrip() + "..."
    return text


def inline(text, rewrite):
    text = html.escape(text, quote=False)

    codes = []

    def stash_code(m):
        codes.append(m.group(1))
        return "\x01%d\x01" % (len(codes) - 1)

    text = re.sub(r"`([^`]+)`", stash_code, text)

    def link_repl(m):
        alt, url = m.group(1), html.unescape(m.group(2))
        return '<a href="%s">%s</a>' % (html.escape(rewrite(url), quote=True), alt)

    text = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", link_repl, text)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link_repl, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    text = re.sub(r"(?<![\w])_([^_]+)_(?![\w])", r"<em>\1</em>", text)

    def unstash(m):
        return "<code>" + codes[int(m.group(1))] + "</code>"

    return re.sub(r"\x01(\d+)\x01", unstash, text)


def split_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def is_sep_row(line):
    if "|" not in line:
        return False
    cells = split_row(line)
    return bool(cells) and all(re.match(r"^:?-+:?$", c) for c in cells)


def parse_list(lines, i):
    """Parse a list block into a tree of items; return (items, next_index)."""
    items, stack = [], []
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n and re.match(r"^(\s*)([-*+]|\d+[.)])\s+", lines[j]):
                i = j
                line = lines[i]
            else:
                break
        m = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", line)
        if not m:
            break
        indent = len(m.group(1))
        ordered = not re.match(r"^[-*+]$", m.group(2))
        item = {"indent": indent, "ordered": ordered,
                "text": m.group(3), "children": []}
        while stack and stack[-1]["indent"] >= indent:
            stack.pop()
        if stack:
            stack[-1]["children"].append(item)
        else:
            items.append(item)
        stack.append(item)
        i += 1
    return items, i


def render_items(items, rewrite):
    if not items:
        return ""
    tag = "ol" if items[0]["ordered"] else "ul"
    out = ["<%s>" % tag]
    for it in items:
        out.append("<li>" + inline(it["text"], rewrite)
                   + render_items(it["children"], rewrite) + "</li>")
    out.append("</%s>" % tag)
    return "".join(out)


def render_table(lines, i, rewrite):
    header = split_row(lines[i])
    j = i + 2
    rows = []
    while j < len(lines) and lines[j].strip() and "|" in lines[j]:
        rows.append(split_row(lines[j]))
        j += 1
    out = ["<table><thead><tr>"]
    for cell in header:
        out.append("<th>" + inline(cell, rewrite) + "</th>")
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for k in range(len(header)):
            cell = row[k] if k < len(row) else ""
            out.append("<td>" + inline(cell, rewrite) + "</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out), j


def render_md(text, rewrite, base_level=2):
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out, para = [], []
    i, n = 0, len(lines)

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(para), rewrite) + "</p>")
            para.clear()

    while i < n:
        line = lines[i]
        if re.match(r"^\s*```", line):
            flush()
            i += 1
            buf = []
            while i < n and not re.match(r"^\s*```", lines[i]):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            flush()
            lvl = min(6, len(m.group(1)) + base_level - 1)
            out.append("<h%d>%s</h%d>" % (lvl, inline(m.group(2), rewrite), lvl))
            i += 1
            continue
        if re.match(r"^\s*([-*_])\s*(\1\s*){2,}$", line):
            flush()
            out.append("<hr>")
            i += 1
            continue
        if "|" in line and i + 1 < n and is_sep_row(lines[i + 1]):
            flush()
            block, i = render_table(lines, i, rewrite)
            out.append(block)
            continue
        if re.match(r"^\s*>", line):
            flush()
            buf = []
            while i < n and re.match(r"^\s*>", lines[i]):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            out.append("<blockquote>" + render_md("\n".join(buf), rewrite, base_level)
                       + "</blockquote>")
            continue
        if re.match(r"^\s*([-*+]|\d+[.)])\s+", line):
            flush()
            items, i = parse_list(lines, i)
            out.append(render_items(items, rewrite))
            continue
        if not line.strip():
            flush()
            i += 1
            continue
        para.append(line.strip())
        i += 1
    flush()
    return "\n".join(out)


def first_heading(text):
    for line in text.split("\n"):
        m = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if m:
            return m.group(1)
    return ""


# --------------------------------------------------------------------------
# markdown link rewriting
# --------------------------------------------------------------------------

def make_rewrite(src_rel, out_path, linkmap):
    src_dir = os.path.dirname(src_rel)
    out_dir = os.path.join(DOCS, os.path.dirname(out_path))
    depth = out_path.count("/")

    def rewrite(url):
        if not url or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url) \
                or url.startswith("#") or url.startswith("//") or url.startswith("/"):
            return url
        frag = ""
        if "#" in url:
            url, frag = url.split("#", 1)
            frag = "#" + frag
        joined = os.path.normpath(os.path.join(src_dir, urllib.parse.unquote(url)))
        joined = joined.replace(os.sep, "/")
        if joined in linkmap:
            return "../" * depth + linkmap[joined] + frag
        if os.path.exists(os.path.join(ROOT, joined)):
            rel = os.path.relpath(os.path.join(ROOT, joined), out_dir).replace(os.sep, "/")
            return rel + frag
        return url + frag

    return rewrite


# --------------------------------------------------------------------------
# page shell
# --------------------------------------------------------------------------

STYLE = (
    ":root{color-scheme:light dark}"
    "body{max-width:62rem;margin:0 auto;padding:1rem;font:16px/1.55 system-ui,sans-serif}"
    "nav{font-size:.9rem;border-bottom:1px solid #888;padding-bottom:.5rem;margin-bottom:1rem}"
    "nav b{margin-right:.35rem}"
    "nav ul{margin:.2rem 0;padding-left:1.3rem}"
    "nav li{display:inline;margin-right:.6rem}"
    "code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em}"
    "pre{overflow:auto;background:#8882;padding:.6rem;border-radius:.3rem}"
    "table{border-collapse:collapse;margin:.5rem 0}"
    "th,td{border:1px solid #888;padding:.2rem .55rem;text-align:left;vertical-align:top}"
    ".purpose{font-style:italic}"
    "blockquote{margin:.5rem 0;padding-left:.8rem;border-left:3px solid #888}"
    "footer{margin-top:2rem;border-top:1px solid #888;padding-top:.5rem;font-size:.85rem;color:#777}"
    "details{margin:.25rem 0}"
    "summary{cursor:pointer;font-weight:600}"
    "hr{border:0;border-top:1px solid #888}"
)

NAV_GROUPS = []


def page(title, source, out_path, content):
    depth = out_path.count("/")
    link = lambda p: "../" * depth + p
    nav = ['<nav><b>Site:</b>']
    nav.append('<a href="%s">Overview</a>' % link("index.html"))
    nav.append('| <a href="%s">Code map</a>' % link("modules.html"))
    nav.append('| <a href="%s">Commits</a>' % link("commits/index.html"))
    nav.append("</nav>")
    nav.append("<nav>")
    for group, items in NAV_GROUPS:
        nav.append("<b>%s:</b><ul>" % html.escape(group))
        for label, outp in items:
            nav.append('<li><a href="%s">%s</a></li>' % (link(outp), html.escape(label)))
        nav.append("</ul>")
    nav.append("</nav>")
    return (
        "<!doctype html>\n<html lang=\"en\">\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>%s</title>\n<style>%s</style>\n<h1>%s</h1>\n%s\n%s\n"
        "<footer>Source: <code>%s</code> &middot; generated "
        "<span class=\"gen\">%s</span></footer>\n</html>\n"
    ) % (html.escape(title), STYLE, html.escape(title), "\n".join(nav), content,
         html.escape(source), GEN_TIME)


def stable(html_text):
    return GEN_SPAN_RE.sub('<span class="gen"></span>', html_text)


# --------------------------------------------------------------------------
# source-file parsing for the code map
# --------------------------------------------------------------------------

MODULE_EXTS = {".rs", ".py", ".sh", ".mjs", ".js"}
DOCSTRING_RE = re.compile(r'^("""|\'\'\')')


def py_module_doc(text):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.startswith("#"):
            i += 1
            continue
        break
    if i >= len(lines):
        return ""
    s = lines[i].strip()
    for q in ('"""', "'''"):
        if s.startswith(q):
            rest = s[3:]
            if q in rest:
                return first_sentence(rest.split(q)[0])
            buf = [rest]
            j = i + 1
            while j < len(lines) and q not in lines[j]:
                buf.append(lines[j])
                j += 1
            return first_sentence(" ".join(x.strip() for x in buf))
    return ""


def docstring_after(lines, idx):
    j = idx + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    if j >= len(lines):
        return ""
    s = lines[j].strip()
    for q in ('"""', "'''"):
        if s.startswith(q):
            rest = s[3:]
            if q in rest:
                return first_sentence(rest.split(q)[0])
            buf = [rest]
            k = j + 1
            while k < len(lines) and q not in lines[k]:
                buf.append(lines[k])
                k += 1
            return first_sentence(" ".join(x.strip() for x in buf))
    return ""


def parse_py(text):
    purpose = py_module_doc(text)
    lines = text.splitlines()
    items = []
    for idx, line in enumerate(lines):
        m = re.match(r"^(?:async\s+)?def\s+([A-Za-z_]\w*)", line) \
            or re.match(r"^class\s+([A-Za-z_]\w*)", line)
        if m:
            name = m.group(1)
            if name.startswith("_"):
                continue
            kind = "class" if line.startswith("class ") else "def"
            items.append((kind, name, docstring_after(lines, idx)))
    if not purpose and items:
        purpose = items[0][2]
    return purpose, items


def parse_rs(text):
    lines = text.splitlines()
    mod_doc = [l.strip()[3:].strip() for l in lines if l.strip().startswith("//!")]
    purpose = first_sentence(" ".join(mod_doc))
    items = []
    doc = []
    fn_re = re.compile(
        r"^\s*pub\s+(?:async\s+|const\s+|unsafe\s+|extern\s+\"[^\"]*\"\s+)*"
        r"fn\s+([A-Za-z_]\w*)")
    type_re = re.compile(r"^\s*pub\s+(struct|enum|trait)\s+([A-Za-z_]\w*)")
    for line in lines:
        s = line.strip()
        if s.startswith("///"):
            doc.append(s[3:].strip())
            continue
        m = fn_re.match(line)
        if m:
            name = m.group(1)
            if not name.startswith("_"):
                items.append(("fn", name, first_sentence(" ".join(doc))))
            doc = []
            continue
        m = type_re.match(line)
        if m:
            name = m.group(2)
            if not name.startswith("_"):
                items.append((m.group(1), name, first_sentence(" ".join(doc))))
            doc = []
            continue
        if s and not s.startswith("//"):
            doc = []
    if not purpose and items:
        purpose = items[0][2]
    return purpose, items


def comment_header(lines, marker):
    buf, started = [], False
    for line in lines:
        s = line.strip()
        if not started and s.startswith("#!"):
            continue
        if s.startswith(marker):
            started = True
            buf.append(s[len(marker):].strip())
        elif started and not s:
            if buf:
                break
        elif started:
            break
    return first_sentence(" ".join(x for x in buf if x))


def parse_sh(text):
    lines = text.splitlines()
    purpose = comment_header(lines, "#")
    items = []
    for idx, line in enumerate(lines):
        m = re.match(r"^([A-Za-z_]\w*)\s*\(\)", line) \
            or re.match(r"^function\s+([A-Za-z_]\w*)", line)
        if m:
            name = m.group(1)
            if name.startswith("_"):
                continue
            desc = ""
            j = idx - 1
            buf = []
            while j >= 0 and lines[j].strip().startswith("#"):
                buf.insert(0, lines[j].strip().lstrip("#").strip())
                j -= 1
            desc = first_sentence(" ".join(x for x in buf if x))
            items.append(("fn", name, desc))
    if not purpose and items:
        purpose = items[0][2]
    return purpose, items


def parse_mjs(text):
    lines = text.splitlines()
    purpose = comment_header(lines, "//")
    items = []
    for idx, line in enumerate(lines):
        m = re.match(r"^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", line) \
            or re.match(r"^(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(", line)
        if m:
            name = m.group(1)
            if name.startswith("_"):
                continue
            j = idx - 1
            buf = []
            while j >= 0 and lines[j].strip().startswith("//"):
                buf.insert(0, lines[j].strip()[2:].strip())
                j -= 1
            items.append(("fn", name, first_sentence(" ".join(x for x in buf if x))))
    return purpose, items


BANNER_RE = re.compile(r"^//\s*([-=_*#]{3,})\s*(.*?)\s*([-=_*#]{3,})\s*$")


def parse_inline_js(text):
    lines = text.splitlines()
    groups = [["(top)", []]]
    pending = []
    for line in lines:
        m = BANNER_RE.match(line)
        if m:
            name = m.group(2).strip() or "(section)"
            groups.append([name, []])
            pending = []
            continue
        m = re.match(r"^function\s+([A-Za-z_$][\w$]*)\s*\(", line)
        if m:
            name = m.group(1)
            if not name.startswith("_"):
                groups[-1][1].append((name, first_sentence(" ".join(pending))))
            pending = []
            continue
        s = line.strip()
        if s.startswith("//"):
            txt = s[2:].strip()
            if txt:
                pending.append(txt)
        else:
            pending = []
    if groups and not groups[0][1]:
        groups = groups[1:]
    return groups


def module_files():
    out = []
    for rel, _ in walk_files():
        if rel.startswith("docs/"):
            continue
        if os.path.basename(rel).startswith("_"):
            continue
        if os.path.splitext(rel)[1] in MODULE_EXTS:
            out.append(rel)
    ignored = ignored_paths(out)
    return [rel for rel in out if rel not in ignored]


def module_entry(rel):
    text = read_text(os.path.join(ROOT, rel))
    ext = os.path.splitext(rel)[1]
    if ext == ".js":
        groups = parse_inline_js(text)
        total = sum(len(g[1]) for g in groups)
        out = ['<section>', '<h3><code>%s</code></h3>' % html.escape(rel),
               '<p class="purpose">%s (%d functions)</p>' % (html.escape("browser script"), total)]
        for gname, items in groups:
            if not items:
                continue
            out.append("<details><summary>%s (%d)</summary><ul>"
                       % (html.escape(gname), len(items)))
            for name, desc in items:
                out.append("<li><code>%s</code>%s</li>"
                           % (html.escape(name),
                              " — " + html.escape(desc) if desc else ""))
            out.append("</ul></details>")
        out.append("</section>")
        return "\n".join(out)

    if ext == ".rs":
        purpose, items = parse_rs(text)
    elif ext == ".py":
        purpose, items = parse_py(text)
    elif ext == ".sh":
        purpose, items = parse_sh(text)
    elif ext == ".mjs":
        purpose, items = parse_mjs(text)
    else:
        purpose, items = "", []

    out = ['<section>', '<h3><code>%s</code></h3>' % html.escape(rel)]
    if purpose:
        out.append('<p class="purpose">%s</p>' % html.escape(purpose))
    if items:
        out.append("<ul>")
        for kind, name, desc in items:
            out.append("<li><code>%s %s</code>%s</li>"
                       % (kind, html.escape(name),
                          " — " + html.escape(desc) if desc else ""))
        out.append("</ul>")
    out.append("</section>")
    return "\n".join(out)


def build_modules_content():
    files = module_files()
    groups = {}
    for rel in files:
        top = rel.split("/")[0] if "/" in rel else "(root)"
        groups.setdefault(top, []).append(rel)
    parts = ["<p>Code map parsed from the working tree. "
             "Names starting with <code>_</code> are private and skipped.</p>"]
    for top in sorted(groups):
        parts.append("<h2>%s</h2>" % html.escape(top))
        for rel in sorted(groups[top]):
            parts.append(module_entry(rel))
    return "\n".join(parts)


# --------------------------------------------------------------------------
# commits
# --------------------------------------------------------------------------

STATUS_WORD = {"A": "added", "M": "modified", "D": "deleted", "R": "renamed",
               "C": "copied", "T": "typechange", "U": "unmerged", "X": "unknown"}


def commit_list():
    proc = subprocess.run(
        ["git", "-c", "core.quotePath=false", "rev-list", "HEAD"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return []
    commits = []
    for sha in proc.stdout.split():
        raw = git("show", "-s", "--format=%H%n%h%n%cI%n%s%n%b", sha)
        lines = raw.split("\n")
        if lines[3].strip().startswith("docs"):
            continue
        commits.append({
            "full": lines[0].strip(),
            "short": lines[0].strip()[:7],
            "date": lines[2].strip(),
            "subject": lines[3].strip(),
            "body": "\n".join(lines[4:]).strip(),
        })
    return commits


def utc_date(iso):
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def commit_stats(sha):
    num = [l for l in git("show", "--numstat", "--format=", sha).split("\n") if l.strip()]
    names = [l for l in git("show", "--name-status", "--format=", sha).split("\n") if l.strip()]
    files = []
    for idx, line in enumerate(names):
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R") or status.startswith("C"):
            old, new = parts[1], parts[2]
        else:
            old = new = parts[1]
        ins = dele = "-"
        if idx < len(num):
            nf = num[idx].split("\t")
            if len(nf) >= 3:
                ins, dele = nf[0], nf[1]
        files.append({"status": status, "old": old, "new": new,
                      "ins": ins, "del": dele})
    return files


def build_commit_page(commit, files, has_older, has_newer, readme_link):
    short, full = commit["short"], commit["full"]
    sidecar = os.path.join(DOCS, "commits", short + ".md")
    if os.path.exists(sidecar):
        rewrite = make_rewrite("docs/commits/" + short + ".md", "commits/%s.html" % short, {})
        summary = render_md(read_text(sidecar), rewrite)
    elif commit["body"]:
        summary = "<pre>" + html.escape(commit["body"]) + "</pre>"
    else:
        summary = "<p>" + html.escape(commit["subject"]) + "</p>"

    out = ["<p><strong>Commit</strong> <code>%s</code> (%s)</p>"
           % (html.escape(short), html.escape(full)),
           "<p><strong>Date</strong> %s</p>" % html.escape(utc_date(commit["date"])),
           "<p><strong>Subject</strong> %s</p>" % html.escape(commit["subject"]),
           summary,
           "<p><a href=\"index.html\">All commits</a>"]
    if has_newer:
        out.append('| <a href="%s.html">newer</a>'
                   % commit["_newer"])
    if has_older:
        out.append('| <a href="%s.html">older</a>'
                   % commit["_older"])
    if readme_link:
        out.append('| <a href="%s">README</a>' % readme_link)
    out.append("</p>")

    out.append("<h2>Changed files (%d)</h2>" % len(files))
    out.append("<table><thead><tr><th>Status</th><th>Path</th>"
               "<th>Insertions</th><th>Deletions</th></tr></thead><tbody>")
    depth_dir = os.path.join(DOCS, "commits")
    for f in files:
        code = f["status"][0]
        status = "%s %s" % (code, STATUS_WORD.get(code, ""))
        if f["old"] != f["new"]:
            disp = "%s &rarr; %s" % (html.escape(f["old"]), html.escape(f["new"]))
        else:
            disp = html.escape(f["new"])
        target = os.path.join(ROOT, f["new"])
        if os.path.isfile(target):
            href = os.path.relpath(target, depth_dir).replace(os.sep, "/")
            href = urllib.parse.quote(href)
            disp = '<a href="%s">%s</a>' % (html.escape(href, quote=True), disp)
        out.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                   % (html.escape(status), disp, html.escape(f["ins"]),
                      html.escape(f["del"])))
    out.append("</tbody></table>")
    return "\n".join(out)


def build_commits(commits, linkmap):
    pages = {}
    readme = linkmap.get("README.md")
    readme_link = "../" + readme if readme else None
    for idx, commit in enumerate(commits):
        commit["_newer"] = commits[idx - 1]["short"] if idx > 0 else None
        commit["_older"] = commits[idx + 1]["short"] if idx + 1 < len(commits) else None
        files = commit_stats(commit["full"])
        content = build_commit_page(commit, files, bool(commit["_older"]),
                                    bool(commit["_newer"]), readme_link)
        pages["commits/%s.html" % commit["short"]] = page(
            "Commit %s" % commit["short"], "git show %s" % commit["short"],
            "commits/%s.html" % commit["short"], content)

    index = ["<p>All commits, newest first.</p>"]
    if not commits:
        index.append("<p>No commits yet.</p>")
    index += ["<table><thead><tr><th>Date (UTC)</th><th>Commit</th>"
              "<th>Subject</th></tr></thead><tbody>"]
    for commit in commits:
        index.append("<tr><td>%s</td><td><a href=\"%s.html\"><code>%s</code></a></td>"
                     "<td>%s</td></tr>"
                     % (html.escape(utc_date(commit["date"])), commit["short"],
                        html.escape(commit["short"]), html.escape(commit["subject"])))
    index.append("</tbody></table>")
    pages["commits/index.html"] = page(
        "Commit history", "git log", "commits/index.html", "\n".join(index))
    return pages


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    global GEN_TIME, NAV_GROUPS
    GEN_TIME = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    os.makedirs(os.path.join(DOCS, "pages"), exist_ok=True)
    os.makedirs(os.path.join(DOCS, "commits"), exist_ok=True)

    # --- discover markdown sources -------------------------------------
    md = []
    for rel, _ in walk_files():
        if not rel.endswith(".md"):
            continue
        if rel == "docs/overview.md" or rel.startswith("docs/commits/"):
            continue
        md.append(rel)
    # docs/ is git-ignored on purpose (it stays local), but its Markdown is ours.
    ignored = ignored_paths([r for r in md if not r.startswith("docs/")])
    md = [r for r in md if r not in ignored]
    md.sort()

    # slug map, unique
    used, linkmap = {}, {}
    for rel in md:
        slug = slugify(rel)
        if slug in used:
            slug = slug + "-" + hashlib.sha1(rel.encode()).hexdigest()[:8]
        used[slug] = rel
        linkmap[rel] = "pages/%s.html" % slug
    if os.path.exists(os.path.join(ROOT, "docs/overview.md")):
        linkmap["docs/overview.md"] = "index.html"

    # --- nav groups -----------------------------------------------------
    groups = {}
    for rel in md:
        text = read_text(os.path.join(ROOT, rel))
        top = rel.split("/")[0] if "/" in rel else ""
        if rel.startswith("docs/notes/"):
            section = "Notes"
            label = first_heading(text) or os.path.basename(rel)
        elif top and os.path.exists(os.path.join(ROOT, top, "TASK.md")):
            section = "Problems"
            label = "%s %s" % (top, os.path.splitext(os.path.basename(rel))[0])
        elif rel.startswith("docs/"):
            section = "Docs"
            label = first_heading(text) or os.path.basename(rel)
        else:
            section = "Repository"
            label = first_heading(text) or os.path.basename(rel)
        groups.setdefault(section, []).append((label, linkmap[rel]))
    NAV_GROUPS = []
    for section in ("Notes", "Problems", "Docs", "Repository"):
        if section in groups:
            NAV_GROUPS.append(
                (section, sorted(groups[section], key=lambda x: x[1])))

    commits = commit_list()

    written = []

    def emit(out_path, html_text):
        target = os.path.join(DOCS, out_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(target):
            with open(target, "r", encoding="utf-8", errors="replace") as fh:
                old = fh.read()
            if stable(old) == stable(html_text):
                return
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(html_text)
        written.append("docs/" + out_path)

    # --- landing page (always rewritten) --------------------------------
    if os.path.exists(os.path.join(ROOT, "docs/overview.md")):
        src = "docs/overview.md"
        overview = read_text(os.path.join(ROOT, src))
    else:
        src = "README.md"
        overview = read_text(os.path.join(ROOT, src))
    rewrite = make_rewrite(src, "index.html", linkmap)
    content = render_md(overview, rewrite)
    content += "\n<h2>Newest commits</h2>\n"
    if commits:
        content += "<ul>"
        for commit in commits[:5]:
            content += '<li>%s — <a href="commits/%s.html"><code>%s</code></a> %s</li>' \
                % (html.escape(utc_date(commit["date"])), commit["short"],
                   html.escape(commit["short"]), html.escape(commit["subject"]))
        content += "</ul>"
    else:
        content += "<p>No commits yet.</p>"
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page("Overview", src, "index.html", content))
    written.append("docs/index.html")

    # --- code map (always rewritten) ------------------------------------
    with open(os.path.join(DOCS, "modules.html"), "w", encoding="utf-8") as fh:
        fh.write(page("Code map", "generated code map", "modules.html",
                      build_modules_content()))
    written.append("docs/modules.html")

    # --- markdown pages -------------------------------------------------
    for rel in md:
        out_path = linkmap[rel]
        rewrite = make_rewrite(rel, out_path, linkmap)
        text = read_text(os.path.join(ROOT, rel))
        content = render_md(text, rewrite)
        title = first_heading(text) or os.path.basename(rel)
        emit(out_path, page(title, rel, out_path, content))

    # --- stale rendered pages (never touches docs/commits/) --------------
    emitted = {linkmap[rel] for rel in md}
    pages_dir = os.path.join(DOCS, "pages")
    for name in sorted(os.listdir(pages_dir)):
        if name.endswith(".html") and "pages/" + name not in emitted:
            os.remove(os.path.join(pages_dir, name))
            written.append("removed docs/pages/" + name)

    # --- commit pages and commit index ----------------------------------
    commit_pages = build_commits(commits, linkmap)
    for out_path in sorted(commit_pages):
        if out_path == "commits/index.html":
            continue
        emit(out_path, commit_pages[out_path])
    with open(os.path.join(DOCS, "commits", "index.html"), "w", encoding="utf-8") as fh:
        fh.write(commit_pages["commits/index.html"])
    written.append("docs/commits/index.html")

    # --- report ---------------------------------------------------------
    for path in sorted(set(written)):
        print(path)

    # --- link check -----------------------------------------------------
    broken = []
    for dirpath, dirnames, filenames in os.walk(DOCS):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in sorted(filenames):
            if not name.endswith(".html"):
                continue
            page_path = os.path.join(dirpath, name)
            for href in re.findall(r'href="([^"]*)"', read_text(page_path)):
                if not href or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", href) \
                        or href.startswith("#") or href.startswith("//"):
                    continue
                path = urllib.parse.unquote(href.split("#")[0].split("?")[0])
                if not path.endswith(".html"):
                    continue
                target = os.path.normpath(os.path.join(dirpath, path))
                if not os.path.exists(target):
                    broken.append("%s -> %s" % (rel_posix(page_path), href))
    if broken:
        for item in broken:
            print("broken link: " + item, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
