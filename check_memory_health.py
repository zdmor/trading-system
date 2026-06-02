"""Memory health checker — orphan/broken-link/backlink/stale audit."""
import os
import re
import glob
import time

MEMORY_DIR = r"C:\Users\sut-b\.claude\projects\D--ClaudeWorkspace\memory"
VAULT_DIR = r"D:\DiskMigration\MySecondBrain"
LINK_RE = re.compile(r"\[\[([^#|\[\]]+?)(?:#[^|]*?)?(?:\|[^]]*?)?\]\]")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
STALE_DAYS = 60


def collect_md_files(root):
    """Return {basename_without_ext: full_path} for all .md files."""
    result = {}
    for p in glob.glob(os.path.join(root, "**", "*.md"), recursive=True):
        key = os.path.splitext(os.path.basename(p))[0]
        result[key] = p
    return result


def collect_md_links(root):
    """Return {source_path: [linked_slug, ...]} across all md files."""
    source_links = {}
    for fpath in glob.glob(os.path.join(root, "**", "*.md"), recursive=True):
        with open(fpath, encoding="utf-8") as f:
            text = f.read()
        matches = LINK_RE.findall(text)
        # deduplicate per file
        seen = set()
        uniq = []
        for m in matches:
            slug = os.path.splitext(m.replace("\\", "/").split("/")[-1])[0]
            if slug not in seen:
                seen.add(slug)
                uniq.append(slug)
        if uniq:
            source_links[fpath] = uniq
    return source_links


def rel(path, start=os.curdir):
    """Cross-drive-safe relpath: fall back to short absolute."""
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return path


def main():
    sep = "-" * 56

    # memory only for backlink analysis (vault is too large for meaningful wikilink graph)
    mem_files = collect_md_files(MEMORY_DIR)
    mem_links = collect_md_links(MEMORY_DIR)

    # track which memory filenames are referenced by other memory files
    file_ref_count = {f: 0 for f in mem_files.values()}
    for source, targets in mem_links.items():
        for t in targets:
            if t in mem_files:
                file_ref_count[mem_files[t]] += 1

    print("=" * 56)
    print("  MEMORY HEALTH CHECK")
    print("=" * 56)

    # 1 — orphans: memory files not referenced by any other memory file
    print(f"\n{sep}")
    print("  1. ORPHAN FILES (no other memory file wikilinks to it)")
    print(sep)
    orphans = [(f, file_ref_count[f]) for f in mem_files.values() if file_ref_count[f] == 0]
    if orphans:
        for fpath, _ in sorted(orphans):
            short = rel(fpath, MEMORY_DIR)
            age = (time.time() - os.path.getmtime(fpath)) / 86400
            print(f"    {short} ({age:.0f}d)")
    else:
        print("    (none)")
    print(f"\n    ({sum(1 for _, c in file_ref_count.items() if c > 0)}/{len(file_ref_count)} files have inbound links)")

    # 2 — broken wikilinks in memory files
    print(f"\n{sep}")
    print("  2. BROKEN WIKILINKS (memory → non-existent)")
    print(sep)
    broken = 0
    for source, targets in sorted(mem_links.items()):
        for t in targets:
            if t not in mem_files:
                short_src = rel(source, MEMORY_DIR)
                print(f"    {short_src} → [[{t}]]")
                broken += 1
    if not broken:
        print("    (none)")

    # 3 — MEMORY.md index coverage check
    print(f"\n{sep}")
    print("  3. MEMORY.md INDEX COVERAGE")
    print(sep)
    mem_index_path = os.path.join(MEMORY_DIR, "MEMORY.md")
    with open(mem_index_path, encoding="utf-8") as f:
        idx_text = f.read()
    idx_lower = idx_text.lower()
    missing = []
    for fname, fpath in sorted(mem_files.items()):
        if fname == "MEMORY":
            continue
        relp = rel(fpath, MEMORY_DIR)
        # listed by path or by wikilink or by name in frontmatter
        if relp.lower() in idx_lower or f"[{fname}]" in idx_text or f"[[{fname}]]" in idx_text:
            continue
        # check if the file's name/description appears in MEMORY.md somewhere
        with open(fpath, encoding="utf-8") as f:
            head = f.read(512)
        fm = FRONTMATTER_RE.search(head)
        title = ""
        desc = ""
        if fm:
            for line in fm.group(1).split("\n"):
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip().strip('"').lower()
                elif line.startswith("description:"):
                    desc = line.split(":", 1)[1].strip().strip('"').lower()
        if title and title in idx_lower:
            continue
        if desc and desc in idx_lower:
            continue
        missing.append(relp)
    if missing:
        for m in missing:
            print(f"    not indexed: {m}")
    else:
        print("    all memory files indexed in MEMORY.md")

    # 4 — stale files (memory only)
    print(f"\n{sep}")
    print(f"  4. STALE (> {STALE_DAYS}d since modification)")
    print(sep)
    now = time.time()
    stale = []
    for fpath in mem_files.values():
        age = (now - os.path.getmtime(fpath)) / 86400
        if age > STALE_DAYS:
            stale.append((age, fpath))
    if stale:
        for age, fpath in sorted(stale, reverse=True):
            short = rel(fpath, MEMORY_DIR)
            print(f"    {short} ({age:.0f}d)")
    else:
        print("    (none)")

    # 5 — backlink ranking (top 15 of memory)
    print(f"\n{sep}")
    print("  5. BACKLINK RANKING (memory → memory, top 15)")
    print(sep)
    ranked = sorted(file_ref_count.items(), key=lambda x: -x[1])
    shown = 0
    for fpath, count in ranked:
        if count > 0:
            short = rel(fpath, MEMORY_DIR)
            print(f"    {count:3d}x  {short}")
            shown += 1
            if shown >= 15:
                break
    if shown == 0:
        print("    (no inter-memory links found)")


if __name__ == "__main__":
    main()
