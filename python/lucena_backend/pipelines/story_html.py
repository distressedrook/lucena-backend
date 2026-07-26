"""The game story, rendered as a standalone HTML page.

Pure presentation over `gamestory.build_story`'s artifact: this module invents
NO chess content. Every claim on the page traces to a number the engine
produced or a phrase a detector wrote — if the story is silent about something,
the page is silent too.

Self-contained by construction (inline CSS, inline SVG boards, no network), so
the output can be opened from disk, emailed, or dropped into the site scaffold
unchanged.

Brand: "Annotated Board" — paper and ink, no gradients. Lora is the brand
serif and is named first in the stack; the page degrades to a system serif
where it is not installed, which is the honest fallback for a file that has to
work standalone.
"""

from __future__ import annotations

import html
import re

import chess
import chess.svg

# The paper-and-ink palette, in one place.
PAPER = "#F1EDE3"
INK = "#1A1A1A"
MUTED = "#6B6459"
RULE = "#D9D1C2"
WHITE_SQ = "#EDE6D8"
BLACK_SQ = "#B9AE97"

# Move-class presentation. Order is the ladder order; colour carries the
# judgement so a reader can scan the move list without reading a word.
CLASS_STYLE = {
    "brilliant":  ("!!", "#0F8B8D", "Brilliant"),
    "great":      ("!",  "#2E7D5B", "Great"),
    "best":       ("✓",  "#4A7A4A", "Best"),
    "excellent":  ("✓",  "#6E8B5A", "Excellent"),
    "good":       ("",   "#7A7A6A", "Good"),
    "book":       ("○",  "#8A8172", "Book"),
    "inaccuracy": ("?!", "#B8860B", "Inaccuracy"),
    "mistake":    ("?",  "#C1622B", "Mistake"),
    "miss":       ("×",  "#A6392E", "Missed win"),
    "blunder":    ("??", "#8B1E1E", "Blunder"),
}

TIER_STYLE = {
    "engine":    ("Engine confirmed", "#2E7D5B"),
    "human":     ("Strong humans play this", "#3A6EA5"),
    "structure": ("The structure suggests this", "#8A8172"),
}

KIND_TITLE = {
    "turning": "Turning point",
    "missed":  "Missed win",
    "drift":   "The slow slide",
    "plan":    "The plan on offer",
}


def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _board_svg(fen: str, lastmove_uci: str = "", size: int = 300) -> str:
    """One position as inline SVG, in the paper palette."""
    if not fen:
        return ""
    try:
        b = chess.Board(fen)
        mv = chess.Move.from_uci(lastmove_uci) if lastmove_uci else None
        return chess.svg.board(
            b, size=size, lastmove=mv, coordinates=True,
            colors={"square light": WHITE_SQ, "square dark": BLACK_SQ,
                    "margin": "#C8BFA9", "coord": "#4A4438",
                    "square light lastmove": "#D8CE8F",
                    "square dark lastmove": "#B3A860"},
        )
    except Exception:
        return ""


def _md(text: str) -> str:
    """The position read's own light markup -> HTML.

    `position_read` emits **bold** headers, _italic_ evidence tags and "- "
    bullets. Nothing else is interpreted; the text is escaped first, so a
    detector string can never inject markup."""
    if not text:
        return ""
    out, buf = [], []

    def flush():
        if buf:
            out.append("<ul class='read'>" + "".join(buf) + "</ul>")
            buf.clear()

    for line in text.split("\n"):
        s = _e(line.strip())
        if not s:
            flush()
            continue
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"_(.+?)_", r"<em>\1</em>", s)
        if s.startswith("- "):
            buf.append(f"<li>{s[2:]}</li>")
        else:
            flush()
            out.append(f"<p>{s}</p>")
    flush()
    return "".join(out)


def _curve(story: dict, width: int = 900, height: int = 150) -> str:
    """The eval curve: White's advantage above the midline, Black's below.

    Clamped to ±600cp because beyond that the shape stops carrying information
    — the point of the picture is WHEN the game turned, not how large a won
    position got."""
    plies = story["plies"]
    if not plies:
        return ""
    cap, n = 600.0, len(plies)
    def x(i): return round(i * width / max(1, n - 1), 1)
    def y(cp): return round(height / 2 - max(-cap, min(cap, cp)) / cap * (height / 2), 1)

    pts = [(x(i), y(p.get("cp_white", 0))) for i, p in enumerate(plies)]
    area = (f"M0,{height/2} " + " ".join(f"L{a},{b}" for a, b in pts)
            + f" L{width},{height/2} Z")
    line = "M" + " L".join(f"{a},{b}" for a, b in pts)

    marks = []
    for m in story["moments"]:
        i = m["ply"] - 1
        if not (0 <= i < n):
            continue
        col = CLASS_STYLE.get(m["label"], ("", MUTED, ""))[1]
        if m["kind"] == "plan":
            col = "#3A6EA5"
        marks.append(
            f"<circle cx='{x(i)}' cy='{y(plies[i].get('cp_white', 0))}' r='4.5' "
            f"fill='{col}' stroke='var(--paper)' stroke-width='1.5'>"
            f"<title>{_e(m['move_no'])}{'.' if m['side']=='w' else '...'} "
            f"{_e(m['san'])} — {_e(KIND_TITLE.get(m['kind'], ''))}</title></circle>")

    return f"""<svg viewBox="0 0 {width} {height}" class="curve" role="img"
 aria-label="Evaluation across the game">
 <rect x="0" y="0" width="{width}" height="{height/2}" fill="var(--crest)"/>
 <rect x="0" y="{height/2}" width="{width}" height="{height/2}" fill="var(--trough)"/>
 <path d="{area}" fill="var(--fill)" opacity="0.35"/>
 <path d="{line}" fill="none" stroke="var(--curve)" stroke-width="1.6"/>
 <line x1="0" y1="{height/2}" x2="{width}" y2="{height/2}"
       stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 3"/>
 {''.join(marks)}
</svg>"""


def _ending_line(story: dict) -> str:
    """How the game stopped. For a repetition draw this IS the story, and the
    first quiet game analysed here printed four chapters about castling while
    saying nothing about the shuffle that ended it."""
    e = story.get("ending") or {}
    rep = e.get("repetition")
    if not rep:
        return ""
    cp = rep.get("cp_white", 0)
    read = ("with the position level — a genuine standoff, neither side able "
            "to make progress" if abs(cp) < 50 else
            f"with {'White' if cp > 0 else 'Black'} still holding the better "
            f"position — the better game was let go")
    moves = " ".join(rep.get("moves") or [])
    return f"""<div class="ending">
      <h3>How it ended: repetition</h3>
      <p>The same position appeared {rep['times']} times, {_e(read)}.</p>
      <p class="pv">from move {rep['from_move']}: {_e(moves)}</p>
    </div>"""


def _track_svg(vals, plies, *, cap, colour, width=900, height=86):
    """One signed track, White-positive above the midline.

    Deliberately the same width and x-mapping as the eval curve so a reader can
    drop a vertical line through all three charts and read one position."""
    n = len(vals)
    if not n:
        return ""
    def x(i): return round(i * width / max(1, n - 1), 1)
    def y(v): return round(height / 2 - max(-cap, min(cap, v)) / cap * (height / 2), 1)
    pts = [(x(i), y(v)) for i, v in enumerate(vals)]
    area = (f"M0,{height/2} " + " ".join(f"L{a},{b}" for a, b in pts)
            + f" L{width},{height/2} Z")
    line = "M" + " L".join(f"{a},{b}" for a, b in pts)
    ticks = "".join(
        f"<line x1='{x(i)}' y1='0' x2='{x(i)}' y2='{height}' "
        f"stroke='var(--rule)' stroke-width='1'/>"
        for i, p in enumerate(plies) if p["move_no"] % 10 == 0 and p["side"] == "w")
    return f"""<svg viewBox="0 0 {width} {height}" class="track" role="img">
 {ticks}
 <path d="{area}" fill="{colour}" opacity="0.22"/>
 <path d="{line}" fill="none" stroke="{colour}" stroke-width="1.5"/>
 <line x1="0" y1="{height/2}" x2="{width}" y2="{height/2}"
       stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 3"/>
</svg>"""


def _tracks(story: dict) -> str:
    """Activity and initiative across the game — what the eval curve cannot say.

    Evaluation answers who is BETTER. These answer why: whose pieces are doing
    more work, and who is dictating. They disagree with the eval often, and
    that disagreement is the point — being worse with the initiative is a
    different game from being worse and passive."""
    t = story.get("tracks") or {}
    a, i = t.get("activity") or [], t.get("initiative") or []
    if not a and not i:
        return ""
    plies = story["plies"]
    bases = t.get("bases") or {}
    # initiative appends a face to the basis ("geometry-prior+development"),
    # so an exact-key lookup silently under-reports the fallback and the page
    # would claim engine backing it does not have.
    prior = sum(n for k, n in bases.items() if k.startswith("geometry-prior"))
    note = ""
    if prior:
        note = (f" {prior} of {sum(bases.values())} positions had no second "
                f"engine line and fall back to the weaker geometry reading.")
    return f"""<section id="tracks">
  <h2>Activity and initiative</h2>
  <p class="lede">The evaluation says who is better. These say why — whose
     pieces are doing more work, and who is dictating play. Both are drawn
     White-positive above the centre line, on the same scale of moves as the
     curve above.</p>
  <div class="track-row">
    <div class="track-label"><b>Activity</b>
      <span>how much more work one side's pieces are doing</span></div>
    {_track_svg(a, plies, cap=200.0, colour="#3A6EA5")}
  </div>
  <div class="track-row">
    <div class="track-label"><b>Initiative</b>
      <span>who is making the threats</span></div>
    {_track_svg(i, plies, cap=1.0, colour="#A6392E")}
  </div>
  <p class="note">Initiative is read from the engine's best-vs-second-line
     spread — the side whose alternatives fall away is the side being
     dictated to.{_e(note)}</p>
</section>"""


def _acts(story: dict) -> str:
    """The arc, as a sentence per act."""
    say = {"level": "level", "edge": "a slight edge for {}",
           "clear": "clearly better for {}", "winning": "winning for {}"}
    rows = []
    for a in story["arc"]:
        who = (a["who"] or "").capitalize()
        text = say[a["kind"]].format(who) if a["who"] else say[a["kind"]]
        span = (f"move {a['from_move']}" if a["from_move"] == a["to_move"]
                else f"moves {a['from_move']}–{a['to_move']}")
        rows.append(f"<li><span class='span'>{_e(span)}</span>"
                    f"<span class='act'>{_e(text)}</span></li>")
    return "<ol class='arc'>" + "".join(rows) + "</ol>"


def _chips(counts: dict) -> str:
    out = []
    for k, n in counts.items():
        sym, col, name = CLASS_STYLE.get(k, ("", MUTED, k.title()))
        out.append(
            f"<span class='chip'><b style='color:{col}'>{_e(sym or '·')}</b>"
            f"{_e(name)}<i>{n}</i></span>")
    return "".join(out)


def _plans_block(m: dict) -> str:
    """The plan menus, tier-tagged, both sides.

    NOT used in the moment card: `position_read` already renders exactly this,
    with the qualifiers ("available right now", "playable in the short term")
    that the raw menu loses. Kept for a caller that wants the bare list."""
    out = []
    for tag, who in (("white", "White"), ("black", "Black")):
        rows = []
        for p in (m.get("plans") or {}).get(tag) or []:
            label, col = TIER_STYLE.get(p["tier"], TIER_STYLE["structure"])
            rows.append(
                f"<li><span class='tier' style='--t:{col}'>{_e(label)}</span>"
                f"<span class='idea'>{_e(p['idea'])}</span></li>")
        if rows:
            out.append(f"<div class='menu'><h4>{who}</h4><ul>"
                       + "".join(rows) + "</ul></div>")
    return f"<div class='menus'>{''.join(out)}</div>" if out else ""


def _endgame_block(m: dict) -> str:
    """The endgame facts, where there are any. Kept visually distinct from the
    plans read because it carries a different contract — geometry a reader can
    check, with no engine or corpus claim behind it."""
    lines = m.get("endgame") or []
    if not lines:
        return ""
    items = "".join(f"<li>{_e(x)}</li>" for x in lines)
    return (f"<div class='endgame-read'><h4>The endgame</h4>"
            f"<ul>{items}</ul></div>")


def _alignment_note(m: dict) -> str:
    """The sentence that is ours alone: plan on offer vs plan actually played."""
    a = m.get("alignment") or {}
    st = a.get("state")
    fams = ", ".join((a.get("families") or a.get("played") or [])).replace("_", " ")
    if st == "right-idea-wrong-move":
        return (f"<p class='align right'><b>Right idea, wrong move.</b> "
                f"The continuation played does carry out <b>{_e(fams)}</b> — "
                f"a plan this position genuinely offers. The idea was not the "
                f"error; this move was.</p>")
    if st == "on-plan":
        return (f"<p class='align on'><b>On plan.</b> What followed carried out "
                f"<b>{_e(fams)}</b>, one of the plans on offer here.</p>")
    if st == "different-plan":
        return (f"<p class='align other'><b>A different plan.</b> What followed "
                f"carried out <b>{_e(fams)}</b>, which is not among the plans "
                f"detected here. That is not automatically wrong — our menu is "
                f"not exhaustive.</p>")
    return ""


def _moment_html(m: dict, headers: dict) -> str:
    sym, col, name = CLASS_STYLE.get(m["label"], ("", MUTED, m["label"].title()))
    mover = headers.get("White" if m["side"] == "w" else "Black", "?")
    num = f"{m['move_no']}{'.' if m['side'] == 'w' else '…'}"
    kind = KIND_TITLE.get(m["kind"], m["kind"].title())

    if m["kind"] == "drift":
        moves = " · ".join(m.get("span_moves") or [])
        drift = (f"<p class='cost'>No single mistake here. Between moves "
                 f"{m['move_no']} and {m.get('span_to_move')}, "
                 f"<b>{m.get('span_cost', 0):.1f}</b> points of winning "
                 f"chances went in {len(m.get('span_moves') or [])} small "
                 f"concessions.</p>"
                 f"<p class='pv'>{_e(moves)}</p>")
    else:
        drift = ""

    cost = ""
    if m["kind"] not in ("plan", "drift") and m["delta_win_pct"] < 0:
        cost = (f"<p class='cost'>Cost: <b>{abs(m['delta_win_pct']):.1f}</b> "
                f"points of winning chances.</p>")
    gift = ""
    if m["kind"] == "missed" and m.get("gift_wp"):
        gift = (f"<p class='gift'>The move before handed over "
                f"<b>{m['gift_wp']:.1f}</b> points — and this move gave them "
                f"back.</p>")

    better = ""
    if (m["kind"] not in ("plan", "drift") and m.get("best_san")
            and m["best_san"] != m["san"]):
        # the PV opens WITH the best move — printing both reads "Instead c3 c3"
        pv = " ".join((m.get("best_pv_san") or [])[1:])
        better = (f"<p class='line'><span class='k'>Instead</span> "
                  f"<b>{_e(m['best_san'])}</b>"
                  + (f" <span class='pv'>{_e(pv)}</span>" if pv else "") + "</p>")
    refut = ""
    if m.get("refutation_pv"):
        refut = (f"<p class='line'><span class='k'>Refutation</span> "
                 f"<span class='pv'>{_e(' '.join(m['refutation_pv']))}</span></p>")

    facts = []
    if m.get("character"):
        facts.append(f"<span class='fact'>{_e(m['character_why'] or m['character'])}</span>")
    ini = m.get("initiative") or {}
    if ini.get("leader"):
        mech = ", ".join(ini.get("mechanism") or [])
        facts.append(f"<span class='fact'>Initiative: {_e(ini['leader'])}"
                     + (f" ({_e(mech)})" if mech else "") + "</span>")
    if m.get("only_move"):
        facts.append("<span class='fact'>Only one move holds this</span>")
    motifs = sorted({x["motif"] for x in (m.get("motifs") or [])})
    if motifs:
        facts.append(f"<span class='fact'>Tactics in the air: "
                     f"{_e(', '.join(motifs))}</span>")

    return f"""<article class="moment {m['kind']}">
  <header>
    <span class="kind">{_e(kind)}</span>
    <h3><span class="num">{_e(num)}</span> {_e(m['san'])}
        <span class="badge" style="--c:{col}">{_e(sym or '·')} {_e(name)}</span></h3>
    <p class="by">{_e(mover)}</p>
  </header>
  <div class="body">
    <div class="diagram">{_board_svg(m['fen_before'], m.get('uci', ''))}</div>
    <div class="prose">
      {drift}{cost}{gift}{better}{refut}
      <div class="facts">{''.join(facts)}</div>
      {_alignment_note(m)}
      <div class="read-block">{_md(m.get('read', ''))}</div>
      {_endgame_block(m)}
    </div>
  </div>
</article>"""


def _movelist(story: dict) -> str:
    rows, plies = [], story["plies"]
    for i in range(0, len(plies), 2):
        w = plies[i]
        b = plies[i + 1] if i + 1 < len(plies) else None

        def cell(p):
            if p is None:
                return "<td></td>"
            sym, col, name = CLASS_STYLE.get(p["label"], ("", MUTED, ""))
            mark = (f"<sup style='color:{col}' title='{_e(name)}'>{_e(sym)}</sup>"
                    if sym else "")
            return (f"<td><span class='mv'>{_e(p['san'])}</span>{mark}"
                    f"<span class='cp'>{p.get('cp_white', 0)/100:+.1f}</span></td>")
        rows.append(f"<tr><th>{w['move_no']}</th>{cell(w)}{cell(b)}</tr>")
    return ("<table class='moves'><thead><tr><th></th><th>White</th>"
            "<th>Black</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def _bank(story: dict, headers: dict) -> str:
    """Aim 2: the tactics that were on the board and went unplayed."""
    items = [m for m in story["moments"] if m["kind"] == "missed"]
    if not items:
        return ""
    rows = []
    for m in items:
        who = headers.get("White" if m["side"] == "w" else "Black", "?")
        num = f"{m['move_no']}{'.' if m['side'] == 'w' else '…'}"
        rows.append(f"""<li>
        <div class="mini">{_board_svg(m['fen_before'], '', 190)}</div>
        <div>
          <p class="who">{_e(who)}, after {_e(num)} {_e(m['san'])}</p>
          <p class="had">The winning move was <b>{_e(m['best_san'])}</b>
             <span class="pv">{_e(' '.join(m.get('best_pv_san') or []))}</span></p>
          <p class="worth">Worth {m['gift_wp']:.1f} points of winning chances.</p>
        </div></li>""")
    return f"""<section id="bank">
  <h2>The tactics bank</h2>
  <p class="lede">Positions where a win was on the board and went unplayed.
     These are the drills this game earned.</p>
  <ul class="bank">{''.join(rows)}</ul></section>"""


def _patterns(story: dict) -> str:
    p = story["patterns"]
    cards = []
    for side in ("w", "b"):
        d = p[side]
        ph = d["by_phase"]
        worst = max(ph, key=lambda k: ph[k]) if any(ph.values()) else None
        motifs = ", ".join(f"{k} ({v})" for k, v in d["motifs"][:4])
        cards.append(f"""<div class="card">
      <h3>{_e(d['name'])}</h3>
      <p class="big">{d['accuracy']}<span>accuracy</span></p>
      <dl>
        <dt>Errors</dt><dd>{d['errors']} of {d['moves']} moves ({d['error_rate']}%)</dd>
        <dt>Where</dt><dd>opening {ph.get('opening',0)} · middlegame
            {ph.get('middlegame',0)} · endgame {ph.get('endgame',0)}
            {f"— concentrated in the {worst}" if worst else ""}</dd>
        <dt>Wins let go</dt><dd>{d['missed']}</dd>
        <dt>Motifs at the sharp moments</dt><dd>{_e(motifs) or '—'}</dd>
      </dl>
      <div class="chips">{_chips(d['counts'])}</div></div>""")
    return f"""<div class="cards">{''.join(cards)}</div>"""


def render(story: dict) -> str:
    h = story["headers"]
    g = story["game"]
    white, black = h.get("White", "?"), h.get("Black", "?")
    we, be = h.get("WhiteElo", ""), h.get("BlackElo", "")
    term = h.get("Termination", "")
    result = g.get("result") or h.get("Result", "")

    op = story.get("opening") or {}
    opening_line = ""
    if op.get("name"):
        opening_line = (f"<p class='opening'>{_e(op['name'])}"
                        + (f" <span>— theory to move {op['left_at_move']}</span>"
                           if op.get("left_at_move") else "") + "</p>")

    hidden = sum(m.get("plans_hidden", 0) for m in story["moments"])
    hidden_note = ""
    if hidden:
        hidden_note = (
            f"<br>{hidden} further plan{'s' if hidden != 1 else ''} were "
            f"detected across these positions but not printed: either the "
            f"engine did not confirm them here, or they were near-universal "
            f"advice that fits almost any position. They remain in the "
            f"underlying data.")

    chapters = "".join(_moment_html(m, h) for m in story["moments"])
    dropped = story.get("moments_dropped", 0)
    note = ""
    if dropped:
        note = (f"<p class='note'>{dropped} further moment"
                f"{'s' if dropped != 1 else ''} of the same kind did not make "
                f"the walkthrough; every move is still classified in the move "
                f"list below.</p>")

    return f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(white)} vs {_e(black)} — Lucena</title>
<style>
 /* Paper and ink is the brand (CLAUDE.md, "Annotated Board"). Dark mode is
    not an inversion of it — it is the same page printed on a dark stock:
    the neutrals keep their warm bias, the board keeps its contrast, and the
    move-quality hues are lifted rather than swapped so green still reads as
    approval on a dark ground. */
 :root {{
   --paper:{PAPER}; --ink:{INK}; --muted:{MUTED}; --rule:{RULE};
   --card:#FBF8F1; --hair:#EAE4D7; --trough:#E3DCCC; --crest:#FBF8F1;
   --curve:{INK}; --fill:#8B8574; --pv:#4A4A4A;
 }}
 @media (prefers-color-scheme: dark) {{
   :root {{
     --paper:#16150F; --ink:#EDE7D9; --muted:#9A917F; --rule:#332F26;
     --card:#1E1C15; --hair:#2A271F; --trough:#221F18; --crest:#12110C;
     --curve:#EDE7D9; --fill:#6E6858; --pv:#B5AD9B;
   }}
 }}
 :root[data-theme="dark"] {{
   --paper:#16150F; --ink:#EDE7D9; --muted:#9A917F; --rule:#332F26;
   --card:#1E1C15; --hair:#2A271F; --trough:#221F18; --crest:#12110C;
   --curve:#EDE7D9; --fill:#6E6858; --pv:#B5AD9B;
 }}
 :root[data-theme="light"] {{
   --paper:{PAPER}; --ink:{INK}; --muted:{MUTED}; --rule:{RULE};
   --card:#FBF8F1; --hair:#EAE4D7; --trough:#E3DCCC; --crest:#FBF8F1;
   --curve:{INK}; --fill:#8B8574; --pv:#4A4A4A;
 }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--paper); color:var(--ink);
   font-family:Lora,"Iowan Old Style",Georgia,serif; line-height:1.55;
   -webkit-font-smoothing:antialiased; }}
 .wrap {{ max-width:980px; margin:0 auto; padding:32px 20px 80px; }}
 h1,h2,h3,h4 {{ font-weight:600; margin:0; letter-spacing:-0.01em; }}
 h2 {{ font-size:26px; margin:0 0 6px; }}
 .lede {{ color:var(--muted); margin:0 0 20px; max-width:62ch; }}
 section {{ margin:48px 0; }}
 .rule {{ height:1px; background:var(--rule); border:0; margin:0; }}

 header.mast {{ display:flex; align-items:baseline; gap:12px;
   border-bottom:2px solid var(--ink); padding-bottom:10px; }}
 .mark {{ font-size:26px; font-weight:700; letter-spacing:-0.03em; }}
 .mast .meta {{ margin-left:auto; color:var(--muted); font-size:13px;
   font-family:Menlo,monospace; }}
 .players {{ display:flex; align-items:center; gap:14px; margin:26px 0 4px;
   flex-wrap:wrap; }}
 .players h1 {{ font-size:32px; }}
 .players .res {{ font-family:Menlo,monospace; font-size:20px;
   border:1px solid var(--ink); padding:2px 10px; }}
 .sub {{ color:var(--muted); font-size:14px; margin:0 0 4px; }}

 .opening {{ font-size:15px; margin:2px 0 0; }}
 .opening span {{ color:var(--muted); font-size:13px; }}
 .ending {{ border:1px solid var(--rule); background:var(--card);
   padding:14px 16px; margin-top:20px; }}
 .ending h3 {{ font-size:13px; letter-spacing:0.08em; text-transform:uppercase;
   color:var(--muted); margin-bottom:6px; }}
 .ending p {{ margin:0 0 4px; }}
 .curve {{ width:100%; height:auto; display:block; margin:10px 0 4px;
   border:1px solid var(--rule); }}
 .curve-key {{ display:flex; justify-content:space-between;
   font-family:Menlo,monospace; font-size:11px; color:var(--muted); }}
 .track {{ width:100%; height:auto; display:block;
   border:1px solid var(--rule); }}
 .track-row {{ margin:14px 0; }}
 .track-label {{ display:flex; align-items:baseline; gap:10px; margin-bottom:3px; }}
 .track-label b {{ font-size:14px; font-weight:600; }}
 .track-label span {{ color:var(--muted); font-size:12px; }}
 ol.arc {{ list-style:none; padding:0; margin:18px 0 0;
   column-width:250px; column-gap:28px; }}
 ol.arc li {{ display:flex; gap:10px; padding:3px 0;
   break-inside:avoid; }}
 .span {{ font-family:Menlo,monospace; font-size:12px; color:var(--muted);
   min-width:104px; }}

 .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
   gap:20px; }}
 .card {{ border:1px solid var(--rule); padding:18px; background:var(--card); }}
 .card h3 {{ font-size:19px; }}
 .big {{ font-family:Menlo,monospace; font-size:40px; margin:6px 0 12px;
   line-height:1; }}
 .big span {{ font-family:Lora,Georgia,serif; font-size:12px;
   color:var(--muted); margin-left:8px; letter-spacing:0.06em;
   text-transform:uppercase; }}
 dl {{ margin:0 0 14px; font-size:14px; }}
 dt {{ color:var(--muted); font-size:11px; text-transform:uppercase;
   letter-spacing:0.07em; margin-top:9px; }}
 dd {{ margin:1px 0 0; }}
 .chips {{ display:flex; flex-wrap:wrap; gap:5px; }}
 .chip {{ display:inline-flex; align-items:center; gap:5px; font-size:11.5px;
   border:1px solid var(--rule); padding:2px 7px; background:var(--paper); }}
 .chip i {{ font-style:normal; font-family:Menlo,monospace;
   color:var(--muted); }}
 .chip b {{ font-family:Menlo,monospace; }}

 .moment {{ border-top:1px solid var(--rule); padding:26px 0 6px; }}
 .moment .kind {{ font-size:11px; letter-spacing:0.1em; text-transform:uppercase;
   color:var(--muted); }}
 .moment.missed .kind {{ color:#A6392E; }}
 .moment.plan .kind {{ color:#3A6EA5; }}
 .moment.drift .kind {{ color:#B8860B; }}
 .moment h3 {{ font-size:23px; margin:3px 0; display:flex;
   align-items:center; gap:10px; flex-wrap:wrap; }}
 .num {{ font-family:Menlo,monospace; color:var(--muted); }}
 .badge {{ font-size:11px; letter-spacing:0.05em; text-transform:uppercase;
   color:#fff; background:var(--c); padding:2px 8px; }}
 .by {{ color:var(--muted); font-size:13px; margin:0 0 14px; }}
 .body {{ display:grid; grid-template-columns:300px 1fr; gap:26px;
   align-items:start; }}
 .diagram svg {{ width:100%; height:auto; border:1px solid var(--rule); }}
 .prose p {{ margin:0 0 9px; }}
 .cost b, .gift b {{ font-family:Menlo,monospace; }}
 .line .k {{ font-size:11px; letter-spacing:0.07em; text-transform:uppercase;
   color:var(--muted); margin-right:8px; }}
 .pv {{ font-family:Menlo,monospace; font-size:13px; color:var(--pv); }}
 .facts {{ display:flex; flex-wrap:wrap; gap:6px; margin:12px 0; }}
 .fact {{ font-size:12px; border:1px solid var(--rule); padding:2px 8px;
   background:var(--card); }}
 .align {{ border-left:3px solid var(--muted); padding:8px 12px;
   background:var(--card); font-size:14.5px; }}
 .align.right {{ border-color:#B8860B; }}
 .align.on {{ border-color:#2E7D5B; }}
 .endgame-read {{ margin-top:14px; border-left:3px solid #3A6EA5;
   padding:8px 12px; background:var(--card); }}
 .endgame-read h4 {{ font-size:11px; letter-spacing:0.08em;
   text-transform:uppercase; color:var(--muted); margin-bottom:4px; }}
 .endgame-read ul {{ margin:0; padding-left:16px; font-size:14.5px; }}
 .endgame-read li {{ margin:2px 0; }}
 .read-block {{ margin-top:14px; font-size:15px; }}
 .read-block p {{ margin:8px 0 4px; }}
 ul.read {{ margin:4px 0 10px; padding-left:18px; }}
 ul.read li {{ margin:3px 0; }}
 .menus {{ display:grid; grid-template-columns:1fr 1fr; gap:18px;
   margin-top:16px; border-top:1px solid var(--rule); padding-top:14px; }}
 .menu h4 {{ font-size:12px; letter-spacing:0.08em; text-transform:uppercase;
   color:var(--muted); margin-bottom:6px; }}
 .menu ul {{ list-style:none; margin:0; padding:0; }}
 .menu li {{ margin:0 0 9px; font-size:13.5px; }}
 .tier {{ display:block; font-size:10px; letter-spacing:0.06em;
   text-transform:uppercase; color:var(--t); }}

 ul.bank {{ list-style:none; padding:0; margin:0; display:grid;
   grid-template-columns:repeat(auto-fit,minmax(400px,1fr)); gap:22px; }}
 ul.bank li {{ display:grid; grid-template-columns:190px 1fr; gap:16px;
   border:1px solid var(--rule); padding:14px; background:var(--card); }}
 .mini svg {{ width:100%; height:auto; }}
 .who {{ font-size:12px; color:var(--muted); margin:0 0 6px; }}
 .had {{ margin:0 0 6px; }}
 .worth {{ font-family:Menlo,monospace; font-size:12.5px; color:#A6392E;
   margin:0; }}

 table.moves {{ width:100%; border-collapse:collapse; font-size:14px; }}
 table.moves th {{ text-align:left; color:var(--muted); font-weight:500;
   font-size:11px; text-transform:uppercase; letter-spacing:0.07em;
   border-bottom:1px solid var(--rule); padding:5px 8px; }}
 table.moves td {{ padding:4px 8px; border-bottom:1px solid var(--hair); }}
 table.moves tbody th {{ font-family:Menlo,monospace; color:var(--muted);
   border-bottom:1px solid var(--hair); width:44px; }}
 .mv {{ font-family:Menlo,monospace; }}
 .cp {{ float:right; font-family:Menlo,monospace; font-size:12px;
   color:var(--muted); }}
 .note {{ color:var(--muted); font-size:13px; font-style:italic; }}
 footer {{ border-top:1px solid var(--rule); margin-top:56px; padding-top:16px;
   color:var(--muted); font-size:12.5px; }}
 @media (max-width:760px) {{
   .body {{ grid-template-columns:1fr; }}
   .menus {{ grid-template-columns:1fr; }}
   ul.bank li {{ grid-template-columns:1fr; }}
 }}
</style>
<div class="wrap">
<header class="mast">
  <span class="mark">Lucena</span>
  <span class="meta">GAME ANALYSIS · {_e(h.get('ECO',''))} · {_e(h.get('Date',''))}</span>
</header>

<div class="players">
  <h1>{_e(white)} <span style="color:var(--muted)">vs</span> {_e(black)}</h1>
  <span class="res">{_e(result)}</span>
</div>
<p class="sub">{_e(we)} vs {_e(be)} · {_e(h.get('TimeControl',''))}
   {("· " + _e(term)) if term else ""}</p>
{opening_line}

<section id="shape">
  <h2>The shape of the game</h2>
  <p class="lede">One line: how the engine judged the position after every move.
     Above the centre favours White, below favours Black. Marked points are the
     moments below.</p>
  {_curve(story)}
  <div class="curve-key"><span>{_e(white)} better ↑</span>
    <span>{_e(black)} better ↓</span></div>
  {_acts(story)}
  {_ending_line(story)}
</section>

<section id="scorecard">
  <h2>Scorecard</h2>
  <p class="lede">Accuracy, where the errors fell, and the full spread of move
     quality. Counts over one game are a tally, not a diagnosis — a pattern
     worth trusting needs many games.</p>
  {_patterns(story)}
</section>

<section id="walkthrough">
  <h2>The moments</h2>
  <p class="lede">Where the game was actually decided — and, in the quiet
     positions between, the plan the structure was asking for. A plan is
     printed only if the engine confirmed it in one of its own equal-value
     lines <em>and</em> it says something specific about this position;
     everything else is held back rather than padding the page.</p>
  {note}
  {chapters}
</section>

{_tracks(story)}

{_bank(story, h)}

<section id="moves">
  <h2>Every move</h2>
  <p class="lede">The full game, each move classified. The number on the right
     is the engine's evaluation after the move, in pawns, from White's side.</p>
  {_movelist(story)}
</section>

<footer>
  Produced by Lucena's deterministic analysis pipeline — Stockfish for
  calculation, Maia for human typicality, and our own structural detectors for
  the plans. No language model was involved in any sentence on this page.
  <br>Schema {_e(story.get('schema',''))} ·
  {len(story['plies'])} plies · {story.get('moments_found',0)} moments found ·
  {story.get('plan_chapters_shown',0)} of
  {story.get('plan_chapters_read', story.get('plan_chapters_considered',0))}
  quiet positions read carried an engine-confirmed plan.{hidden_note}
</footer>
</div>"""
