"""Client <-> backend server (B4/B5, pivoted): WebSocket for the live loop + REST for the rest.

The backend uses the in-process ToolContext (the full legacy coach: analyze/evaluate/hints, drills
via DrillState + build_line_tree, poisoned lines, Maia move-meaning) — NOT a thin gRPC path — so
drills and Maia coaching behave exactly as before. (The open/closed engine firewall is set aside for
now; ToolContext calls lucena_engine in-process. Re-splitting it over gRPC is a follow-up.)

WS `/ws`: snapshot on connect then the StateStore's deltas; client sends turn/explain/position/view/
input/move/drill. `/move` adjudicates against the live drill (opponent reply, feedback) via
ctx.play_move; `/drill` arms a forcing-line drill on a position.
"""

from __future__ import annotations

import asyncio
import os
import re

from fastapi import FastAPI, Header, WebSocket, WebSocketDisconnect
from starlette.responses import JSONResponse

from lucena_engine.uci import Engine
from . import auth
from .engine_io.enginepool import EnginePool, SingleEnginePool, default_size
from .persistence.state import StateStore
from .persistence.db import DB
from .grounding_tools.tools import ToolContext

# A FEN-like token (7 rank separators then a side-to-move) anywhere in a typed turn → a POSITION
# set-up rather than chat. Same shape as the orchestrator's own detector.
_FEN_RE = re.compile(r"(?:[pnbrqkPNBRQK1-8]+/){7}[pnbrqkPNBRQK1-8]+\s+[wb]\b")


def _walk_target(store, view: dict | None) -> dict | None:
    """The variation move a fresh /view has walked ONTO and the coach should now comment on, or None.

    Walking a variation is the shared analysis board stepping through a sideline — the app reports it as
    a `view` (fen + cursor + resolved line), never a `move`. This picks out the comment-worthy ones and
    is the single gate on when a walk speaks. It fires ONLY when:
      - the cursor sits on a `variation` row (not the mainline prefix that flows into the sideline), and
      - no Lesson owns the chat (a walk is browsing, never a drill answer — silent during a drill), and
      - `mark_walked` claims this move for the first time (once per move landed on — stepping BACKWARD
        onto or re-visiting a move already read stays silent).
    Reads `active_lesson`/`mark_walked` on `store`, so the caller must run it with the chat bound.
    Returns `{fen, san, uci}` for the move, or None to stay silent."""
    if not view or not view.get("in_variation"):
        return None
    line = view.get("line") or []
    cur = view.get("cursor")
    if not isinstance(cur, int) or not (0 <= cur < len(line)):
        return None
    node = line[cur] or {}
    if node.get("kind") != "variation":            # cursor on the mainline prefix → not a walk
        return None
    san, uci = node.get("san"), node.get("uci")
    fen = node.get("fen") or view.get("fen")
    if not (san or uci) or not fen:
        return None
    if store.active_lesson() is not None:          # a lesson owns the chat → walks stay silent
        return None
    if not store.mark_walked(fen):                 # already read this move → silent re-visit
        return None
    return {"fen": fen, "san": san, "uci": uci}

_DEFAULT_MODEL = os.environ.get("LUCENA_MODEL", "gemini-flash-lite-latest")
# Engine pool sizing. Every knob is explicit and env-overridable because the cost is real and per
# instance: `size` concurrent analyses, each a Stockfish process holding `hash_mb`. The floor is
# size × hash_mb of RAM — the release checklist says to size this against real load, not a guess.
_POOL_SIZE = int(os.environ.get("LUCENA_POOL_SIZE") or default_size())
_POOL_THREADS = int(os.environ.get("LUCENA_POOL_THREADS", "1"))
_POOL_HASH_MB = int(os.environ.get("LUCENA_POOL_HASH_MB", "64"))
# Accounts off => every request is one implicit anonymous user (the single-user desktop mode this
# started as, and what the existing suite exercises). On => every route except /health and /auth/*
# needs a bearer token. Off by default so single-user local runs keep working unchanged.
_AUTH_REQUIRED = os.environ.get("LUCENA_REQUIRE_AUTH", "").lower() in ("1", "true", "yes")


def _unauthorized():
    return JSONResponse({"error": "unauthenticated"}, status_code=401)


def _forbidden():
    """Auth is off, so a PermissionError here means an ownership violation, not a missing token."""
    return JSONResponse({"error": "forbidden"}, status_code=403)


def _open_chat_for_bound(store, sub, sid: str) -> None:
    """`open_chat_for` off the loop (it does DB writes). Runs via asyncio.to_thread, which copies the
    caller's context, so the bind it performs lands where the caller expects."""
    store.open_chat_for(sub, sid)


def _arm_drill(ctx, store, fen):
    """Arm a drill on a position — a deliberate APP action. It supersedes any pending Socratic probe,
    so clear the turn gate first (else the `_scoped` guard refuses with `awaiting_input`)."""
    store.set_gate(False)
    return ctx.build_and_arm_drill(fen)


def _make_maia():
    """A MaiaEngine if LUCENA_MAIA is configured, else None (poisoned lines / move-meaning off)."""
    if not os.environ.get("LUCENA_MAIA"):
        return None
    try:
        from lucena_engine.maia import MaiaEngine
        return MaiaEngine()
    except Exception:  # noqa: BLE001
        return None


def build_app(*, home: str, llm=None, model: str = _DEFAULT_MODEL,
              rating: int = 1500, engine=None) -> FastAPI:
    app = FastAPI(title="lucena-backend")
    db = DB(os.path.join(home, "lucena"))
    store = StateStore(home, db=db)
    # A bounded pool, shared by both contexts: engines are fungible (new_game() on acquire), so a
    # chat borrows one for a call rather than owning one. `engine=` still overrides for tests.
    # threads/hash are explicit — Engine's own defaults are per-INSTANCE (threads = cpu-2, hash =
    # 256MB), so a pool at those defaults would oversubscribe the CPU ~size× and reserve size×256MB.
    pool = (SingleEnginePool(engine) if engine is not None
            else EnginePool(size=_POOL_SIZE, threads=_POOL_THREADS, hash_mb=_POOL_HASH_MB))
    maia = _make_maia()
    ctx = ToolContext(store=store, pool=pool, maia=maia, player_rating=rating)
    ctx._poisoned_line_engine_factory = lambda: Engine(threads=1)   # dedicated single-thread detector
    # A SEPARATE ToolContext for the coach's read-only LLM grounding (evaluate/analyze). It shares the
    # store AND the pool, but has its own per-chat tool locks, so the coach's multi-second analysis
    # never blocks an interactive move/drill on the main ctx (which was making a Retry right after a
    # wrong move sit locked for seconds while coach_move ran). Do NOT collapse the two: one ctx would
    # serialize a chat's own coaching against its own move again.
    # Maia off here: the main ctx owns the single Maia subprocess. (It would BLOCK, not corrupt —
    # MaiaEngine.top_human_moves holds its own lock across the whole conversation — but one predictor
    # per process is deliberate: each instance is a ~485MB torch model.)
    ground_ctx = ToolContext(store=store, pool=pool, maia=None, player_rating=rating)


    # The conversation spine (LLD): one ConversationLoop routes turn+move by mode. It fully REPLACES
    # the retired Orchestrator/QuickCoach; `_dispatch` routes every turn/move through it. (Prompt
    # WORDING is still a co-design work-in-progress, but the architecture is the live one.)
    from .coaching.loop import ConversationLoop, Input as _Input
    from .coaching.freeform import FreeformHandler
    from .coaching.coach import CoachHandler
    from .llm import make_adapter as _make_adapter
    _spine_llm = llm or _make_adapter({"provider": "gemini", "default_model": model})
    from . import margin as _margin_mod
    _margin_mod.configure(pool=pool, maia=maia)   # the margin deep layer's engine access
    loop = ConversationLoop(
        store=store,
        freeform=FreeformHandler(ctx=ctx, store=store, llm=_spine_llm, model=model, ground=ground_ctx),
        coach=CoachHandler(ctx=ctx, store=store, llm=_spine_llm, model=model, ground=ground_ctx),
    )

    app.state.store = store
    app.state.ctx = ctx

    def _set_view_and_walk(msg: dict):
        """The /view fast path AND its slow follow-up decision, together in the worker thread (where the
        bound-chat context is live): report the board (fast, ordered, ALWAYS) and, if this /view walked
        onto a fresh sideline move, return the `walk` Input to run the comment in the background."""
        walk = _walk_target(store, store.set_view(msg))
        return _Input(kind="walk", fen=walk["fen"], san=walk["san"], uci=walk["uci"]) if walk else None

    async def _dispatch(msg: dict, sid: str):
        t = msg.get("type")
        if t == "turn":
            # A typed turn: a FEN-shaped message is a POSITION set-up (Input.position), everything else
            # is chat (Input.text). The FEN-detection lives here so the loop sees a resolved kind.
            text = msg.get("text") or ""
            # Echo the player's own words as a persisted "you" beat BEFORE the loop runs — the app has
            # already rendered it optimistically under `client_id`; persisting+broadcasting it here makes
            # it durable (survives reconnect's snapshot) and reconcilable (the app matches the nonce so
            # the message isn't shown twice). Only when there's text to echo.
            if text:
                with store.bound(sid):
                    await asyncio.to_thread(store.append_beats, [
                        {"kind": "you", "stops": False, "client_id": msg.get("client_id"),
                         "segments": [{"text": text}]}])
            inp = (_Input(kind="position", text=text) if _FEN_RE.search(text)
                   else _Input(kind="text", text=text))
            await loop.handle_input(sid, inp)
        elif t == "move":
            await loop.handle_input(sid, _Input(kind="move", uci=msg.get("uci"), fen=msg.get("fen"),
                                                client_id=msg.get("client_id")))
        elif t == "continue":                       # the player clicked Continue → walk the next branch
            await loop.handle_input(sid, _Input(kind="continue"))
        elif t == "position":                       # navigation: report the board being shown (NOT a turn)
            await asyncio.to_thread(store.set_board_view, msg.get("fen"))
        elif t == "view":
            # Report the board inline (fast, ordered), and if this /view landed on a fresh variation
            # move, hand back a `walk` Input so the caller runs the (slow, LLM) comment in the background.
            return await asyncio.to_thread(_set_view_and_walk, msg)
        elif t == "input":
            await asyncio.to_thread(store.set_input, msg.get("data") or msg)
        elif t == "explain":
            # On-demand "Why?" for a HELD wrong drill move (v1: the verdict is no longer auto-shown).
            # The wrong move is still on the board, so fen+uci are enough to regenerate today's text.
            await loop.handle_input(sid, _Input(kind="explain", uci=msg.get("uci"), fen=msg.get("fen")))
        # RETIRED: `drill` (old arm path) → coach-mode lesson entry. Intentionally not routed.

    async def _handle(msg: dict, sid: str):
        # The transport has no catch-all: an unhandled error here (an LLM outage, a bad move) would
        # break the WS receive loop and drop the connection mid-turn, leaving the app spinning on a
        # status that never clears. Contain it — clear the working status and surface a plain beat so
        # the player always gets an answer, and the socket stays up. Returns the dispatch result (a
        # `walk` Input to run as a slow follow-up, or None).
        try:
            return await _dispatch(msg, sid)
        except Exception as exc:  # noqa: BLE001
            with store.bound(sid):          # the apology belongs to THIS chat, not whatever is current
                store.publish_status(None)
                store.append_beats([{
                    "kind": "say", "tone": "teach", "stops": False,
                    "segments": [{"text": "I hit a snag reaching my coaching brain just now — the position "
                                          "is still set, so try that again in a moment."}],
                }])
            print(f"[_handle] {msg.get('type')} failed: {exc!r}", flush=True)
            return None

    async def _handle_walk(inp, sid: str) -> None:
        # A variation-walk comment is AMBIENT — the player is browsing a line, not asking a question — so
        # a failure clears the working status QUIETLY, with no apology beat (unlike a real turn/move).
        try:
            await loop.handle_input(sid, inp)
        except Exception as exc:  # noqa: BLE001
            with store.bound(sid):
                store.publish_status(None)
            print(f"[_handle_walk] failed: {exc!r}", flush=True)

    # Loop work (turn AND move) can take several seconds of LLM+engine work — both now flow through the
    # unified loop, which may narrate. Running INLINE in the receive loop would block every other
    # message (clicks, navigation) and look frozen, so both run as background tasks. The fast,
    # state-mutating messages (position/view/input) stay inline and ordered. The store's per-chat tool
    # lock still serializes shared-state mutation across them. (The move's fast board-apply happens
    # early in the handler coroutine, so the board still lands promptly before the slow narration.)
    _SLOW = {"turn", "move"}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        # AuthMiddleware has already authenticated this handshake (and closed it 1008 before accept if
        # the token was bad), and bound the user for this connection's whole context.
        await websocket.accept()
        # This connection's chat. `?session=<id>` picks one explicitly; otherwise fall back to this
        # user's active-chat pointer (minting one if they have never had a chat).
        requested = websocket.query_params.get("session")
        try:
            sid = await asyncio.to_thread(store.ensure_session_id, requested)
        except PermissionError:
            await websocket.close(code=1008)         # asked for someone else's chat
            return
        sub = store.subscribe(sid)
        bg: set[asyncio.Task] = set()
        try:
            with store.bound(sid):
                snap = store.snapshot()
            for channel, payload in snap:
                await websocket.send_json({"type": channel, **payload})
            await websocket.send_json({"type": "ready"})

            async def pump():
                while True:
                    # next_event drops events queued for a chat this socket has since left. The stamp
                    # is then RE-CHECKED under the gate: send_json is an await point, so an open_chat
                    # can retarget this socket between the dequeue and the send, and that dequeued
                    # event would otherwise still go out to the new chat.
                    ev_sid, ev_epoch, channel, payload = await sub.next_event()
                    async with sub.gate:
                        if not sub.is_live(ev_sid, ev_epoch):
                            continue        # retargeted while we were between dequeue and gate
                        await websocket.send_json({"type": channel, **payload})

            sender = asyncio.create_task(pump())
            try:
                while True:
                    msg = await websocket.receive_json()
                    if msg.get("type") == "open_chat":
                        new_sid = msg.get("session_id")
                        if new_sid and new_sid != sid:
                            # ONE store call, not a sequence: authorize + activate + retarget + this
                            # socket's baseline are a single ordered operation (see open_chat_for).
                            # Doing it here as steps is what created both bugs this line has had —
                            # publishing before the retarget sent the snapshot to nobody; retargeting
                            # before the publish let a concurrent beat on the destination chat land a
                            # delta ahead of the baseline. The store owns the order because only the
                            # store holds the locks that make it atomic.
                            # `gate` still wraps it: it waits out any in-flight send, so no event
                            # dequeued for the OLD chat can land after the switch.
                            async with sub.gate:
                                try:
                                    await asyncio.to_thread(_open_chat_for_bound, store, sub, new_sid)
                                except PermissionError:
                                    continue               # not this user's chat — ignore the request
                                sid = new_sid              # only after the store accepted the move
                        continue
                    # Bind BEFORE spawning: create_task copies the current context, so the task
                    # inherits this chat and cannot re-resolve a different one when it finally writes.
                    if msg.get("type") in _SLOW:
                        with store.bound(sid):
                            task = asyncio.create_task(_handle(msg, sid))
                        bg.add(task)
                        task.add_done_callback(bg.discard)
                    else:
                        # Fast, ordered, inline. A /view may hand back a `walk` Input — the slow (LLM)
                        # variation-walk comment — which then runs as a tracked background task like a
                        # move, so rapid stepping never blocks the receive loop.
                        with store.bound(sid):
                            walk = await _handle(msg, sid)
                        if walk is not None:
                            with store.bound(sid):
                                wtask = asyncio.create_task(_handle_walk(walk, sid))
                            bg.add(wtask)
                            wtask.add_done_callback(bg.discard)
            finally:
                sender.cancel()
                for t in bg:
                    t.cancel()
        except WebSocketDisconnect:
            pass
        finally:
            store.unsubscribe(sub)

    # -- REST ---------------------------------------------------------------
    @app.get("/health")
    async def health():
        return {"ok": True}

    # -- accounts ----------------------------------------------------------
    @app.post("/auth/register")
    async def register(body: dict):
        try:
            uid = await auth.register(db, body.get("email") or "", body.get("password") or "")
        except auth.RegisterError as exc:
            # `error` is the stable CODE the client maps to its own copy; `detail` is prose for a
            # human reading a log or a curl. The client never renders `detail` — see AuthClient.
            return JSONResponse({"error": exc.code, "detail": str(exc)}, status_code=400)
        except ValueError as exc:
            return JSONResponse({"error": "invalid_request", "detail": str(exc)}, status_code=400)
        return {"user_id": uid}

    @app.post("/auth/login")
    async def login_route(body: dict):
        tok = await auth.login(db, body.get("email") or "", body.get("password") or "")
        if tok is None:
            return JSONResponse({"error": "bad_credentials"}, status_code=401)
        return {"token": tok}

    @app.post("/auth/logout")
    async def logout_route(authorization: str | None = Header(default=None)):
        if authorization and authorization.lower().startswith("bearer "):
            await asyncio.to_thread(db.delete_token, auth.token_hash(authorization.split()[1]))
        return {"ok": True}

    # Auth is handled ONCE, in auth.AuthMiddleware (applied below), which rejects before any route
    # runs and binds the user for the whole request. Routes therefore never check tokens: they just
    # read the ambient user via the store, exactly as the coach paths do.

    # REST has no connection to carry a chat, so these take one explicitly (`session_id` in the body /
    # `?session=`), falling back to this user's active-chat pointer. They must never run against an
    # unbound cursor.
    async def _rest_sid(requested: str | None) -> str:
        return await asyncio.to_thread(store.ensure_session_id, requested)

    @app.get("/sessions")
    async def sessions(session: str | None = None):
        """This user's chats only — un-scoped, the rail would show everyone everyone else's."""
        from .persistence.sessions import list_sessions
        try:
            current = await _rest_sid(session)      # ?session= can name someone else's chat
        except PermissionError:
            return _forbidden()
        rows = await asyncio.to_thread(list_sessions, store.home, store.db,
                                       user_id=store.current_user)
        return {"sessions": rows, "current": current}

    @app.get("/session")
    async def get_session():
        try:
            return {"session_id": await _rest_sid(None)}
        except PermissionError:
            return _forbidden()

    @app.post("/session/new")
    async def new_session():
        """'New chat' — the BACKEND mints the id, makes it this user's active chat (publishing reset +
        a clean snapshot to that chat's sockets), and returns it for the app to adopt."""
        return {"session_id": await asyncio.to_thread(store.new_session)}

    @app.post("/session")
    async def post_session(body: dict):
        sid = body.get("session_id")
        if not sid:
            return JSONResponse({"error": "bad_session"}, status_code=400)
        try:
            return {"session_id": await asyncio.to_thread(store.write_session_id, sid)}
        except PermissionError:
            return _forbidden()

    @app.post("/move")
    async def move(body: dict):
        """Play a move through the NEW spine (unified entry): coach mode adjudicates it against the
        live Lesson (opponent reply + beats stream over the WS), freeform explains it. Returns
        {ok, drill, correct?, finished?} — the mac app drives Retry / the poisoned-line button off
        this. The result rides a ContextVar set by the coach handler during the awaited call."""
        from .coaching.coach import move_result
        uci, fen = body.get("uci"), body.get("fen")
        if not uci:
            return JSONResponse({"error": "bad_move"}, status_code=400)
        try:
            sid = await _rest_sid(body.get("session_id"))
        except PermissionError:
            return _forbidden()
        move_result.set(None)
        await loop.handle_input(sid, _Input(kind="move", uci=uci, fen=fen,
                                            client_id=body.get("client_id")))
        return {"ok": True, **(move_result.get() or {"drill": False})}

    @app.post("/drill")
    async def drill(body: dict):
        """Arm a forcing-line drill on a position (build the tree + detect the poisoned line)."""
        fen = body.get("fen")
        if not fen:
            return JSONResponse({"error": "bad_fen"}, status_code=400)
        try:
            # Same as /move: _arm_drill writes tree/board/drill state and publishes, so it must run
            # against a resolved chat — never the unbound "" bucket.
            sid = await _rest_sid(body.get("session_id"))
            with store.bound(sid):
                return await asyncio.to_thread(_arm_drill, ctx, store, fen)
        except PermissionError:
            return _forbidden()

    @app.post("/lesson")
    async def lesson(body: dict):
        """Leave the current drill (the puzzle screen's back button): mark this chat's live — or
        what-if-parked — lesson `open` (the resumable state the coach's spoken 'stop' produces, which
        flips the chat back to freeform), AND finish its activity — drop a card into the base
        conversation and switch the view home, so the saved puzzle can be reopened. Deterministic, no
        LLM. Idempotent: a solved drill (no live lesson) still finishes its activity."""
        from .coaching.lesson import OPEN
        if body.get("op") != "leave":
            return JSONResponse({"error": "bad_op"}, status_code=400)
        try:
            sid = await _rest_sid(body.get("session_id"))
        except PermissionError:
            return _forbidden()

        def _leave() -> bool:
            with store.bound(sid):
                live = store.active_lesson() or store.suspended_lesson()
                if live is not None:
                    store.set_lesson_state(live.spec.id, OPEN)
                finished = store.finish_activity()   # card + view home (no-op if already on the base)
                return live is not None or finished

        return {"ok": True, "left": await asyncio.to_thread(_leave)}

    @app.post("/activity")
    async def activity(body: dict):
        """Switch the in-view activity. `op:"open", idx:N` reopens a saved activity by index (a card
        click, or `idx:0` to return to the base conversation). The saved frame's board/beats/variations
        replay over the WS. Deterministic, no LLM."""
        if body.get("op") != "open":
            return JSONResponse({"error": "bad_op"}, status_code=400)
        idx = body.get("idx")
        if not isinstance(idx, int):
            return JSONResponse({"error": "bad_idx"}, status_code=400)
        try:
            sid = await _rest_sid(body.get("session_id"))
        except PermissionError:
            return _forbidden()

        def _open() -> bool:
            with store.bound(sid):
                if not store.open_activity(idx):
                    return False
                # Reopening a saved puzzle makes its drill LIVE again (reactivate the lesson, mode →
                # coach) so the player can keep solving. Done INLINE — not through the coach turn loop —
                # because it produces no beat and must NOT toggle the "Thinking…" status: that flagged a
                # silent turn and flashed the "Uh oh, I didn't catch that" net. The saved board/line/
                # variations are restored by open_activity's re-render; nothing is re-presented.
                lid = store.active_frame_lesson_id()
                if lid:
                    store.reactivate_lesson(lid)
                return True

        return {"ok": True, "opened": await asyncio.to_thread(_open)}

    @app.post("/margin")
    async def margin(body: dict):
        """The margin's content for one position (mac V1_LAYOUT.md) —
        deterministic and engine-free (lucena_backend.margin), safe to call
        on every navigator scrub. `session_id` seeds the move-1 epigraph so
        a session keeps its quote."""
        from . import margin as margin_mod
        fen = body.get("fen")
        if not fen:
            return JSONResponse({"error": "bad_fen"}, status_code=400)
        try:
            return margin_mod.build(fen, seed=str(body.get("session_id") or ""),
                                    live=bool(body.get("live")))
        except ValueError:
            return JSONResponse({"error": "bad_fen"}, status_code=400)

    @app.post("/analyze")
    async def analyze(body: dict):
        on, fen = body.get("on"), body.get("fen")
        # Live analysis toggle is a follow-up; accept the call so the app doesn't 404.
        return {"ok": True, "on": bool(on), "fen": fen}

    @app.get("/config")
    async def config():
        return {"model": model, "maia": maia is not None, "rating": rating}

    # One gate for the whole app, HTTP and WS alike. Everything not in AuthMiddleware.PUBLIC needs a
    # token, so a route added later is refused by default instead of silently exposed.
    app.add_middleware(auth.AuthMiddleware, db=db, store=store, required=_AUTH_REQUIRED)
    return app


def serve(*, home: str, host: str = "127.0.0.1", port: int = 8766, **kw):
    import uvicorn
    uvicorn.run(build_app(home=home, **kw), host=host, port=port, log_level="info")


if __name__ == "__main__":
    serve(home=os.environ.get("LUCENA_HOME", os.path.expanduser("~/.lucena")),
          port=int(os.environ.get("LUCENA_BACKEND_PORT", "8766")))
