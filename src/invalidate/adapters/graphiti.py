"""Graphiti adapter: govern the entity edges (facts) of one graphiti-core group.

    from graphiti_core import Graphiti
    from invalidate.adapters import Governor
    from invalidate.adapters.graphiti import GraphitiAdapter

    graphiti = Graphiti(uri, user, password)
    gov = Governor(GraphitiAdapter(graphiti, group_id="alice"), "ledger.db", mode="flag")
    gov.sync(); gov.observe("we migrated to SQLite", source="slack")

graphiti-core is async; this adapter is sync (the `Adapter` protocol is) and runs each coroutine with
`_run`: `asyncio.run` when no loop is running in this thread, otherwise a fresh loop on a helper thread.
Nothing from graphiti_core is imported at module import time; the edge/episode classes are imported inside
methods (or injected for tests). Mirrored (graphiti-core 0.30.2):

  edges.py:263   class EntityEdge(Edge): uuid, group_id, source_node_uuid, target_node_uuid, created_at, name,
                 fact, fact_embedding, episodes, expired_at, valid_at, invalid_at, reference_time,
                 attributes: dict[str, Any]   (edges.py:271-285)
  edges.py:335   await edge.save(driver)      writes the fields above; free `attributes` become edge properties
                 and are read back by get_entity_edge_from_record (edges.py:968), minus the reserved keys
  edges.py:481   await EntityEdge.get_by_group_ids(driver, group_ids, limit=None, uuid_cursor=None,
                 with_embeddings=False)   MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) (edges.py:508)
  edges.py:378   await EntityEdge.get_by_uuid(driver, uuid)
  nodes.py       await EpisodicNode.get_by_group_ids(driver, group_ids, limit=None, uuid_cursor=None);
                 EpisodicNode(uuid, name, content, source_description, valid_at, ...)
  graphiti.py    await graphiti.add_episode(name, episode_body, source_description, reference_time,
                 source=EpisodeType.message, group_id=None, ...) -> AddEpisodeResults(episode: EpisodicNode,
                 episodic_edges, nodes, edges: list[EntityEdge], ...)   (graphiti.py:114)
  graphiti.py:208  Graphiti.driver
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Iterable
from datetime import datetime, timezone
from typing import Any, TypeVar

from ..types import Status
from .base import DEAD, HostMemory, Reason

T = TypeVar("T")
EPISODE_PREFIX = "invalidate:"


def _run(coro: Awaitable[T]) -> T:
    """Run a coroutine to completion from sync code, whether or not an event loop is already running here."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = asyncio.run(coro)  # type: ignore[arg-type]
        except BaseException as e:  # noqa: BLE001 - re-raised on the caller's thread
            box["error"] = e

    t = threading.Thread(target=target, name="invalidate-graphiti", daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GraphitiAdapter:
    """Entity edges of one `group_id`, one edge (`.fact`) = one memory. The episodes invalidate inserted are
    pulled too (their `content` is the verbatim event) so successors survive a re-sync.

    flag   -> the receipt (`reason.as_metadata()`) is merged into `edge.attributes` and the edge saved.
              For a dead verdict (contradicted / superseded / deleted) `invalid_at` and `expired_at` are also set
              to now: Graphiti's own "no longer true" representation, so its search stops returning the fact.
              A restore (status active) clears both again. needs_review only writes the attributes.
    delete -> identical to a dead flag. Graphiti keeps history; the adapter never hard-deletes an edge.
    insert -> `graphiti.add_episode(name="invalidate:<event id>", episode_body=text, source_description=source,
              reference_time=now, group_id=...)`; returns the episode uuid. The episode content is verbatim, but
              the edges Graphiti extracts from it are its own wording and appear as new memories on the next sync.
    """

    name = "graphiti"

    def __init__(
        self,
        graphiti: Any,
        group_id: str,
        driver: Any = None,
        *,
        edge_class: Any = None,
        episode_class: Any = None,
        include_episodes: bool = True,
    ) -> None:
        self.graphiti = graphiti
        self.group_id = group_id
        self.driver = driver if driver is not None else getattr(graphiti, "driver", None)
        if self.driver is None:
            raise ValueError("GraphitiAdapter needs a driver: pass driver= or a Graphiti with .driver")
        self._edge_class = edge_class
        self._episode_class = episode_class
        self.include_episodes = include_episodes
        self._edges: dict[str, Any] = {}
        self._episode_ids: set[str] = set()

    # -- lazy SDK classes -----------------------------------------------------------
    @property
    def EntityEdge(self) -> Any:  # noqa: N802 - mirrors the SDK name
        if self._edge_class is None:
            from graphiti_core.edges import EntityEdge  # noqa: PLC0415

            self._edge_class = EntityEdge
        return self._edge_class

    @property
    def EpisodicNode(self) -> Any:  # noqa: N802
        if self._episode_class is None:
            from graphiti_core.nodes import EpisodicNode  # noqa: PLC0415

            self._episode_class = EpisodicNode
        return self._episode_class

    # -- Adapter --------------------------------------------------------------------
    def pull(self) -> Iterable[HostMemory]:
        out: list[HostMemory] = []
        self._edges = {}
        for edge in _run(self.EntityEdge.get_by_group_ids(self.driver, [self.group_id])) or []:
            if getattr(edge, "expired_at", None) is not None:
                continue
            self._edges[edge.uuid] = edge
            out.append(HostMemory(id=str(edge.uuid), text=edge.fact, source="graphiti",
                                  metadata={"edge_name": getattr(edge, "name", "")}))
        if self.include_episodes:
            self._episode_ids = set()
            for ep in _run(self.EpisodicNode.get_by_group_ids(self.driver, [self.group_id])) or []:
                if not str(getattr(ep, "name", "")).startswith(EPISODE_PREFIX):
                    continue
                self._episode_ids.add(str(ep.uuid))
                out.append(HostMemory(id=str(ep.uuid), text=ep.content, kind="event",
                                      source=getattr(ep, "source_description", "graphiti") or "graphiti"))
        return out

    def _edge(self, host_id: str) -> Any:
        edge = self._edges.get(host_id)
        if edge is None:
            edge = _run(self.EntityEdge.get_by_uuid(self.driver, host_id))
            self._edges[host_id] = edge
        return edge

    def _apply(self, host_id: str, reason: Reason, *, force_dead: bool) -> None:
        if host_id in self._episode_ids:
            return  # episodes have no validity window: verdicts on them live in the ledger only
        edge = self._edge(host_id)
        edge.attributes = {**(getattr(edge, "attributes", None) or {}), **reason.as_metadata()}
        if force_dead or reason.status in DEAD or reason.status is Status.DELETED:
            now = _utcnow()
            edge.invalid_at = edge.invalid_at or now
            edge.expired_at = now
        elif reason.status is Status.ACTIVE:
            edge.invalid_at = None
            edge.expired_at = None
        _run(edge.save(self.driver))

    def flag(self, host_id: str, reason: Reason) -> None:
        self._apply(host_id, reason, force_dead=False)

    def delete(self, host_id: str, reason: Reason) -> None:
        self._apply(host_id, reason, force_dead=True)

    def insert(self, text: str, source: str, metadata: dict[str, Any]) -> str | None:
        name = f"{EPISODE_PREFIX}{metadata.get('invalidate_event_id') or int(_utcnow().timestamp() * 1000)}"
        res = _run(self.graphiti.add_episode(
            name=name, episode_body=text, source_description=source, reference_time=_utcnow(), group_id=self.group_id,
        ))
        episode = getattr(res, "episode", None)
        uuid = getattr(episode, "uuid", None)
        if not uuid:
            return None
        self._episode_ids.add(str(uuid))
        return str(uuid)


__all__ = ["GraphitiAdapter", "EPISODE_PREFIX"]
