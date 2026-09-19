# Cognee

Governs the **data items** of one Cognee dataset: the rows `cognee.add(...)` creates, which are what
`cognify()` chunks, embeds and turns into a graph. One row (the verbatim file behind it) = one memory;
the id is `Data.id` (a UUID), which is stable across cognify and is the id `cognee.delete` takes.

Cognee has no per-item "no longer true" representation and no public per-item metadata update, so the
verdicts live in invalidate's ledger and reach Cognee in two ways: hard delete of the row (and
everything cognify derived from it), or filtering of search results by `document_id`. Flagging inside the
host is opt-in and uses an internal API (see below).

## Install

```sh
pip install invalidate cognee   # read against cognee 1.6.0 (github.com/topoteretes/cognee main @ 663a2dc, 2026-09-19)
```

`pip install cognee` succeeded in the project venv in under a minute on macOS/Python 3.14 (it pulls
lancedb, ladybug, litellm, fastembed, sqlalchemy/alembic, fastapi). The adapter never imports it at module
import time; the tests do not import it at all.

## Usage

```python
import cognee
from invalidate.adapters import Governor
from invalidate.adapters.cognee import CogneeAdapter, governed_search

# adapter is sync; it runs cognee's coroutines for you (asyncio.run, or a helper thread inside a loop)
adapter = CogneeAdapter(cognee, dataset_name="alice")          # or dataset_id=UUID(...); user=... optional
gov = Governor(adapter, "ledger.db", mode="ledger")           # the supported path (see "What flag can do")

gov.sync()                                                    # every text Data row -> ledger, verbatim
report = gov.observe("Alice migrated to SQLite", source="slack")
print(report.summary())                                       # verdicts in the ledger; Cognee untouched

hits = governed_search(cognee, gov, "which database?", query_type=cognee.SearchType.CHUNKS)
# -> cognee.search(...) scoped to the dataset, minus chunks whose Data row is dead or under review
hits = governed_search(cognee, gov, "which database?", query_type=cognee.SearchType.CHUNKS, annotate=True)
# -> keep all; each chunk payload dict gains "invalidate_note" (filter_results(raw, gov, annotate=True) likewise)

# inside async code, search yourself and filter:
#   raw = await cognee.search("which database?", query_type=cognee.SearchType.CHUNKS, dataset_ids=[adapter.dataset_id])
#   hits = filter_results(raw, gov)

# stronger: remove dead rows from Cognee itself (and their chunks/entities/summaries)
gov = Governor(CogneeAdapter(cognee, dataset_name="alice"), "ledger.db", mode="delete", successors=True)
```

## What each push does in Cognee

| action | call | notes |
|---|---|---|
| pull | `cognee.datasets.list_data(dataset_id, user=None)` ([datasets.py:138](https://github.com/topoteretes/cognee/blob/main/cognee/api/v1/datasets/datasets.py)) | rows with `mime_type` `text/*` or a text extension (txt, md, csv, json, yaml, rst, html), unless `include_non_text=True`. id = `str(Data.id)`, text = the file at `Data.raw_data_location` (raw text is stored verbatim as `text_<md5>.txt`, [save_data_to_file.py:78](https://github.com/topoteretes/cognee/blob/main/cognee/modules/ingestion/save_data_to_file.py)); `file://` paths are read directly, other schemes through `open_data_file`. Metadata: `name`, `mime_type`, `label` |
| flag | nothing, by default (`adapter.flagged_in_host is False`) | with `flag_in_host=True`: `Data.external_metadata = {**existing, **reason.as_metadata()}` through the internal relational session (the same merge-and-commit as [publish_updated_data.py:96-123](https://github.com/topoteretes/cognee/blob/main/cognee/modules/data/methods/publish_updated_data.py)); or pass your own `metadata_writer=(data_id, receipt) -> None \| Awaitable` |
| delete | `cognee.datasets.delete_data(dataset_id=..., data_id=..., user=None)` ([datasets.py:217](https://github.com/topoteretes/cognee/blob/main/cognee/api/v1/datasets/datasets.py)), `mode="soft"` | falls back to the deprecated `cognee.delete(data_id=..., dataset_id=...)` ([delete/\_\_init\_\_.py:9](https://github.com/topoteretes/cognee/blob/main/cognee/api/v1/delete/__init__.py), a wrapper of the same). Removes the row, its file reference, and the chunks/entities/summaries cognify derived from it (`delete_data_nodes_and_edges`) |
| insert | `cognee.add(DataItem(data=text, data_id=uuid4(), external_metadata={"invalidate_source", "invalidate_supersedes", "invalidate_event_id"}), dataset_id=..., node_set=insert_node_set)` ([add.py:35](https://github.com/topoteretes/cognee/blob/main/cognee/api/v1/add/add.py), [data_item.py:14](https://github.com/topoteretes/cognee/blob/main/cognee/tasks/ingestion/data_item.py)) | the successor is a new row with an id known up front (a pinned `data_id` becomes the row id and skips content dedup, [ingest_data.py:277-283, 484](https://github.com/topoteretes/cognee/blob/main/cognee/tasks/ingestion/ingest_data.py)). `add()` "stages data and makes no LLM call of its own" (add.py:230); **no `cognify()` is run**, so the successor is in the dataset and in the next `sync()` but not in the graph until you cognify. Without an importable `DataItem` (older cognee) the id is recovered by diffing `list_data` before/after, or by content hash on a dedup no-op |
| search | `cognee.search(query_text, query_type=SearchType.HYBRID_COMPLETION, user=None, datasets=None, dataset_ids=None, top_k=15, ...)` ([search.py:41](https://github.com/topoteretes/cognee/blob/main/cognee/api/v1/search/search.py)) | `governed_search` adds `dataset_ids=[adapter.dataset_id]` (and `user`) when you give no scope, then filters |

## How search results expose ids (what `governed_search` filters on)

- With backend access control (the default whenever the graph and vector DBs support it) `search` returns
  `[{"dataset_id", "dataset_name", "dataset_tenant_id", "search_result": <result>}]`; otherwise the bare
  `<result>` ([modules/search/methods/search.py:632-700](https://github.com/topoteretes/cognee/blob/main/cognee/modules/search/methods/search.py)).
  Both shapes are handled.
- `SearchType.CHUNKS` / `CHUNKS_LEXICAL`: `<result>` is `[{**chunk.payload, "score"}]`
  ([chunks_retriever.py:57](https://github.com/topoteretes/cognee/blob/main/cognee/modules/retrieval/chunks_retriever.py));
  the payload is the `DocumentChunk` fields, including `document_id`, which is `str(document.id)`
  ([TextChunker.py:17](https://github.com/topoteretes/cognee/blob/main/cognee/modules/chunking/TextChunker.py))
  and the Document node is created with `id=data_item.id`
  ([classify_documents.py:169](https://github.com/topoteretes/cognee/blob/main/cognee/tasks/documents/classify_documents.py)).
  So `document_id == str(Data.id)`, the host id in the ledger. **These are filtered.**
- `SearchType.SUMMARIES`: payloads carry the summary's own `id` and `text` but no document id
  ([summaries_retriever.py:114](https://github.com/topoteretes/cognee/blob/main/cognee/modules/retrieval/summaries_retriever.py)). **Pass through.**
- Completion types (`GRAPH_COMPLETION`, `RAG_COMPLETION`, `HYBRID_COMPLETION`, ...): strings; the LLM has
  already read the stale chunk. **Pass through.** Use `mode="delete"` if these must not see dead facts.

## What `flag` can and cannot do (be precise)

- Cognee's public API has **no per-item metadata update**. `cognee.update(data_id, data, dataset_id, ...)`
  replaces the *content* and re-cognifies ([update.py:28](https://github.com/topoteretes/cognee/blob/main/cognee/api/v1/update/update.py));
  passing a `DataItem` with `external_metadata` forces a full rebuild (drop memory, re-add, cognify:
  update.py:364-370), which needs an LLM and rewrites the graph. invalidate never rewrites fact text, so
  this is not used.
- Default: `flag` writes nothing to Cognee. The ledger holds the verdict; `governed_search` /
  `filter_results` apply it at read time. `Governor(mode="ledger")` (or `mode="flag"`, which is the same
  for Cognee plus successor inserts when `successors=True`) is the supported path.
- Opt-in: `CogneeAdapter(..., flag_in_host=True)` merges `Reason.as_metadata()` into
  `Data.external_metadata` (the "user's free-form field", [Data.py](https://github.com/topoteretes/cognee/blob/main/cognee/modules/data/models/Data.py))
  through `cognee.infrastructure.databases.relational.get_relational_engine()` and a session
  merge/commit. This is an internal API, not `cognee.*`; the receipt is visible in
  `datasets.list_data(...)[i].external_metadata` and on the Document node after the next cognify, but it
  changes **no** search result. `node_set` and other keys in `external_metadata` are preserved (merge, not
  replace). `keep()` writes an `invalidate_status: active` receipt the same way.
- `mode="delete"` is the only mode that changes what Cognee's own completions see.

## Limits

- **Unit is the data item, not the extracted fact.** Cognee's chunks, entities and summaries cannot be
  deleted one by one (`delete_data` is per row), so a long document is judged as one memory and dies as
  one. Add short facts as separate `cognee.add()` calls if you want per-fact verdicts.
- **Successors are not cognified.** `insert` only stages the row; run `cognee.cognify()` (needs an LLM
  key) to put it in the graph. The row is pulled on the next `sync()` under the same id, so the ledger
  stays consistent either way.
- **Only text rows are pulled** by default. PDFs, images and audio have their original file (or an S3
  object) as `raw_data_location`; pass `include_non_text=True` to pull them as raw bytes decoded with
  `errors="replace"`, or supply `read_text=` to extract text your own way.
- `pull` reads local files directly, bypassing Cognee's path allowlist; the paths come from Cognee's own
  rows. Remote mode (`cognee.serve()`) returns `DataDTO` rows whose `raw_data_location` is on the server:
  provide `read_text=` there.
- `add()` validates the embedding provider config (`validate_provider_config(needs_llm=False)`); keyless
  ingestion with the local fastembed embedder is supported by Cognee, but the environment must be set up
  the way Cognee expects.
- Sync-over-async: when a loop is already running in the calling thread, each call runs on a short-lived
  helper thread with its own loop. Cognee's engines are process-global and cope with that, but prefer
  plain scripts or a worker thread for bulk work, or call `cognee.search` yourself and use
  `filter_results`.
- The V2 memory API (`cognee.remember` / `recall` / `forget`), `memify`, node sets as a unit, and
  multi-dataset governance are out of scope: one adapter = one dataset.
