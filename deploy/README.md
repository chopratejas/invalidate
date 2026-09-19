# invalidate playground on Vercel

A stateless copy of the `invalidate ui` demo (conversation landing page at `/`, two-box paste
playground at `/paste`), packaged as four Python Vercel Functions (Fluid Compute, Python runtime)
plus two static pages. Nothing is persisted; every request builds a
throwaway in-memory `Invalidate`, judged by Jev via the TypeSafe SDK.

```
deploy/
  api/chat.py         POST /api/chat     -> {"ok": true, "reply": "...", "facts": [...], "changes": [...], ...}
  api/story.py        GET  /api/story    -> {"ok": true, "story": [...]}
  api/presets.py      GET  /api/presets  -> {"ok": true, "presets": {...}}
  api/check.py        POST /api/check    -> {"ok": true, "facts": [...], "steps": [...], "summary": {...}}
  public/index.html   landing page / chat demo (GENERATED: copy of src/invalidate/ui/static/chat.html)
  public/paste.html   paste playground at /paste (GENERATED: copy of src/invalidate/ui/static/playground.html)
  invalidate/         GENERATED: copy of src/invalidate (minus ui/static and __pycache__)
  requirements.txt    typesafe-sdk
  vercel.json         framework null; rewrites / -> /index.html, /paste -> /paste.html; includeFiles
  sync.sh             regenerates the two GENERATED entries
```

`invalidate/` and `public/*.html` are copies, not symlinks (Vercel uploads do not follow symlinks
reliably). Edit the originals under `src/invalidate`, then re-sync.

## 1. Sync the package

```bash
./deploy/sync.sh
```

Run this after every change under `src/invalidate`, before deploying.

## 2. Set the API key

`api/check.py` reads `TYPESAFE_API_KEY` from the environment. Add it per environment; the
value is prompted for (or piped in), never stored in this directory:

```bash
cd deploy
vercel env add TYPESAFE_API_KEY preview
vercel env add TYPESAFE_API_KEY production
```

Non-interactive (the value never hits the terminal):

```bash
grep '^TYPESAFE_API_KEY=' ../.env | cut -d= -f2- | tr -d '\n' | vercel env add TYPESAFE_API_KEY preview
```

## 3. Deploy

Production is live at https://invalidate-playground.vercel.app. Both Preview and Production have the key set.

```bash
cd deploy
vercel link --yes --project invalidate-playground   # first time only
vercel --yes                                        # PREVIEW deployment (safe default)
vercel --prod --yes                                 # PRODUCTION (only when you mean it)
```

`vercel --yes` gives you a preview URL. Preview deployments are usually behind Vercel
Authentication; to hit one from the CLI use `vercel curl <path>` (see the
`vercel:access-protected-vercel-deployment` skill), e.g.

```bash
vercel curl /api/presets
vercel curl /api/check -X POST -H 'content-type: application/json' \
  -d '{"facts":"user prefers Postgres","events":"we migrated to SQLite last Tuesday"}'
```

## Limits baked into the functions

- `invalidate.ui.check`: at most 60 facts, 25 events, 600 chars per line.
- `invalidate.ui.chat`: at most 80 facts carried per turn, 1200 chars of text.
- `api/check.py`: 64 KB request body cap; 20 checks per 10 minutes per IP.
- `api/chat.py`: 64 KB request body cap; 40 turns per 10 minutes per IP.
  Rate limits are in-memory per function instance (a soft limit; use the Vercel Firewall for a
  hard one). All responses carry `Cache-Control: no-store`.
- Errors: `400` bad input (`{"ok": false, "error": "..."}`), `413` body too large, `429` rate limited,
  `500` missing `TYPESAFE_API_KEY` or unexpected failure, `502` TypeSafe upstream error.
