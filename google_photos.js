// Import from Google Photos via the Google Photos Picker API.
//
// You pick photos in Google's own picker (phone or browser); the server
// downloads the originals into images/, where the prediction watcher and the
// bake watchers take over as for photos dropped on the gallery.
//
// Setup (once): a Google Cloud project with the "Photos Picker API" enabled
// and an OAuth client of type "Web application" (consent screen in testing
// mode, your account added as a test user). Put its credentials in
// secrets/google_oauth.json (git-ignored):
//   { "client_id": "...", "client_secret": "...",
//     "redirect_uri": "http://localhost:3000/google/callback" }
// and add the same redirect URI to the OAuth client. Google only accepts
// localhost or public domains as redirect URIs, so sign in once from a
// browser that reaches the server as localhost (on the server itself, or
// through `ssh -L 3000:localhost:3000 <server>`). The refresh token is kept in
// secrets/google_token.json; imports then work from any device.
//
// Endpoints:
//   GET  /google/status        -> { configured, signedIn, redirectUri }
//   GET  /google/login         -> redirect to Google sign-in
//   GET  /google/callback      -> OAuth redirect target (stores the token)
//   POST /google/session       -> { id, pickerUri } (open pickerUri, pick photos)
//   GET  /google/session/:id   -> { state, total, done, files, error }
//   POST /google/session/:id/check  (user is back: check the selection now instead of at the next poll)
//   POST /google/logout        -> forget the stored token
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const SCOPE = 'https://www.googleapis.com/auth/photospicker.mediaitems.readonly';
const PICKER = 'https://photospicker.googleapis.com/v1';

// imagesDir: a folder, or a function (req) -> folder for an import started by that request (null: refused).
module.exports = function googlePhotos(app, { imagesDir, secretsDir }) {
  const importDir = typeof imagesDir === 'function' ? imagesDir : () => imagesDir;
  const credFile = path.join(secretsDir, 'google_oauth.json');
  const tokenFile = path.join(secretsDir, 'google_token.json');
  const sessions = new Map();   // id -> { state, total, done, files, error, pickerUri }
  const oauthStates = new Set();

  const readJson = (f) => { try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch (e) { return null; } };
  // Accepts our flat format or Google's downloaded client file ({"web": {...}} or {"installed": {...}} for a
  // "Desktop app" client, whose localhost redirects need no registration). redirect_uri is optional.
  const creds = () => {
    const raw = readJson(credFile);
    const c = raw && (raw.web || raw.installed || raw);
    if (!c || !c.client_id || !c.client_secret) return null;
    const listed = (c.redirect_uris || []).find((u) => u.includes('/google/callback'));
    return { ...c, redirect_uri: c.redirect_uri || listed || 'http://localhost:3000/google/callback' };
  };
  const saveToken = (t) => {
    fs.mkdirSync(secretsDir, { recursive: true });
    fs.writeFileSync(tokenFile, JSON.stringify(t, null, 1), { mode: 0o600 });
  };

  async function accessToken() {
    const c = creds(), t = readJson(tokenFile);
    if (!c || !t) throw new Error('not signed in');
    if (t.access_token && t.expires_at > Date.now() + 60000) return t.access_token;
    if (!t.refresh_token) throw new Error('no refresh token; sign in again');
    const r = await fetch('https://oauth2.googleapis.com/token', {
      method: 'POST',
      body: new URLSearchParams({ client_id: c.client_id, client_secret: c.client_secret,
                                  refresh_token: t.refresh_token, grant_type: 'refresh_token' }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(`token refresh failed: ${j.error_description || j.error || r.status}`);
    saveToken({ ...t, access_token: j.access_token, expires_at: Date.now() + j.expires_in * 1000 });
    return j.access_token;
  }

  async function api(method, url, body) {
    const r = await fetch(url, {
      method,
      headers: { Authorization: `Bearer ${await accessToken()}`, 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    });
    const text = await r.text();
    const j = text ? JSON.parse(text) : {};
    if (!r.ok) throw new Error(`${method} ${url.replace(PICKER, '')}: ${j.error?.message || r.status}`);
    return j;
  }

  // "5s" / "2.5s" (protobuf Duration as JSON) -> ms
  const durMs = (d, dflt) => (typeof d === 'string' && d.endsWith('s') ? parseFloat(d) * 1000 : dflt);

  // Same naming as gallery uploads: <name>_<random>.<ext>, so imports never overwrite a scene.
  function targetName(filename, mime) {
    const ext = (path.extname(filename || '') ||
      ({ 'image/jpeg': '.jpg', 'image/png': '.png', 'image/heic': '.heic', 'image/webp': '.webp' }[mime] || '.jpg'))
      .toLowerCase();
    const base = path.basename(filename || 'photo', path.extname(filename || '')).replace(/[^\w.\-]+/g, '_') || 'photo';
    return `${base}_${Math.round(Math.random() * 1e5)}${ext}`;
  }

  async function runSession(id) {
    const s = sessions.get(id);
    try {
      // 1. Wait for the selection (poll as the API asks; give up after its timeout or 30 min).
      let info = await api('GET', `${PICKER}/sessions/${id}`);
      const deadline = Date.now() + Math.min(durMs(info.pollingConfig?.timeoutIn, 30 * 60e3), 30 * 60e3);
      while (!info.mediaItemsSet) {
        if (Date.now() > deadline) throw new Error('timed out waiting for a selection');
        // Sleep until the next poll, or until the gallery reports the user is back (POST .../check).
        await new Promise((r) => {
          const timer = setTimeout(r, Math.max(durMs(info.pollingConfig?.pollInterval, 5000), 2000));
          s.wake = () => { clearTimeout(timer); r(); };
        });
        s.wake = null;
        info = await api('GET', `${PICKER}/sessions/${id}`);
      }
      // 2. List the picked items.
      s.state = 'downloading';
      let items = [], pageToken = '';
      do {
        const q = new URLSearchParams({ sessionId: id, pageSize: '100', ...(pageToken ? { pageToken } : {}) });
        const page = await api('GET', `${PICKER}/mediaItems?${q}`);
        items = items.concat(page.mediaItems || []);
        pageToken = page.nextPageToken || '';
      } while (pageToken);
      const photos = items.filter((m) => m.type !== 'VIDEO' && m.mediaFile?.baseUrl);
      s.total = photos.length;
      s.skipped = items.length - photos.length;
      // 3. Download originals ("=d"; the base URLs expire after 60 min). Each file is written to its final
      // name in one go: the prediction watcher reacts when an image file is closed.
      for (const m of photos) {
        const r = await fetch(`${m.mediaFile.baseUrl}=d`, { headers: { Authorization: `Bearer ${await accessToken()}` } });
        if (!r.ok) { s.failed = (s.failed || 0) + 1; continue; }
        const name = targetName(m.mediaFile.filename, m.mediaFile.mimeType);
        fs.writeFileSync(path.join(s.dir, name), Buffer.from(await r.arrayBuffer()));
        s.files.push(name);
        s.done++;
        console.log(`[google] imported ${name} (${s.done}/${s.total})`);
      }
      s.state = 'done';
    } catch (e) {
      s.state = 'error';
      s.error = e.message;
      console.error('[google] import failed:', e.message);
    } finally {
      api('DELETE', `${PICKER}/sessions/${id}`).catch(() => {});
    }
  }

  app.get('/google/status', (req, res) => {
    const c = creds(), t = readJson(tokenFile);
    res.json({ configured: !!c, signedIn: !!(c && t && t.refresh_token), redirectUri: c?.redirect_uri || null });
  });

  app.get('/google/login', (req, res) => {
    const c = creds();
    if (!c) return res.status(400).send('Google Photos import is not configured: see google_photos.js for setup.');
    const state = crypto.randomBytes(16).toString('hex');
    oauthStates.add(state);
    const q = new URLSearchParams({ client_id: c.client_id, redirect_uri: c.redirect_uri, response_type: 'code',
                                    scope: SCOPE, access_type: 'offline', prompt: 'consent', state });
    res.redirect(`https://accounts.google.com/o/oauth2/v2/auth?${q}`);
  });

  app.get('/google/callback', async (req, res) => {
    const c = creds();
    if (!c || !oauthStates.delete(req.query.state)) return res.status(400).send('Invalid sign-in state; try again.');
    if (req.query.error) return res.status(400).send(`Google sign-in failed: ${req.query.error}`);
    try {
      const r = await fetch('https://oauth2.googleapis.com/token', {
        method: 'POST',
        body: new URLSearchParams({ code: req.query.code, client_id: c.client_id, client_secret: c.client_secret,
                                    redirect_uri: c.redirect_uri, grant_type: 'authorization_code' }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error_description || j.error || r.status);
      const old = readJson(tokenFile) || {};
      saveToken({ refresh_token: j.refresh_token || old.refresh_token, access_token: j.access_token,
                  expires_at: Date.now() + j.expires_in * 1000 });
      res.send('<p style="font:16px sans-serif">Signed in to Google Photos. You can close this tab and use '
               + '"Import from Google" in the gallery (from any device).</p>');
    } catch (e) {
      res.status(500).send(`Google sign-in failed: ${e.message}`);
    }
  });

  app.post('/google/session', async (req, res) => {
    const dir = importDir(req);
    if (!dir) return res.status(403).json({ error: 'not allowed to import there (log in for the private collection)' });
    try {
      const j = await api('POST', `${PICKER}/sessions`, {});
      sessions.set(j.id, { state: 'picking', total: 0, done: 0, files: [], error: null, pickerUri: j.pickerUri, dir });
      runSession(j.id);
      res.json({ id: j.id, pickerUri: j.pickerUri });
    } catch (e) {
      res.status(400).json({ error: e.message });
    }
  });

  // The gallery tab regained focus (user is back from the picker): check the selection now.
  app.post('/google/session/:id/check', (req, res) => {
    const s = sessions.get(req.params.id);
    if (!s) return res.status(404).json({ error: 'unknown session' });
    if (s.wake) s.wake();
    res.sendStatus(204);
  });

  app.get('/google/session/:id', (req, res) => {
    const s = sessions.get(req.params.id);
    if (!s) return res.status(404).json({ error: 'unknown session' });
    const { wake, ...out } = s;
    res.json(out);
  });

  app.post('/google/logout', (req, res) => {
    try { fs.unlinkSync(tokenFile); } catch (e) {}
    res.sendStatus(204);
  });
};
