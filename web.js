const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const { execFile } = require('child_process');
const { COLLECTIONS, mounted, parseId, makeId, dir } = require('./collections');

const app = express();
const PORT = 3000;

// Data collections (collections.js). The server shows one of them, chosen at start:
//   --pw and data/private mounted (gocryptfs): the private collection only. Asks for the ZIP password
//     (basic_auth.js); every request needs the login; nothing is cached by browsers.
//   otherwise: the public collection, open to everyone (--pw without the mount: a warning, no prompt).
// Uploads and Google imports go to the shown collection.
const PUB = COLLECTIONS.public, PRIV = COLLECTIONS.private;
const wantPrivate = process.argv.includes('--pw');
const PRIVATE_MODE = wantPrivate && mounted(PRIV);
if (wantPrivate && !PRIVATE_MODE) console.warn('--pw: data/private is not mounted; showing the public collection.');
const login = PRIVATE_MODE ? require('./basic_auth').createLogin() : null;
const ACTIVE = PRIVATE_MODE ? PRIV : PUB;
console.log(PRIVATE_MODE ? 'Showing the private collection (visitors log in).' : 'Showing the public collection.');
if (PRIVATE_MODE) {
  app.use((req, res, next) => {
    if (!login.check(req)) return login.challenge(req, res);
    res.set('Cache-Control', 'no-store');   // keep private pages and files out of browser caches
    next();
  });
}
/** Collections this request may see. */
const visible = () => [ACTIVE];
/** Access check for a collection: only the shown one exists (404 otherwise). */
function allow(req, res, coll) {
  if (coll === ACTIVE && (!coll.private || mounted(coll))) return true;
  res.sendStatus(404);
  return false;
}
const collOf = () => ACTIVE;


// Bare root goes to the gallery; with query params (?scene=..., etc.) it's the viewer.
app.get('/', (req, res) => {
  if (Object.keys(req.query).length === 0) return res.redirect('/list');
  res.set('Cache-Control', 'no-cache');   // revalidate every load: viewer changes show on a plain reload
  res.sendFile(path.join(__dirname, 'web', 'index.html'));
});

// Old login link: in private mode the login is already required for everything.
app.get('/login', (req, res) => res.redirect('/list'));

// Scene names (json basenames in an output2 dir), most recently modified first.
function sortedSceneNames(outDir) {
  if (!fs.existsSync(outDir)) return [];
  return fs.readdirSync(outDir)
    .filter(f => f.endsWith('.json'))
    .map(f => ({ name: f.replace(/\.json$/, ''), mtime: fs.statSync(path.join(outDir, f)).mtimeMs }))
    .sort((a, b) => b.mtime - a.mtime)
    .map(s => s.name)
    .filter(name => name != "rgb" && name != "rgb2");
}

// --- 360 panoramas (docs/pano-360.md) ---
// Scene id "[private:]pano:<name>" for <collection>/pano/inputs/<name>.jpg. Content: multi-sphere images
// (scripts/pano_msi.py) in pano/<variant dir>/<name>_msi[_ktx2].glb and splats pano/out_ico2/<name>.spz.
// MSI variants, best first: the viewer shows the first one a pano has (the others are selectable).
const PANO_VARIANTS = [
  { id: 'out_ico2_photo4', label: 'Photo detail v4' },
  { id: 'out_ico2_photo2', label: 'Photo detail v2' },
  { id: 'out_ico2_photo3', label: 'Photo detail v3 (dark specks)' },
  { id: 'out_ico2_hires', label: 'High-res' },
  { id: 'out_ico2', label: 'Standard' },
];
function panoInfo(coll, name) {
  if (!/^[\w.-]+$/.test(name)) return null;
  const panoDir = dir(coll, 'pano');
  const input = ['.jpg', '.jpeg', '.png'].map((e) => path.join(panoDir, 'inputs', name + e)).find((p) => fs.existsSync(p));
  if (!input) return null;
  const url = (rel) => `/c/${coll.id}/pano/${rel}`;
  let mtime = fs.statSync(input).mtimeMs;
  const variants = [];
  for (const v of PANO_VARIANTS) {
    const ktx2 = path.join(panoDir, v.id, `${name}_msi_ktx2.glb`), png = path.join(panoDir, v.id, `${name}_msi.glb`);
    if (!fs.existsSync(ktx2) && !fs.existsSync(png)) continue;
    variants.push({ ...v, msi: fs.existsSync(ktx2) ? url(`${v.id}/${name}_msi_ktx2.glb`) : null,
                    msiPng: fs.existsSync(png) ? url(`${v.id}/${name}_msi.glb`) : null });
    mtime = Math.max(mtime, fs.statSync(fs.existsSync(ktx2) ? ktx2 : png).mtimeMs);
  }
  const spz = fs.existsSync(path.join(panoDir, 'out_ico2', `${name}.spz`)) ? url(`out_ico2/${name}.spz`) : null;
  if (!variants.length && !spz) return null;
  return { id: makeId(coll, name, true), name, input, mtime, variants, spz };
}
function panoScenes(coll) {
  const inputs = path.join(dir(coll, 'pano'), 'inputs');
  if (!fs.existsSync(inputs)) return [];
  return fs.readdirSync(inputs).filter((f) => /\.(jpe?g|png)$/i.test(f))
    .map((f) => panoInfo(coll, f.replace(/\.[^.]+$/, ''))).filter(Boolean);
}
function panoModes(p) {
  return [...(p.variants.length ? ['msi'] : []), ...(p.spz ? ['splats'] : [])];
}
// Photo scenes and panoramas of the given collections in one list, most recently modified first.
function allSceneEntries(colls) {
  const out = [];
  for (const coll of colls) {
    const outDir = dir(coll, 'output2');
    for (const name of sortedSceneNames(outDir)) {
      out.push({ id: makeId(coll, name), coll, name, mtime: fs.statSync(path.join(outDir, name + '.json')).mtimeMs });
    }
    for (const p of panoScenes(coll)) out.push({ id: p.id, coll, name: p.name, mtime: p.mtime, pano: p });
  }
  return out.sort((a, b) => b.mtime - a.mtime);
}
// Info of a panorama by scene id ("[private:]pano:<name>"; a bare name means a public panorama).
app.get('/pano-info/:id', (req, res) => {
  let s = parseId(req.params.id);
  if (s && !s.pano) s = parseId(makeId(s.coll, s.name, true));
  if (!s) return res.sendStatus(404);
  if (!allow(req, res, s.coll)) return;
  const p = panoInfo(s.coll, s.name);
  if (!p) return res.sendStatus(404);
  const { input, ...info } = p;
  res.json({ ...info, modes: panoModes(p) });
});

// Per-scene view settings (scene size in XR, depth scale and offset; set in the viewer and saved as they
// change): <collection>/output2/<name>.viewcfg, panoramas <collection>/pano/inputs/<name>.viewcfg. Deleting
// a scene removes it with its other <name>.* files.
const VIEW_KEYS = ['scale', 'depthScale', 'depthOffset'];
const viewCfgPath = (s) => path.join(dir(s.coll, s.pano ? path.join('pano', 'inputs') : 'output2'), `${s.name}.viewcfg`);
// Every model of a scene, for the viewer's picker: its view modes (sceneModes: incl. every MPI-only variant bake).
app.get('/scene-models/:scene', (req, res) => {
  const s = parseId(req.params.scene);
  if (!s) return res.sendStatus(404);
  if (!allow(req, res, s.coll)) return;
  if (s.pano) return res.json([]);
  try {
    const outDir = dir(s.coll, 'output2');
    res.json(sceneModes(outDir, s.name).map((m) => ({ lod: m })));
  } catch (e) {
    res.status(500).send('Cannot read directory');
  }
});

app.get('/view-config/:scene', (req, res) => {
  const s = parseId(req.params.scene);
  if (!s) return res.sendStatus(404);
  if (!allow(req, res, s.coll)) return;
  try { res.json(JSON.parse(fs.readFileSync(viewCfgPath(s), 'utf8'))); } catch (e) { res.json({}); }
});
app.post('/view-config/:scene', express.json(), (req, res) => {
  const s = parseId(req.params.scene);
  if (!s) return res.sendStatus(404);
  if (!allow(req, res, s.coll)) return;
  const cfg = {};
  for (const k of VIEW_KEYS) {
    const v = Number(req.body?.[k]);
    if (Number.isFinite(v)) cfg[k] = v;
  }
  if (!fs.existsSync(path.dirname(viewCfgPath(s)))) return res.sendStatus(404);
  fs.writeFile(viewCfgPath(s), JSON.stringify(cfg), (err) => res.sendStatus(err ? 500 : 204));
});

// Collection files of the shown collection: /c/<id>/...; in public mode also the old paths
// (/output2, /images, /pano) for existing links.
app.use(`/c/${ACTIVE.id}`, express.static(ACTIVE.root));
if (!PRIVATE_MODE) {
  app.use('/output2', express.static(dir(PUB, 'output2')));
  app.use('/images', express.static(dir(PUB, 'images')));
  app.use('/pano', express.static(dir(PUB, 'pano')));
}

// FPS reports from the viewer while in XR: printed, and appended to
// logs/xr_perf.jsonl (one JSON object per line) for later analysis. Private scene names are hashed.
const PERF_LOG = path.join(__dirname, 'logs', 'xr_perf.jsonl');
app.post('/fps', express.json(), (req, res) => {
  const body = { ...req.body };
  if (typeof body.scene === 'string' && body.scene.startsWith('private:')) {
    body.scene = 'private:' + crypto.createHash('sha256').update(body.scene).digest('hex').slice(0, 12);
  }
  console.log('[xr fps]', JSON.stringify(body));
  // The browser's user agent identifies the device (Quest / phone / desktop) beyond the client's own guess.
  const entry = { time: new Date().toISOString(), ...body, ua: (req.headers['user-agent'] || '').slice(0, 200) };
  fs.mkdir(path.dirname(PERF_LOG), { recursive: true }, () => {
    fs.appendFile(PERF_LOG, JSON.stringify(entry) + '\n', (err) => {
      if (err) console.error('Could not write', PERF_LOG, err.message);
    });
  });
  res.sendStatus(204);
});

// Small JPEG thumbnails (in-XR gallery, pending items), made once with the sharp conda env's Pillow ($SHARP_ENV)
// (EXIF orientation applied, HEIC via pillow_heif) and cached in the collection's renderings/thumbs.
const THUMB_PX = 512;   // gallery cards are 250 css px (500 device px on high-DPI screens)
const thumbDir = (coll) => path.join(dir(coll, 'renderings'), 'thumbs');
function sendThumb(coll, src, outName, res, what) {
  const out = path.join(thumbDir(coll), outName);
  if (fs.existsSync(out) && fs.statSync(out).mtimeMs >= fs.statSync(src).mtimeMs) return res.sendFile(out);
  fs.mkdirSync(thumbDir(coll), { recursive: true });
  const py = 'import sys\nfrom PIL import Image, ImageOps\ntry:\n import pillow_heif; pillow_heif.register_heif_opener()\n' +
             'except ImportError:\n pass\nim = ImageOps.exif_transpose(Image.open(sys.argv[1])).convert("RGB")\n' +
             `im.thumbnail((${THUMB_PX}, ${THUMB_PX}))\nim.save(sys.argv[2], quality=85)`;
  execFile(path.join(process.env.SHARP_ENV || '/conda_envs/sharp', 'bin', 'python'), ['-c', py, src, out], (err) => {
    if (err) { console.error('thumbnail failed:', what, err.message); return res.sendStatus(500); }
    res.sendFile(out);
  });
}
app.get('/thumb/:scene', (req, res) => {
  const s = parseId(req.params.scene);
  if (!s) return res.sendStatus(404);
  if (!allow(req, res, s.coll)) return;
  if (s.pano) {
    const p = panoInfo(s.coll, s.name);
    if (!p) return res.sendStatus(404);
    return sendThumb(s.coll, p.input, `_pano_${p.name}_${THUMB_PX}.jpg`, res, req.params.scene);
  }
  const jsonPath = path.join(dir(s.coll, 'output2'), s.name + '.json');
  if (!fs.existsSync(jsonPath)) return res.sendStatus(404);
  let image;
  try { image = JSON.parse(fs.readFileSync(jsonPath, 'utf8')).image; } catch (e) { return res.sendStatus(404); }
  const src = path.join(dir(s.coll, 'images'), image || '');
  if (!image || image.includes('/') || !fs.existsSync(src)) return res.sendStatus(404);
  sendThumb(s.coll, src, `${s.name}_${THUMB_PX}.jpg`, res, req.params.scene);
});
// Thumbnail of a photo in a collection's images/ by file name (pending items have no scene yet).
function thumbImg(req, res, coll, file) {
  if (!allow(req, res, coll)) return;
  const src = path.join(dir(coll, 'images'), file);
  if (file.includes('/') || file.includes('..') || !fs.existsSync(src)) return res.sendStatus(404);
  sendThumb(coll, src, `_img_${file}_${THUMB_PX}.jpg`, res, file);
}
app.get('/thumb-img/:file', (req, res) => thumbImg(req, res, PUB, req.params.file));
app.get('/thumb-img/:coll/:file', (req, res) => {
  const coll = COLLECTIONS[req.params.coll];
  if (!coll) return res.sendStatus(404);
  thumbImg(req, res, coll, req.params.file);
});

// Photos in a collection's images/ that have no scene yet (being predicted, or waiting for it), newest first.
const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp', '.avif', '.tif', '.tiff', '.bmp']);
function pendingImages(coll) {
  const outDir = dir(coll, 'output2'), imgDir = dir(coll, 'images');
  if (!fs.existsSync(imgDir)) return [];
  const used = new Set();
  for (const f of fs.existsSync(outDir) ? fs.readdirSync(outDir) : []) {
    if (!f.endsWith('.json')) continue;
    used.add(f.slice(0, -5));
    try { const im = JSON.parse(fs.readFileSync(path.join(outDir, f), 'utf8')).image; if (im) used.add(im); } catch (e) {}
  }
  return fs.readdirSync(imgDir)
    .filter((f) => IMAGE_EXTS.has(path.extname(f).toLowerCase()))
    .filter((f) => !used.has(f) && !used.has(path.basename(f, path.extname(f))))
    .map((f) => ({ file: f, coll: coll.id, mtime: fs.statSync(path.join(imgDir, f)).mtimeMs }))
    .sort((a, b) => b.mtime - a.mtime);
}
const allPending = (req) => visible(req).flatMap(pendingImages).sort((a, b) => b.mtime - a.mtime);
app.get('/pending', (req, res) => {
  try { res.json(allPending(req)); } catch (e) { res.status(500).send('Cannot read directory'); }
});

// Import from Google Photos (Picker API): see google_photos.js for setup; credentials in secrets/.
// Photos go to the shown collection.
require('./google_photos')(app, { imagesDir: () => dir(ACTIVE, 'images'), secretsDir: path.join(__dirname, 'secrets') });

// View modes a scene can be shown in (gallery page and in-XR gallery grey out the rest).
function sceneModes(outDir, name) {
  const has = (suffix) => fs.existsSync(path.join(outDir, name + suffix));
  const modes = ['full'];
  if (has('_hybrid.glb') && has('_hybrid.ply')) modes.push('hybrid');
  if (has('_mpi.glb')) modes.push('mpi');
  if (has('_mpionly.glb')) modes.push('mpionly');
  // MPI-only variant bakes (<name>_mpionly_<tag>.glb, e.g. surf*, ldi, ad25): view modes named by their tag.
  const pre = `${name}_mpionly_`;
  try {
    for (const f of fs.readdirSync(outDir).sort()) {
      const tag = f.startsWith(pre) && f.endsWith('.glb') ? f.slice(pre.length, -4) : '';
      if (/^[a-z0-9]+$/i.test(tag) && tag !== 'ktx2' && tag !== 'astc') modes.push(tag);   // not compressed copies
    }
  } catch (e) {}
  return modes;
}
const entryModes = (s) => (s.pano ? panoModes(s.pano) : sceneModes(dir(s.coll, 'output2'), s.name));
app.get('/scene-modes', (req, res) => {
  try {
    res.json(Object.fromEntries(allSceneEntries(visible(req)).map((s) => [s.id, entryModes(s)])));
  } catch (e) {
    res.status(500).send('Cannot read directory');
  }
});

// Same order as /list, as JSON (used by the viewer to step through scenes); panoramas are "pano:<name>",
// private scenes "private:...".
app.get('/scenes', (req, res) => {
  try {
    res.json(allSceneEntries(visible(req)).map((s) => s.id));
  } catch (e) {
    res.status(500).send('Cannot read directory');
  }
});
app.get('/list', (req, res) => {
  // get "ui" get parameter
  const ui = req.query.ui || 'true';
  let entries;
  try { entries = allSceneEntries(visible(req)); } catch (e) { res.status(500).send('Cannot read directory'); return; }
  // Photo scenes and 360 panoramas (badge, own view modes), most recently modified first; private ones
  // with a lock badge.
  const items = entries.map((s) => {
    const name = s.id;
    // Cached thumbnail (fast) instead of the full photo; made on first request.
    const imgSrc = `/thumb/${encodeURIComponent(name)}`;
    const priv = s.coll.private;
    const cls = `scene-item${s.pano ? ' pano' : ''}${priv ? ' private' : ''}`;
    const lock = priv ? '<span class="private-badge" title="Private collection">🔒</span>' : '';
    if (s.pano) {
      return `
        <div class="${cls}" data-kind="pano" data-modes="${entryModes(s).join(' ')}" title="360° panorama: ${s.name}">
          <a href='/?scene=${encodeURIComponent(name)}&ui=${ui}'>
            <div class="pano-wrap"><img src='${imgSrc}' alt='${s.name}' loading='lazy' /><span class="pano-badge">360°</span>${lock}</div>
          </a>
        </div>
      `;
    }
    return `
        <div class="${cls}" data-modes="${entryModes(s).join(' ')}">
          <a href='/?scene=${encodeURIComponent(name)}&ui=${ui}'>
            <div class="pano-wrap"><img src='${imgSrc}' alt='${s.name}' loading='lazy' />${lock}</div>
          </a>
        </div>
      `;
  }).join('\n');
  // Read the template and inject the items
  let template = fs.readFileSync(path.join(__dirname, 'web', 'list.html'), 'utf8');
  // Photos still being processed: dimmed thumbnail with a spinner (not clickable), newest first.
  let pending = '';
  try {
    pending = allPending(req).map(({ file, coll, mtime }) => {
      const stale = Date.now() - mtime > 30 * 60 * 1000;
      return `
        <div class="scene-item pending" data-file="${encodeURIComponent(coll + '/' + file)}" title="${file}">
          <div class="pending-wrap">
            <img src='/thumb-img/${coll}/${encodeURIComponent(file)}' alt='${file}' loading='lazy' />
            ${stale ? '' : '<div class="pending-spinner"></div>'}
            <div class="pending-label">${stale ? 'Still not processed — check logs/predict.log' : 'Processing…'}</div>
          </div>
        </div>`;
    }).join('\n');
  } catch (e) {}
  const state = { mode: ACTIVE.id };
  template = template.replace('<!-- PRIVATE_STATE -->', `<script>window.PRIVATE_STATE = ${JSON.stringify(state)};</script>`);
  template = template.replace('<!-- SCENE_ITEMS_PLACEHOLDER -->', pending + items);
  res.send(template);
});


// Viewer modules (e.g. web/js/XRSceneController.js)
app.use('/js', express.static(path.join(__dirname, 'web', 'js')));

// Uploads go to the shown collection.
const storage = multer.diskStorage({
  destination: function(req, file, cb) {
    cb(null, dir(collOf(req), 'images'));
  },
  filename: function(req, file, cb) {
    let base = path.basename(file.originalname, path.extname(file.originalname));

    // const randomName = Date.now() + '-' + Math.round(Math.random() * 1E9);
    const randomName = Math.round(Math.random() * 1E5);
    let finalName = base + '_' + randomName + path.extname(file.originalname).toLowerCase();

    cb(null, finalName);
  }
});

const upload = multer({ storage: storage });
app.post('/upload', (req, res, next) => { if (allow(req, res, collOf(req))) next(); }, upload.array('files[]'), async (req, res) => {
  try {
    await Promise.all(req.files.map(async (file) => {
      const ext = path.extname(file.originalname).toLowerCase();

      if (ext === '.jpg' || ext === '.jpeg' || ext === '.png' || ext === '.webp') {
              // const newFileName = file.path.replace(/\.[^/.]+$/, '_resized.jpg');
        // console.log("change from ", file.path, " to ", newFileName);
        // await sharp(file.path).rotate().resize({ height: 1400 }).toFile(newFileName);
        // fs.unlinkSync(file.path); // Delete original file

        // no resize anymore, just rotate with exif
        console.log("process image ", collOf(req).private ? '(private)' : file.path);
        // fs.unlinkSync(file.path);
        // const newFileName = file.path.replace(/\.[^/.]+$/, '_tmp.jpg');
        // await sharp(file.path).rotate().resize({ height: 1400 }).toFile(newFileName);
        // fs.unlinkSync(file.path); // Delete original file
        // fs.renameSync(newFileName, file.path);
      }
    }));

    res.send('Files uploaded and processed successfully.');
  } catch (error) {
    console.error('Error processing files:', error);
    res.status(500).send('Error processing files');
  }
});

// Delete scene and associated files (within its collection)
app.post('/delete', express.urlencoded({ extended: true }), (req, res) => {
  const s = parseId(req.body.scene);
  if (!s) return res.status(400).send('No scene specified');
  if (!allow(req, res, s.coll)) return;
  // Panoramas live in pano/ with several baked variants: not deleted from the gallery.
  if (s.pano) return res.status(400).send({ success: false, error: 'Panoramas are not deleted here' });
  const scene = s.name;

  // Delete image from images/
  const outputDir = dir(s.coll, 'output2');
  let imageName = '';
  try {
    const jsonPath = path.join(outputDir, scene + '.json');
    if (fs.existsSync(jsonPath)) {
      const jsonData = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
      imageName = jsonData.image || '';
      if (imageName && !imageName.includes('/')) {
        const imgPath = path.join(dir(s.coll, 'images'), imageName);
        if (fs.existsSync(imgPath)) fs.unlinkSync(imgPath);
      }
    }
  } catch (e) {}

  // Delete all files in output2/ with the same base name, plus files derived
  // from it (bake-hybrid / bake-layers / quest-lod outputs and previews).
  // Exact suffixes, so deleting "foo" never touches "foo_2".
  const derived = ['_hybrid.glb', '_hybrid.ply', '_surfaces.glb', '_patches.glb', '_mpi.glb', '_mpionly.glb', '_mpi_ktx2.glb', '_mpionly_ktx2.glb', '_mpi_astc.glb', '_mpionly_astc.glb', '_surfaces_mesh.glb', '_patches_mesh.glb', '_mpi_mesh.glb', '_layers.glb', '_quest.ply',
                   '_hybrid_vs_splats.jpg', '_hybrid_coverage.jpg', '_layers_vs_splats.jpg']
    .map(suffix => scene + suffix);
  const files = fs.readdirSync(outputDir);
  files.forEach(f => {
    if (f.startsWith(scene + '.') || derived.includes(f) || /^_mpionly_[a-z0-9]+(_ktx2|_astc)?\.glb$/i.test(f.slice(scene.length)) && f.startsWith(scene)) {
      fs.unlinkSync(path.join(outputDir, f));
    }
  });
  // Cached thumbnail too (a private one must not outlive its scene).
  try { fs.unlinkSync(path.join(thumbDir(s.coll), `${scene}_${THUMB_PX}.jpg`)); } catch (e) {}

  res.send({ success: true });
});

// 404 handler for other routes
app.use((req, res) => {
  res.status(404).send('404 Not Found');
});


app.listen(PORT, () => {
  console.log(`Express server running at http://localhost:${PORT}/`);
});

// HTTPS is required for WebXR (AR on Meta Quest) when not on localhost.
// Enabled when certs/key.pem and certs/cert.pem exist.
const HTTPS_PORT = 3443;
const keyPath = path.join(__dirname, 'certs', 'key.pem');
const certPath = path.join(__dirname, 'certs', 'cert.pem');
if (fs.existsSync(keyPath) && fs.existsSync(certPath)) {
  require('https')
    .createServer({ key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath) }, app)
    .listen(HTTPS_PORT, () => {
      console.log(`HTTPS server running at https://localhost:${HTTPS_PORT}/`);
    });
}
