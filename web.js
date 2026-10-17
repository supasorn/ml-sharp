const express = require('express');
const multer = require('multer');
const path = require('path');
const { execFile } = require('child_process');

const app = express();
// Password protection with --pw (asks for the ZIP password at start; visitors log in): see basic_auth.js.
app.use(require('./basic_auth')());
const PORT = 3000;
const imgdir = 'images';


// Bare root goes to the gallery; with query params (?scene=..., etc.) it's the viewer.
app.get('/', (req, res) => {
  if (Object.keys(req.query).length === 0) return res.redirect('/list');
  res.sendFile(path.join(__dirname, 'web', 'index.html'));
});

// List all .json files in output2 as links
const fs = require('fs');

// Scene names (json basenames in output2), most recently modified first.
function sortedSceneNames(dir) {
  return fs.readdirSync(dir)
    .filter(f => f.endsWith('.json'))
    .map(f => ({ name: f.replace(/\.json$/, ''), mtime: fs.statSync(path.join(dir, f)).mtimeMs }))
    .sort((a, b) => b.mtime - a.mtime)
    .map(s => s.name)
    .filter(name => name != "rgb" && name != "rgb2");
}

// --- 360 panoramas (docs/pano-360.md) ---
// Scene id "pano:<name>" for pano/inputs/<name>.jpg. Content: multi-sphere images (scripts/pano_msi.py) in
// pano/<variant dir>/<name>_msi[_ktx2].glb and splats pano/out_ico2/<name>.spz (scripts/pano_spz.py).
const PANO_DIR = path.join(__dirname, 'pano');
const PANO_PREFIX = 'pano:';
// MSI variants, best first: the viewer shows the first one a pano has (the others are selectable).
const PANO_VARIANTS = [
  { id: 'out_ico2_photo4', label: 'Photo detail v4' },
  { id: 'out_ico2_photo2', label: 'Photo detail v2' },
  { id: 'out_ico2_photo3', label: 'Photo detail v3 (dark specks)' },
  { id: 'out_ico2_hires', label: 'High-res' },
  { id: 'out_ico2', label: 'Standard' },
];
const isPanoScene = (scene) => typeof scene === 'string' && scene.startsWith(PANO_PREFIX);
function panoInfo(name) {
  if (!/^[\w.-]+$/.test(name)) return null;
  const input = ['.jpg', '.jpeg', '.png'].map((e) => path.join(PANO_DIR, 'inputs', name + e)).find((p) => fs.existsSync(p));
  if (!input) return null;
  let mtime = fs.statSync(input).mtimeMs;
  const variants = [];
  for (const v of PANO_VARIANTS) {
    const ktx2 = path.join(PANO_DIR, v.id, `${name}_msi_ktx2.glb`), png = path.join(PANO_DIR, v.id, `${name}_msi.glb`);
    if (!fs.existsSync(ktx2) && !fs.existsSync(png)) continue;
    variants.push({ ...v, msi: fs.existsSync(ktx2) ? `/pano/${v.id}/${name}_msi_ktx2.glb` : null,
                    msiPng: fs.existsSync(png) ? `/pano/${v.id}/${name}_msi.glb` : null });
    mtime = Math.max(mtime, fs.statSync(fs.existsSync(ktx2) ? ktx2 : png).mtimeMs);
  }
  const spz = fs.existsSync(path.join(PANO_DIR, 'out_ico2', `${name}.spz`)) ? `/pano/out_ico2/${name}.spz` : null;
  if (!variants.length && !spz) return null;
  return { id: PANO_PREFIX + name, name, input, mtime, variants, spz };
}
function panoScenes() {
  const dir = path.join(PANO_DIR, 'inputs');
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).filter((f) => /\.(jpe?g|png)$/i.test(f))
    .map((f) => panoInfo(f.replace(/\.[^.]+$/, ''))).filter(Boolean);
}
function panoModes(p) {
  return [...(p.variants.length ? ['msi'] : []), ...(p.spz ? ['splats'] : [])];
}
// Photo scenes and panoramas in one list, most recently modified first.
function allSceneEntries() {
  const dir = path.join(__dirname, 'output2');
  const photos = sortedSceneNames(dir).map((name) => ({ id: name, mtime: fs.statSync(path.join(dir, name + '.json')).mtimeMs }));
  return [...photos, ...panoScenes().map((p) => ({ id: p.id, mtime: p.mtime, pano: p }))].sort((a, b) => b.mtime - a.mtime);
}
app.get('/pano-info/:name', (req, res) => {
  const p = panoInfo(req.params.name);
  if (!p) return res.sendStatus(404);
  const { input, ...info } = p;
  res.json({ ...info, modes: panoModes(p) });
});
app.use('/pano', express.static(PANO_DIR));

// FPS reports from the viewer while in XR: printed, and appended to
// logs/xr_perf.jsonl (one JSON object per line) for later analysis.
const PERF_LOG = path.join(__dirname, 'logs', 'xr_perf.jsonl');
app.post('/fps', express.json(), (req, res) => {
  console.log('[xr fps]', JSON.stringify(req.body));
  // The browser's user agent identifies the device (Quest / phone / desktop) beyond the client's own guess.
  const entry = { time: new Date().toISOString(), ...req.body, ua: (req.headers['user-agent'] || '').slice(0, 200) };
  fs.mkdir(path.dirname(PERF_LOG), { recursive: true }, () => {
    fs.appendFile(PERF_LOG, JSON.stringify(entry) + '\n', (err) => {
      if (err) console.error('Could not write', PERF_LOG, err.message);
    });
  });
  res.sendStatus(204);
});

// Small JPEG thumbnails (in-XR gallery, pending items), made once with the sharp conda env's Pillow
// (EXIF orientation applied, HEIC via pillow_heif) and cached.
const THUMB_DIR = path.join(__dirname, 'renderings', 'thumbs');
const THUMB_PX = 512;   // gallery cards are 250 css px (500 device px on high-DPI screens)
function sendThumb(src, out, res, what) {
  if (fs.existsSync(out) && fs.statSync(out).mtimeMs >= fs.statSync(src).mtimeMs) return res.sendFile(out);
  fs.mkdirSync(THUMB_DIR, { recursive: true });
  const py = 'import sys\nfrom PIL import Image, ImageOps\ntry:\n import pillow_heif; pillow_heif.register_heif_opener()\n' +
             'except ImportError:\n pass\nim = ImageOps.exif_transpose(Image.open(sys.argv[1])).convert("RGB")\n' +
             `im.thumbnail((${THUMB_PX}, ${THUMB_PX}))\nim.save(sys.argv[2], quality=85)`;
  execFile('/conda_envs/sharp/bin/python', ['-c', py, src, out], (err) => {
    if (err) { console.error('thumbnail failed:', what, err.message); return res.sendStatus(500); }
    res.sendFile(out);
  });
}
app.get('/thumb/:scene', (req, res) => {
  const scene = req.params.scene;
  if (isPanoScene(scene)) {
    const p = panoInfo(scene.slice(PANO_PREFIX.length));
    if (!p) return res.sendStatus(404);
    return sendThumb(p.input, path.join(THUMB_DIR, `_pano_${p.name}_${THUMB_PX}.jpg`), res, scene);
  }
  const jsonPath = path.join(__dirname, 'output2', scene + '.json');
  if (scene.includes('/') || scene.includes('..') || !fs.existsSync(jsonPath)) return res.sendStatus(404);
  let image;
  try { image = JSON.parse(fs.readFileSync(jsonPath, 'utf8')).image; } catch (e) { return res.sendStatus(404); }
  const src = path.join(__dirname, 'images', image || '');
  if (!image || !fs.existsSync(src)) return res.sendStatus(404);
  sendThumb(src, path.join(THUMB_DIR, `${scene}_${THUMB_PX}.jpg`), res, scene);
});
// Thumbnail of a photo in images/ by file name (pending items have no scene yet).
app.get('/thumb-img/:file', (req, res) => {
  const file = req.params.file;
  const src = path.join(__dirname, imgdir, file);
  if (file.includes('/') || file.includes('..') || !fs.existsSync(src)) return res.sendStatus(404);
  sendThumb(src, path.join(THUMB_DIR, `_img_${file}_${THUMB_PX}.jpg`), res, file);
});

// Photos in images/ that have no scene yet (being predicted, or waiting for it), newest first.
const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp', '.avif', '.tif', '.tiff', '.bmp']);
function pendingImages() {
  const outDir = path.join(__dirname, 'output2');
  const used = new Set();
  for (const f of fs.readdirSync(outDir)) {
    if (!f.endsWith('.json')) continue;
    used.add(f.slice(0, -5));
    try { const im = JSON.parse(fs.readFileSync(path.join(outDir, f), 'utf8')).image; if (im) used.add(im); } catch (e) {}
  }
  return fs.readdirSync(path.join(__dirname, imgdir))
    .filter((f) => IMAGE_EXTS.has(path.extname(f).toLowerCase()))
    .filter((f) => !used.has(f) && !used.has(path.basename(f, path.extname(f))))
    .map((f) => ({ file: f, mtime: fs.statSync(path.join(__dirname, imgdir, f)).mtimeMs }))
    .sort((a, b) => b.mtime - a.mtime);
}
app.get('/pending', (req, res) => {
  try { res.json(pendingImages()); } catch (e) { res.status(500).send('Cannot read directory'); }
});

// Import from Google Photos (Picker API): see google_photos.js for setup; credentials in secrets/.
require('./google_photos')(app, { imagesDir: path.join(__dirname, imgdir), secretsDir: path.join(__dirname, 'secrets') });

// View modes a scene can be shown in (gallery page and in-XR gallery grey out the rest).
function sceneModes(dir, name) {
  const has = (suffix) => fs.existsSync(path.join(dir, name + suffix));
  const modes = ['full'];
  if (has('_hybrid.glb') && has('_hybrid.ply')) modes.push('hybrid');
  if (has('_mpi.glb')) modes.push('mpi');
  if (has('_mpionly.glb')) modes.push('mpionly');
  return modes;
}
app.get('/scene-modes', (req, res) => {
  const dir = path.join(__dirname, 'output2');
  try {
    res.json(Object.fromEntries(allSceneEntries().map((s) => [s.id, s.pano ? panoModes(s.pano) : sceneModes(dir, s.id)])));
  } catch (e) {
    res.status(500).send('Cannot read directory');
  }
});

// Same order as /list, as JSON (used by the viewer to step through scenes); panoramas are "pano:<name>".
app.get('/scenes', (req, res) => {
  try {
    res.json(allSceneEntries().map((s) => s.id));
  } catch (e) {
    res.status(500).send('Cannot read directory');
  }
});
app.get('/list', (req, res) => {
  const dir = path.join(__dirname, 'output2');
  // get "ui" get parameter
  const ui = req.query.ui || 'true';

  {
    let entries;
    try { entries = allSceneEntries(); } catch (e) { res.status(500).send('Cannot read directory'); return; }
    // Photo scenes and 360 panoramas (badge, own view modes), most recently modified first.
    const items = entries.map(({ id: name, pano }) => {
      // Cached thumbnail (fast) instead of the full photo; made on first request.
      const imgSrc = `/thumb/${encodeURIComponent(name)}`;
      if (pano) {
        return `
        <div class="scene-item pano" data-kind="pano" data-modes="${panoModes(pano).join(' ')}" title="360° panorama: ${pano.name}">
          <a href='/?scene=${encodeURIComponent(name)}&ui=${ui}'>
            <div class="pano-wrap"><img src='${imgSrc}' alt='${pano.name}' loading='lazy' /><span class="pano-badge">360°</span></div>
          </a>
        </div>
      `;
      }
      return `
        <div class="scene-item" data-modes="${sceneModes(dir, name).join(' ')}">
          <a href='/?scene=${encodeURIComponent(name)}&ui=${ui}'>
            <img src='${imgSrc}' alt='${name}' loading='lazy' />
          </a>
        </div>
      `;
    }).join('\n');
    // Read the template and inject the items
    const templatePath = path.join(__dirname, 'web', 'list.html');
    let template = fs.readFileSync(templatePath, 'utf8');
    // Photos still being processed: dimmed thumbnail with a spinner (not clickable), newest first.
    let pending = '';
    try {
      pending = pendingImages().map(({ file, mtime }) => {
        const stale = Date.now() - mtime > 30 * 60 * 1000;
        return `
        <div class="scene-item pending" data-file="${encodeURIComponent(file)}" title="${file}">
          <div class="pending-wrap">
            <img src='/thumb-img/${encodeURIComponent(file)}' alt='${file}' loading='lazy' />
            ${stale ? '' : '<div class="pending-spinner"></div>'}
            <div class="pending-label">${stale ? 'Still not processed — check logs/predict.log' : 'Processing…'}</div>
          </div>
        </div>`;
      }).join('\n');
    } catch (e) {}
    template = template.replace('<!-- SCENE_ITEMS_PLACEHOLDER -->', pending + items);
    res.send(template);
  }
});


// Optionally serve static files from 'web' directory
app.use('/output2', express.static(path.join(__dirname, 'output2')));

// Optionally serve static files from 'web' directory
app.use('/images', express.static(path.join(__dirname, 'images')));

// Viewer modules (e.g. web/js/XRSceneController.js)
app.use('/js', express.static(path.join(__dirname, 'web', 'js')));

const storage = multer.diskStorage({
  destination: function(req, file, cb) {
    cb(null, path.join(__dirname, imgdir));
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
app.post('/upload', upload.array('files[]'), async (req, res) => {
  try {
    await Promise.all(req.files.map(async (file) => {
      const ext = path.extname(file.originalname).toLowerCase();

      if (ext === '.jpg' || ext === '.jpeg' || ext === '.png' || ext === '.webp') {
              // const newFileName = file.path.replace(/\.[^/.]+$/, '_resized.jpg');
        // console.log("change from ", file.path, " to ", newFileName);
        // await sharp(file.path).rotate().resize({ height: 1400 }).toFile(newFileName);
        // fs.unlinkSync(file.path); // Delete original file

        // no resize anymore, just rotate with exif
        console.log("process image ", file.path);
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

// Delete scene and associated files
app.post('/delete', express.urlencoded({ extended: true }), (req, res) => {
  const scene = req.body.scene;
  if (!scene) return res.status(400).send('No scene specified');
  // Panoramas live in pano/ with several baked variants: not deleted from the gallery.
  if (isPanoScene(scene)) return res.status(400).send({ success: false, error: 'Panoramas are not deleted here' });

  // Delete image from /images
  const outputDir = path.join(__dirname, 'output2');
  let imageName = '';
  try {
    const jsonPath = path.join(outputDir, scene + '.json');
    if (fs.existsSync(jsonPath)) {
      const jsonData = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
      imageName = jsonData.image || '';
      if (imageName) {
        const imgPath = path.join(__dirname, 'images', imageName);
        if (fs.existsSync(imgPath)) fs.unlinkSync(imgPath);
      }
    }
  } catch (e) {}

  // Delete all files in output2/ with the same base name, plus files derived
  // from it (bake-hybrid / bake-layers / quest-lod outputs and previews).
  // Exact suffixes, so deleting "foo" never touches "foo_2".
  const derived = ['_hybrid.glb', '_hybrid.ply', '_surfaces.glb', '_patches.glb', '_mpi.glb', '_mpionly.glb', '_mpi_ktx2.glb', '_mpionly_ktx2.glb', '_surfaces_mesh.glb', '_patches_mesh.glb', '_mpi_mesh.glb', '_layers.glb', '_quest.ply',
                   '_hybrid_vs_splats.jpg', '_hybrid_coverage.jpg', '_layers_vs_splats.jpg']
    .map(suffix => scene + suffix);
  const files = fs.readdirSync(outputDir);
  files.forEach(f => {
    if (f.startsWith(scene + '.') || derived.includes(f)) {
      fs.unlinkSync(path.join(outputDir, f));
    }
  });

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
