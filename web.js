const express = require('express');
const multer = require('multer');
const path = require('path');
const { execFile } = require('child_process');

const app = express();
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

// FPS reports from the viewer while in XR: printed, and appended to
// logs/xr_perf.jsonl (one JSON object per line) for later analysis.
const PERF_LOG = path.join(__dirname, 'logs', 'xr_perf.jsonl');
app.post('/fps', express.json(), (req, res) => {
  console.log('[xr fps]', JSON.stringify(req.body));
  const entry = { time: new Date().toISOString(), ...req.body };
  fs.mkdir(path.dirname(PERF_LOG), { recursive: true }, () => {
    fs.appendFile(PERF_LOG, JSON.stringify(entry) + '\n', (err) => {
      if (err) console.error('Could not write', PERF_LOG, err.message);
    });
  });
  res.sendStatus(204);
});

// Small JPEG thumbnail of a scene's photo (for the in-XR gallery), made once with
// the sharp conda env's Pillow (EXIF orientation applied) and cached.
const THUMB_DIR = path.join(__dirname, 'renderings', 'thumbs');
app.get('/thumb/:scene', (req, res) => {
  const scene = req.params.scene;
  const jsonPath = path.join(__dirname, 'output2', scene + '.json');
  if (scene.includes('/') || scene.includes('..') || !fs.existsSync(jsonPath)) return res.sendStatus(404);
  const out = path.join(THUMB_DIR, scene + '.jpg');
  if (fs.existsSync(out)) return res.sendFile(out);
  let image;
  try { image = JSON.parse(fs.readFileSync(jsonPath, 'utf8')).image; } catch (e) { return res.sendStatus(404); }
  const src = path.join(__dirname, 'images', image || '');
  if (!image || !fs.existsSync(src)) return res.sendStatus(404);
  fs.mkdirSync(THUMB_DIR, { recursive: true });
  const py = 'import sys\nfrom PIL import Image, ImageOps\nim = ImageOps.exif_transpose(Image.open(sys.argv[1])).convert("RGB")\n' +
             'im.thumbnail((384, 384))\nim.save(sys.argv[2], quality=85)';
  execFile('/conda_envs/sharp/bin/python', ['-c', py, src, out], (err) => {
    if (err) { console.error('thumbnail failed:', scene, err.message); return res.sendStatus(500); }
    res.sendFile(out);
  });
});

// View modes a scene can be shown in (gallery page and in-XR gallery grey out the rest).
function sceneModes(dir, name) {
  const has = (suffix) => fs.existsSync(path.join(dir, name + suffix));
  const modes = ['full'];
  if (has('_hybrid.glb') && has('_hybrid.ply')) modes.push('hybrid');
  if (has('_mpi.glb')) modes.push('mpi');
  return modes;
}
app.get('/scene-modes', (req, res) => {
  const dir = path.join(__dirname, 'output2');
  try {
    res.json(Object.fromEntries(sortedSceneNames(dir).map((n) => [n, sceneModes(dir, n)])));
  } catch (e) {
    res.status(500).send('Cannot read directory');
  }
});

// Same order as /list, as JSON (used by the viewer to step through scenes).
app.get('/scenes', (req, res) => {
  try {
    res.json(sortedSceneNames(path.join(__dirname, 'output2')));
  } catch (e) {
    res.status(500).send('Cannot read directory');
  }
});
app.get('/list', (req, res) => {
  const dir = path.join(__dirname, 'output2');
  // get "ui" get parameter
  const ui = req.query.ui || 'true';

  fs.readdir(dir, (err, files) => {
    if (err) {
      res.status(500).send('Cannot read directory');
      return;
    }
    // Sort json files by modified time, recent first
    const jsonFiles = files.filter(f => f.endsWith('.json'));
    jsonFiles.sort((a, b) => {
      const aTime = fs.statSync(path.join(dir, a)).mtimeMs;
      const bTime = fs.statSync(path.join(dir, b)).mtimeMs;
      return bTime - aTime;
    });
    const items = jsonFiles.map(f => {
      const name = f.replace(/\.json$/, '');
      // read the content of the json file to get the image name
      let imageName = '';
      if (name == "rgb" || name == "rgb2") return '';
      try {
        const jsonPath = path.join(dir, f);
        const jsonData = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
        imageName = jsonData.image || '';
      } catch (e) {
        imageName = '';
      }
      const imgSrc = `/images/${encodeURIComponent(imageName)}`;
      return `
        <div class="scene-item" data-modes="${sceneModes(dir, name).join(' ')}">
          <a href='/?scene=${encodeURIComponent(name)}&ui=${ui}'>
            <img src='${imgSrc}' alt='${name}' />
          </a>
        </div>
      `;
            // <div class="scene-filename">${f}</div>
    }).join('\n');
    // Read the template and inject the items
    const templatePath = path.join(__dirname, 'web', 'list.html');
    let template = fs.readFileSync(templatePath, 'utf8');
    template = template.replace('<!-- SCENE_ITEMS_PLACEHOLDER -->', items);
    res.send(template);
  });
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
  const derived = ['_hybrid.glb', '_hybrid.ply', '_surfaces.glb', '_patches.glb', '_mpi.glb', '_surfaces_mesh.glb', '_patches_mesh.glb', '_mpi_mesh.glb', '_layers.glb', '_quest.ply',
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
