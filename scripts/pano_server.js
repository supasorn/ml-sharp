// Standalone server for the 360 panorama viewer (separate from web.js).
//   node scripts/pano_server.js [--pw]   ->  http://<host>:3100/  (https on 3543 if certs/ exist; WebXR needs it;
//   --pw: visitors must log in, see basic_auth.js)
// Lists the scenes under pano/<dir>/: splats (<name>.spz, scripts/pano_spz.py) and multi-sphere images
// (<name>_msi.glb / <name>_msi_ktx2.glb, scripts/pano_msi.py).
const express = require('express');
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const panoDir = path.join(root, 'pano');
const app = express();
// Password protection with --pw (asks for the ZIP password at start; visitors log in): see basic_auth.js.
app.use(require(path.join(root, 'basic_auth'))());

app.get('/', (req, res) => res.sendFile(path.join(root, 'web', 'pano.html')));

app.get('/scenes', (req, res) => {
  const scenes = new Map();
  const mb = (f) => +(fs.statSync(f).size / 1e6).toFixed(1);
  for (const dir of fs.readdirSync(panoDir)) {
    const full = path.join(panoDir, dir);
    if (dir.startsWith('_') || !fs.statSync(full).isDirectory()) continue;
    for (const f of fs.readdirSync(full)) {
      const m = /^(.*?)(\.spz|_msi\.glb|_msi_ktx2\.glb)$/.exec(f);
      if (!m) continue;
      const id = `${dir}/${m[1]}`;
      if (!scenes.has(id)) {
        const json = path.join(full, `${m[1]}.json`);
        const info = fs.existsSync(json) ? JSON.parse(fs.readFileSync(json, 'utf8')) : {};
        scenes.set(id, { id, name: m[1], dir, layout: info.layout, splats: info.splats });
      }
      const s = scenes.get(id);
      const url = `/files/${dir}/${f}`;
      if (m[2] === '.spz') Object.assign(s, { spz: url, spzMB: mb(path.join(full, f)) });
      else if (m[2] === '_msi.glb') Object.assign(s, { msiPng: url });
      else Object.assign(s, { msi: url, msiMB: mb(path.join(full, f)) });
    }
  }
  res.json([...scenes.values()].sort((a, b) => a.id.localeCompare(b.id)));
});

app.use('/files', express.static(panoDir));

const PORT = 3100, HTTPS_PORT = 3543;
app.listen(PORT, () => console.log(`Pano viewer at http://localhost:${PORT}/`));
const key = path.join(root, 'certs', 'key.pem'), cert = path.join(root, 'certs', 'cert.pem');
if (fs.existsSync(key) && fs.existsSync(cert)) {
  require('https')
    .createServer({ key: fs.readFileSync(key), cert: fs.readFileSync(cert) }, app)
    .listen(HTTPS_PORT, () => console.log(`Pano viewer at https://localhost:${HTTPS_PORT}/`));
}
