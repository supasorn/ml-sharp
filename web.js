const express = require('express');
const path = require('path');

const app = express();
const PORT = 3000;

// Serve index.html at root
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'web', 'index.html'));
});

// List all .json files in output2 as links
const fs = require('fs');
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
        <div class="scene-item">
          <a href='/?scene=${encodeURIComponent(name)}&ui=${ui}'>
            <img src='${imgSrc}' alt='${name}' />
            <div class="scene-filename">${f}</div>
          </a>
        </div>
      `;
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

// 404 handler for other routes
app.use((req, res) => {
  res.status(404).send('404 Not Found');
});

app.listen(PORT, () => {
  console.log(`Express server running at http://localhost:${PORT}/`);
});
