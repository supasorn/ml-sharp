const express = require('express');
var https = require('https');
const multer = require('multer');
const path = require('path');

const app = express();
const PORT = 3000;
const imgdir = 'images';


// Serve index.html at root
app.get('/', (req, res) => {
  // redirect to /list if no scene parameter
  if (!req.query.scene) {
    res.redirect('/list');
    return;
  }
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
          <a href='/?scene=${encodeURIComponent(name)}&ui=${ui}&hybrid=true'>
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

  // Delete all files in output2/ with the same base name
  const files = fs.readdirSync(outputDir);
  files.forEach(f => {
    if (f.startsWith(scene + '.')) {
      fs.unlinkSync(path.join(outputDir, f));
    }
  });

  res.send({ success: true });
});

// 404 handler for other routes
app.use((req, res) => {
  res.status(404).send('404 Not Found');
});


const options = {
  key: fs.readFileSync('server.key'),
  cert: fs.readFileSync('server.cert')
};

https.createServer(options, app).listen(PORT, () => {
  console.log(`Express server running at https://localhost:${PORT}/`);
});
