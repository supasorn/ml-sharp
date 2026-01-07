const express = require('express');
const path = require('path');

const app = express();
const PORT = 3000;

// Serve index.html at root
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'web', 'index.html'));
});

// Optionally serve static files from 'web' directory
app.use('/output2', express.static(path.join(__dirname, 'output2')));

// 404 handler for other routes
app.use((req, res) => {
  res.status(404).send('404 Not Found');
});

app.listen(PORT, () => {
  console.log(`Express server running at http://localhost:${PORT}/`);
});
