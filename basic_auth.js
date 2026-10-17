// Password protection (HTTP Basic Auth) for web.js and scripts/pano_server.js, on when the server is
// started with --pw (e.g. `node web.js --pw`); without it the site is open.
// With --pw the server asks for the ZIP password on its terminal (hidden; or takes $PASSWORD, e.g. for
// detached starts) and decrypts the login stored below (AES-256-CBC, key = SHA-256 of the ZIP password,
// same scheme as img3dviewer). Visitors then log in with that user name and password. Nothing secret is
// stored on disk or logged; a wrong ZIP password stops the server instead of serving unprotected.
// Use the HTTPS ports from other devices: Basic Auth over plain HTTP is readable on the network.
const crypto = require('crypto');
const fs = require('fs');
const { execSync } = require('child_process');

const ENCRYPTED_LOGIN = '116323e299ca287f27559bebdad8737b:40d99ad252a679d37beafce886a8b72a';

function decrypt(text, secret) {
  try {
    const key = crypto.createHash('sha256').update(secret).digest();
    const [iv, ...rest] = text.split(':');
    const decipher = crypto.createDecipheriv('aes-256-cbc', key, Buffer.from(iv, 'hex'));
    return decipher.update(Buffer.from(rest.join(':'), 'hex'), undefined, 'utf8') + decipher.final('utf8');
  } catch (e) {
    return null;
  }
}

/** Read a line from the terminal without echo, synchronously (before the server starts listening). */
function askHidden(query) {
  const tty = fs.openSync('/dev/tty', 'r+');
  fs.writeSync(tty, query);
  const stty = (args) => execSync(`stty ${args}`, { stdio: [tty, 'ignore', 'ignore'] });
  stty('-echo');
  let line = '';
  try {
    const buf = Buffer.alloc(1);
    while (fs.readSync(tty, buf, 0, 1) === 1 && buf[0] !== 10 && buf[0] !== 13) line += buf.toString('latin1');
  } finally {
    stty('echo');
    fs.writeSync(tty, '\n');
    fs.closeSync(tty);
  }
  return Buffer.from(line, 'latin1').toString('utf8');
}

function zipPassword() {
  const fromEnv = process.env.PASSWORD;
  delete process.env.PASSWORD;                    // not inherited by anything this server starts
  if (fromEnv) return fromEnv;
  try {
    return askHidden('ZIP Pass: ');
  } catch (e) {
    console.error('--pw: no terminal to ask for the ZIP password; set PASSWORD instead.');
    process.exit(1);
  }
}

function sameText(a, b) {
  const x = crypto.createHash('sha256').update(String(a)).digest();
  const y = crypto.createHash('sha256').update(String(b)).digest();
  return crypto.timingSafeEqual(x, y);
}

/** Express middleware: a pass-through without --pw, else Basic Auth with the decrypted login. */
module.exports = function basicAuth(realm = 'ml-sharp') {
  if (!process.argv.includes('--pw')) {
    console.log('Password protection off (start with --pw to require a login).');
    return (req, res, next) => next();
  }
  const dec = decrypt(ENCRYPTED_LOGIN, zipPassword());
  const i = dec ? dec.indexOf(':') : -1;
  if (i <= 0) {
    console.error('--pw: wrong ZIP password (cannot decrypt the login). Not starting.');
    process.exit(1);
  }
  const user = dec.slice(0, i), password = dec.slice(i + 1);
  console.log('Password protection on: visitors log in.');
  return (req, res, next) => {
    const m = /^Basic (.+)$/.exec(req.headers.authorization || '');
    if (m) {
      const text = Buffer.from(m[1], 'base64').toString('utf8');
      const j = text.indexOf(':');
      if (j >= 0 && sameText(text.slice(0, j), user) & sameText(text.slice(j + 1), password)) return next();
    }
    res.set('WWW-Authenticate', `Basic realm="${realm}", charset="UTF-8"`);
    // Failed attempts answer after a short delay (slows guessing).
    setTimeout(() => res.status(401).send('Password required'), m ? 500 : 0);
  };
};
