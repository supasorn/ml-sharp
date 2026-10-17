// Data collections: the same layout (images/, output2/, pano/, renderings/) under each root.
//   data/public    plain files (the old top-level images/, output2/, ... are symlinks into it)
//   data/private   gocryptfs mount of data/private.cipher (mounted by hand outside this repo's tools);
//                  used only while mounted, and served only to logged-in visitors (web.js --pw)
// Scene ids: public scenes keep their plain names ("<name>", "pano:<name>"); private ones are prefixed
// ("private:<name>", "private:pano:<name>").
const fs = require('fs');
const path = require('path');

const ROOT = __dirname;
const COLLECTIONS = {
  public: { id: 'public', root: path.join(ROOT, 'data', 'public'), private: false },
  private: { id: 'private', root: path.join(ROOT, 'data', 'private'), private: true },
};
const LAYOUT = ['images', 'output2', 'pano', 'renderings'];

/** A collection is usable when its root holds the layout (an unmounted gocryptfs mount point is empty). */
function mounted(c) {
  try { return fs.existsSync(path.join(c.root, 'output2')) && fs.existsSync(path.join(c.root, 'images')); }
  catch (e) { return false; }
}

/** Parse a scene id -> { coll, pano, name } (null for an invalid id). */
function parseId(id) {
  if (typeof id !== 'string') return null;
  let coll = COLLECTIONS.public, rest = id;
  if (rest.startsWith('private:')) { coll = COLLECTIONS.private; rest = rest.slice('private:'.length); }
  const pano = rest.startsWith('pano:');
  const name = pano ? rest.slice('pano:'.length) : rest;
  if (!name || name.includes('/') || name.includes('\\') || name.startsWith('.')) return null;
  return { coll, pano, name };
}

/** Scene id for a name in a collection. */
function makeId(coll, name, pano = false) {
  return (coll.private ? 'private:' : '') + (pano ? 'pano:' : '') + name;
}

const dir = (coll, sub) => path.join(coll.root, sub);

module.exports = { COLLECTIONS, LAYOUT, mounted, parseId, makeId, dir };
