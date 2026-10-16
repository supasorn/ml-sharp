/**
 * XRGallery: a scene gallery inside the AR/VR session.
 *
 * A floating panel in front of you: a row of view-mode buttons (Original
 * splats / Hybrid / Hybrid-MPI), a grid of scene thumbnails (the current
 * scene outlined in white) and page arrows. Point at an item with either
 * controller's ray and pull the trigger to pick it; thumbstick left/right also
 * pages. XRSceneController opens/closes it and routes input to it while open.
 *
 * Everything draws on top of the scene (no depth test) so splats can't hide it.
 */
import * as THREE from "three";

const COLS = 6, ROWS = 4, PER_PAGE = COLS * ROWS;
const CELL_W = 0.16, CELL_H = 0.12, GAP = 0.015;
const DIST = 0.9;                       // metres in front of the head
const RAY_LEN = 3;

function overlayMaterial(opts) {
  return new THREE.MeshBasicMaterial({ depthTest: false, depthWrite: false, transparent: true, ...opts });
}

/** A plane showing text drawn on a canvas; `.setText()` redraws it. */
function label(w, h, text, { bg = "rgba(40,40,40,0.9)", fg = "#fff", px = 64 } = {}) {
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(w * 1600); canvas.height = Math.round(h * 1600);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), overlayMaterial({ map: tex }));
  mesh.setText = (t, colors = {}) => {
    const c = canvas.getContext("2d");
    c.clearRect(0, 0, canvas.width, canvas.height);
    c.fillStyle = colors.bg ?? bg;
    c.beginPath(); c.roundRect(0, 0, canvas.width, canvas.height, canvas.height * 0.2); c.fill();
    c.fillStyle = colors.fg ?? fg;
    c.font = `bold ${px}px sans-serif`;
    c.textAlign = "center"; c.textBaseline = "middle";
    c.fillText(t, canvas.width / 2, canvas.height / 2);
    tex.needsUpdate = true;
  };
  mesh.setText(text);
  return mesh;
}

export class XRGallery {
  /**
   * @param {THREE.WebGLRenderer} renderer
   * @param {THREE.Scene} scene
   * @param {object} o
   *   listScenes(): Promise<string[]> scene names in gallery order.
   *   sceneModes(): Promise<{name: string[]}> view modes each scene has (others are greyed out).
   *   thumbUrl(name): thumbnail URL.
   *   current(): current scene name;  lod(): current view mode id.
   *   lods: [{ id, label }] view modes;  onSelectScene(name);  onSelectLod(id).
   */
  constructor(renderer, scene, o) {
    this.renderer = renderer;
    this.o = o;
    this.group = new THREE.Group();
    this.group.visible = false;
    this.group.renderOrder = 1000;
    scene.add(this.group);
    this.names = [];
    this.page = 0;
    this.thumbs = new Map();             // name -> texture (kept across pages)
    this.loader = new THREE.TextureLoader();
    this.targets = [];                   // pickable meshes: { mesh, action }
    this.hover = null;
    this._stickMoved = false;

    const gridW = COLS * CELL_W + (COLS - 1) * GAP, gridH = ROWS * CELL_H + (ROWS - 1) * GAP;
    this.gridW = gridW; this.gridH = gridH;
    const bg = new THREE.Mesh(new THREE.PlaneGeometry(gridW + 0.06, gridH + 0.2),
                              overlayMaterial({ color: 0x000000, opacity: 0.75 }));
    bg.position.set(0, 0.0, -0.002);
    bg.renderOrder = 1000;
    this.group.add(bg);

    // View-mode buttons (top).
    this.lodButtons = o.lods.map((l, i) => {
      const w = 0.2, b = label(w, 0.045, l.label, { px: 44 });
      b.position.set((i - (o.lods.length - 1) / 2) * (w + 0.02), gridH / 2 + 0.05, 0);
      this.group.add(b);
      this.targets.push({ mesh: b, action: () => o.onSelectLod(l.id) });
      return { mesh: b, id: l.id, text: l.label };
    });

    // Thumbnail cells with a highlight frame behind each.
    this.cells = [];
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const x = -gridW / 2 + CELL_W / 2 + c * (CELL_W + GAP);
        const y = gridH / 2 - CELL_H / 2 - r * (CELL_H + GAP);
        const frame = new THREE.Mesh(new THREE.PlaneGeometry(CELL_W + 0.012, CELL_H + 0.012),
                                     overlayMaterial({ color: 0xffffff }));
        frame.position.set(x, y, -0.001);
        const img = new THREE.Mesh(new THREE.PlaneGeometry(CELL_W, CELL_H), overlayMaterial({ color: 0x333333 }));
        img.position.set(x, y, 0);
        this.group.add(frame, img);
        const cell = { img, frame, name: null };
        this.targets.push({ mesh: img, action: () => cell.name && cell.available && o.onSelectScene(cell.name), cell });
        this.cells.push(cell);
      }
    }

    // Page arrows and indicator (bottom).
    const y = -gridH / 2 - 0.05;
    this.prev = label(0.1, 0.045, "<", { px: 52 }); this.prev.position.set(-0.17, y, 0);
    this.next = label(0.1, 0.045, ">", { px: 52 }); this.next.position.set(0.17, y, 0);
    this.pageLabel = label(0.2, 0.045, "", { bg: "rgba(0,0,0,0)", px: 40 }); this.pageLabel.position.set(0, y, 0);
    this.group.add(this.prev, this.next, this.pageLabel);
    this.targets.push({ mesh: this.prev, action: () => this.setPage(this.page - 1) },
                      { mesh: this.next, action: () => this.setPage(this.page + 1) });
    // Fixed draw order by role. Without depth testing, equal renderOrders would be sorted by distance to the
    // eye, and the big background (centre often nearer than off-centre cells) would cover the thumbnails.
    for (const c of this.cells) { c.frame.renderOrder = 1001; c.img.renderOrder = 1002; }
    for (const m of [...this.lodButtons.map((b) => b.mesh), this.prev, this.next, this.pageLabel]) m.renderOrder = 1002;

    // A ray on each controller, shown while the gallery is open.
    this.rays = [0, 1].map((i) => {
      const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 0, -1)]);
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true }));
      line.scale.z = RAY_LEN; line.visible = false; line.renderOrder = 1003;
      renderer.xr.getController(i).add(line);
      return line;
    });
    this.raycaster = new THREE.Raycaster();
  }

  get visible() { return this.group.visible; }

  async open() {
    try { this.names = await this.o.listScenes(); } catch (e) { console.error("gallery: scene list failed", e); }
    try { this.modes = (await this.o.sceneModes?.()) ?? null; } catch (e) { this.modes = null; }
    const idx = this.names.indexOf(this.o.current());
    this.page = idx >= 0 ? Math.floor(idx / PER_PAGE) : 0;
    // Place in front of the head, upright, facing it.
    const cam = this.renderer.xr.getCamera();
    const pos = new THREE.Vector3(), quat = new THREE.Quaternion();
    cam.getWorldPosition(pos); cam.getWorldQuaternion(quat);
    const yaw = new THREE.Euler(0, 0, 0, "YXZ").setFromQuaternion(quat).y;
    const fwd = new THREE.Vector3(0, 0, -1).applyEuler(new THREE.Euler(0, yaw, 0, "YXZ"));
    this.group.position.copy(pos).addScaledVector(fwd, DIST);
    this.group.position.y -= 0.05;
    this.group.rotation.set(0, yaw, 0, "YXZ");
    this.group.visible = true;
    this.rays.forEach((r) => { r.visible = true; });
    this.refresh();
  }

  close() {
    this.group.visible = false;
    this.rays.forEach((r) => { r.visible = false; });
    this.hover = null;
  }

  toggle() { this.visible ? this.close() : this.open(); }

  setPage(p) {
    const pages = Math.max(1, Math.ceil(this.names.length / PER_PAGE));
    this.page = (p + pages) % pages;
    this.refresh();
  }

  /** Redraw thumbnails, frames, mode buttons and the page label. */
  refresh() {
    const pages = Math.max(1, Math.ceil(this.names.length / PER_PAGE));
    this.pageLabel.setText(`${this.page + 1} / ${pages}`);
    const cur = this.o.current(), lod = this.o.lod();
    this.cells.forEach((cell, i) => {
      const name = this.names[this.page * PER_PAGE + i] ?? null;
      cell.name = name;
      // Scenes without the current view mode are greyed out and can't be picked.
      cell.available = !name || !this.modes || (this.modes[name] ?? ["full"]).includes(lod);
      cell.img.visible = !!name;
      cell.frame.visible = !!name && (name === cur || (this.hover?.cell === cell && cell.available));
      cell.frame.material.color.set(this.hover?.cell === cell ? 0xffd000 : 0xffffff);
      if (!name) return;
      const tex = this._thumb(name);
      cell.img.material.map = tex;
      cell.img.material.color.set(!tex.image ? 0x333333 : cell.available ? 0xffffff : 0x3a3a3a);
      cell.img.material.needsUpdate = true;
    });
    this.lodButtons.forEach((b) => {
      const on = b.id === lod, hov = this.hover?.mesh === b.mesh;
      b.mesh.setText(b.text, { bg: on ? "rgba(255,208,0,0.95)" : hov ? "rgba(90,90,90,0.95)" : "rgba(40,40,40,0.9)",
                               fg: on ? "#000" : "#fff" });
    });
    for (const b of [this.prev, this.next]) {
      b.setText(b === this.prev ? "<" : ">", { bg: this.hover?.mesh === b ? "rgba(90,90,90,0.95)" : "rgba(40,40,40,0.9)" });
    }
  }

  _thumb(name) {
    let tex = this.thumbs.get(name);
    if (tex) return tex;
    tex = this.loader.load(this.o.thumbUrl(name), (t) => {
      t.colorSpace = THREE.SRGBColorSpace;
      // Cover the cell (crop the centre) whatever the photo's aspect ratio.
      const a = t.image.width / t.image.height, cellA = CELL_W / CELL_H;
      if (a > cellA) { t.repeat.set(cellA / a, 1); t.offset.set((1 - cellA / a) / 2, 0); }
      else { t.repeat.set(1, a / cellA); t.offset.set(0, (1 - a / cellA) / 2); }
      if (this.visible) this.refresh();
    });
    this.thumbs.set(name, tex);
    return tex;
  }

  /**
   * Per frame while open. controllers: [left, right] (three controller objects or null);
   * triggers: [left, right] trigger clicked this frame; sticksX: [left, right] thumbstick x.
   */
  update(controllers, triggers, sticksX) {
    if (!this.visible) return;
    // Hover: the nearest pickable item under either ray (right hand wins ties).
    let best = null, bestHand = -1;
    const meshes = this.targets.filter((t) => t.mesh.visible).map((t) => t.mesh);
    const m = new THREE.Matrix4();
    [0, 1].forEach((h) => {
      const c = controllers[h];
      if (!c) return;
      m.identity().extractRotation(c.matrixWorld);
      this.raycaster.ray.origin.setFromMatrixPosition(c.matrixWorld);
      this.raycaster.ray.direction.set(0, 0, -1).applyMatrix4(m);
      const hit = this.raycaster.intersectObjects(meshes, false)[0];
      if (hit && (!best || hit.distance <= best.distance)) { best = hit; bestHand = h; }
    });
    const target = best ? this.targets.find((t) => t.mesh === best.object) : null;
    if (target !== this.hover) { this.hover = target; this.refresh(); }
    if (target && [0, 1].some((h) => triggers[h] && controllers[h])) target.action();

    // Thumbstick left/right pages.
    const sx = sticksX.find((x) => Math.abs(x) > 0.6) ?? 0;
    if (sx && !this._stickMoved) { this._stickMoved = true; this.setPage(this.page + (sx > 0 ? 1 : -1)); }
    else if (sticksX.every((x) => Math.abs(x) < 0.2)) this._stickMoved = false;
  }
}
