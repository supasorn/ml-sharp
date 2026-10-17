/**
 * 360 panorama support for the viewer (web/index.html): desktop look-around controls and the material
 * of multi-sphere images (scripts/pano_msi.py). Background: docs/pano-360.md.
 *
 * Desktop controls (PanoLook): drag = look around (no orbit), right-drag / shift-drag or two fingers =
 * slide sideways and up, WASD / arrows = move, Q / E = down / up, wheel or pinch = zoom (FOV), R = back
 * to the centre. Movement stays within `maxMove` metres of the capture point (parallax, not walking).
 */
import * as THREE from "three";

export class PanoLook {
  constructor(camera, dom, { fov = 75, maxMove = 0.5 } = {}) {
    this.camera = camera;
    this.dom = dom;
    this.yaw = 0;
    this.pitch = 0;
    this.pos = new THREE.Vector3();
    this.fov = fov;
    this.maxMove = maxMove;
    this.wiggle = false;
    this.keys = new Set();
    camera.rotation.order = "YXZ";
    this._pointers = new Map();
    this._pinch = null;
    this._bind();
  }

  reset() { this.pos.set(0, 0, 0); }

  _clamp() {
    const r = this.pos.length();
    if (r > this.maxMove) this.pos.multiplyScalar(this.maxMove / r);
  }

  _axes() {
    return {
      fwd: new THREE.Vector3(-Math.sin(this.yaw), 0, -Math.cos(this.yaw)),
      right: new THREE.Vector3(Math.cos(this.yaw), 0, -Math.sin(this.yaw)),
      up: new THREE.Vector3(0, 1, 0),
    };
  }

  _slide(dx, dy) {
    const { right, up } = this._axes();
    this.pos.addScaledVector(right, -dx * 0.002).addScaledVector(up, dy * 0.002);
    this._clamp();
  }

  _bind() {
    const el = this.dom;
    el.style.touchAction = "none";
    el.style.cursor = "grab";
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    el.addEventListener("pointerdown", (e) => {
      el.setPointerCapture(e.pointerId);
      this._pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, slide: e.button === 2 || e.shiftKey });
      el.style.cursor = "grabbing";
      if (this._pointers.size === 2) this._pinch = null;
    });
    const end = (e) => {
      this._pointers.delete(e.pointerId);
      if (!this._pointers.size) el.style.cursor = "grab";
      this._pinch = null;
    };
    el.addEventListener("pointerup", end);
    el.addEventListener("pointercancel", end);
    el.addEventListener("pointermove", (e) => {
      const p = this._pointers.get(e.pointerId);
      if (!p) return;
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;
      if (this._pointers.size === 2) {
        const [a, b] = [...this._pointers.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (this._pinch) this.fov = THREE.MathUtils.clamp(this.fov * this._pinch / d, 30, 110);
        this._pinch = d;
        this._slide(dx / 2, dy / 2);
        return;
      }
      if (p.slide) { this._slide(dx, dy); return; }
      // Grab-the-world look: radians per pixel follow the vertical FOV.
      const k = THREE.MathUtils.degToRad(this.fov) / el.clientHeight;
      this.yaw += dx * k;
      this.pitch = THREE.MathUtils.clamp(this.pitch + dy * k, -Math.PI / 2 + 0.01, Math.PI / 2 - 0.01);
    });
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.fov = THREE.MathUtils.clamp(this.fov * Math.exp(e.deltaY * 0.001), 30, 110);
    }, { passive: false });
    addEventListener("keydown", (e) => {
      if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
      this.keys.add(e.code);
      if (e.code === "KeyR") this.reset();
    });
    addEventListener("keyup", (e) => this.keys.delete(e.code));
    addEventListener("blur", () => this.keys.clear());
  }

  /** Per frame (desktop only): keyboard movement, wiggle, camera pose and FOV. */
  update(dt, now) {
    const { fwd, right, up } = this._axes();
    const v = new THREE.Vector3();
    const k = this.keys;
    if (k.has("KeyW") || k.has("ArrowUp")) v.add(fwd);
    if (k.has("KeyS") || k.has("ArrowDown")) v.sub(fwd);
    if (k.has("KeyD") || k.has("ArrowRight")) v.add(right);
    if (k.has("KeyA") || k.has("ArrowLeft")) v.sub(right);
    if (k.has("KeyE")) v.add(up);
    if (k.has("KeyQ")) v.sub(up);
    if (v.lengthSq()) { this.pos.addScaledVector(v.normalize(), 0.6 * dt); this._clamp(); }
    this.camera.rotation.set(this.pitch, this.yaw, 0);
    this.camera.position.copy(this.pos);
    if (this.wiggle) {
      const t = now / 1000, r = Math.min(0.15, this.maxMove);
      this.camera.position.addScaledVector(right, r * Math.sin(t * 1.6)).addScaledVector(up, 0.4 * r * Math.sin(t * 3.2));
    }
    if (this.camera.fov !== this.fov) { this.camera.fov = this.fov; this.camera.updateProjectionMatrix(); }
  }
}

/**
 * Material of an MSI atlas page: premultiplied display-space RGBA sampled raw (an sRGB decode would
 * brighten soft edges), blended (1, 1 - srcAlpha), no depth test (the spheres are drawn far to near by
 * renderOrder). KTX2 pages carry 3 box mips: trilinear with `lodBias` (docs/quest-xr-findings.md §4b.14).
 */
export function msiMaterial(map, lodBias) {
  const hasMips = !!(map.mipmaps && map.mipmaps.length > 1);
  map.colorSpace = THREE.NoColorSpace;
  map.generateMipmaps = false;
  map.minFilter = hasMips ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter;
  map.needsUpdate = true;
  return new THREE.ShaderMaterial({
    uniforms: { map: { value: map }, lodBias: { value: hasMips ? lodBias : 0.0 } },
    vertexShader: "varying vec2 vUv; void main() { vUv = uv; " +
                  "gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
    fragmentShader: "uniform sampler2D map; uniform float lodBias; varying vec2 vUv; " +
                    "void main() { gl_FragColor = texture2D(map, vUv, lodBias); }",
    side: THREE.DoubleSide, transparent: true, depthWrite: false, depthTest: false,
    blending: THREE.CustomBlending, blendEquation: THREE.AddEquation,
    blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
  });
}

/** MSI scene from a loaded glTF: page materials, far spheres first (surface_<k>, larger k = farther). */
export function setupMsi(gltfScene, lodBias) {
  let cells = 0;
  gltfScene.traverse((child) => {
    if (!child.isMesh) return;
    const k = /surface_(\d+)/.exec(child.name + " " + (child.parent?.name ?? ""));
    const map = child.material.map;
    child.material.dispose();
    child.material = msiMaterial(map, lodBias);
    child.renderOrder = -Number(k ? k[1] : 0);
    child.frustumCulled = false;
    cells += child.geometry.index ? child.geometry.index.count / 6 : 0;
  });
  return cells;
}
