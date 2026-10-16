/**
 * Draw Spark's splats without instancing.
 *
 * Spark draws every splat as an instance of one 4-vertex quad. On Quest 3's
 * GPU that is very inefficient: vertices are shaded in wide batches, and
 * vertices from different instances apparently don't share a batch, so each
 * splat's 4 vertices occupy a mostly empty batch. Measured: skipping the
 * splat maths (+6%), the data reads (+1%) or using 3 vertices instead of 4
 * (+0%) barely mattered with instancing, while drawing the same quads
 * non-instanced was +39% (34 -> 48 fps) with an identical image.
 *
 * Here all splats' corners are spelled out in one vertex buffer, in Spark's
 * sorted order, and drawn in a single non-instanced call. The vertex shader
 * is Spark's, with the corner taken from gl_VertexID instead of the instanced
 * `position` attribute -- same maths, order and pixels.
 *
 * corners = 4: Spark's quad (corners at +-1, two triangles).
 * corners = 3: one triangle whose inscribed circle is the unit circle
 *   (vertices at distance 2): after Spark's scaling it still covers the
 *   ellipse it cuts at maxStdDev, so the image is unchanged, with 3/4 of the
 *   vertices and ~30% more pixels. Measured +16% over non-instanced quads
 *   (48 -> 55 fps); the viewer's default.
 *
 * The expanded buffers are rebuilt when Spark finishes a new sort (rare with
 * the fixed sort pose) or the corner count changes. Call sync() every frame
 * before rendering.
 */
function patchVertexShader(vs) {
  if (!vs.includes("void main() {") || !/\bposition\b/.test(vs)) return null;
  return vs
    .replace(/\bposition\b/g, "splatCorner")
    .replace("void main() {", `uniform int splatCorners;
vec3 splatCorner;
void main() {
    // Corner from the vertex id (splatCorners vertices per splat).
    if (splatCorners == 3) {
        int c = gl_VertexID % 3;
        splatCorner = c == 0 ? vec3(0.0, 2.0, 0.0) : vec3(c == 1 ? -1.7320508 : 1.7320508, -1.0, 0.0);
    } else {
        int c = gl_VertexID & 3;  // Spark's quad order: (-1,-1) (1,-1) (1,1) (-1,1)
        splatCorner = vec3((c == 1 || c == 2) ? 1.0 : -1.0, c >= 2 ? 1.0 : -1.0, 0.0);
    }`);
}

/** Returns { mesh, set(on), setCorners(3|4), corners, sync() } or null if Spark's shader isn't recognised. */
export function createFlatSplats(THREE, spark, { corners = 4 } = {}) {
  const vertexShader = patchVertexShader(spark.material.vertexShader);
  if (!vertexShader) {
    console.warn("splatFlat: Spark vertex shader not recognised; keeping Spark's instanced splats.");
    return null;
  }
  const m = spark.material;
  const uniforms = { ...spark.uniforms, splatCorners: { value: corners } };  // shares Spark's uniform objects
  const material = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader,
    fragmentShader: m.fragmentShader,
    uniforms,
    transparent: m.transparent,
    depthTest: m.depthTest,
    depthWrite: m.depthWrite,
    premultipliedAlpha: m.premultipliedAlpha,
    side: m.side,
  });
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
  mesh.frustumCulled = false;
  mesh.visible = false;
  // Spark sets its uniforms (per-eye transforms, renderSize, ...) here.
  mesh.onBeforeRender = (r, scene, camera) => spark.onBeforeRender(r, scene, camera);

  let enabled = false;
  let source = null, sourceVersion = -1, builtCorners = 0, capacity = 0;
  let splatIndexAttr = null;

  function rebuild(ordering, n) {
    const k = corners;
    if (n > capacity || k !== builtCorners) {
      capacity = Math.max(capacity, Math.ceil(n * 1.1));
      const idx = new Uint32Array(capacity * (k === 3 ? 3 : 6));
      if (k === 3) {
        for (let i = 0; i < capacity * 3; i++) idx[i] = i;
      } else {
        for (let i = 0, v = 0, o = 0; i < capacity; i++, v += 4, o += 6) {
          idx[o] = v; idx[o + 1] = v + 1; idx[o + 2] = v + 2;
          idx[o + 3] = v; idx[o + 4] = v + 2; idx[o + 5] = v + 3;
        }
      }
      mesh.geometry.dispose();
      const geo = new THREE.BufferGeometry();
      geo.setIndex(new THREE.BufferAttribute(idx, 1));
      splatIndexAttr = new THREE.BufferAttribute(new Uint32Array(capacity * k), 1);
      splatIndexAttr.setUsage(THREE.DynamicDrawUsage);
      geo.setAttribute("splatIndex", splatIndexAttr);
      mesh.geometry = geo;
      builtCorners = k;
    }
    const out = splatIndexAttr.array;
    for (let i = 0, v = 0; i < n; i++, v += k) {
      const s = ordering[i];
      for (let c = 0; c < k; c++) out[v + c] = s;
    }
    splatIndexAttr.clearUpdateRanges();
    splatIndexAttr.addUpdateRange(0, n * k);
    splatIndexAttr.needsUpdate = true;
  }

  function sync() {
    spark.visible = !enabled;
    mesh.visible = enabled;
    if (!enabled) return;
    const g = spark.geometry;
    const ordering = g && g.ordering;
    const n = (g && g.instanceCount) | 0;
    if (!ordering || !n) {
      mesh.geometry.setDrawRange(0, 0);
      return;
    }
    const attr = g.getAttribute("splatIndex");
    const version = attr ? attr.version : -1;
    if (ordering !== source || version !== sourceVersion || corners !== builtCorners) {
      rebuild(ordering, n);
      source = ordering;
      sourceVersion = version;
    }
    uniforms.splatCorners.value = corners;
    mesh.geometry.setDrawRange(0, n * (corners === 3 ? 3 : 6));
  }

  return {
    mesh,
    set(on) { enabled = !!on; },
    setCorners(k) { corners = k === 3 ? 3 : 4; },
    get corners() { return corners; },
    sync,
  };
}
