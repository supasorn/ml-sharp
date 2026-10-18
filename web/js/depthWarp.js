/**
 * Depth warp: the depth controls of img3dviewer (z-scale, z-offset) for SHARP scenes.
 *
 * Every point moves along its own viewing ray from the capture camera (the content's origin), so the
 * image seen from there doesn't change, only the parallax does:
 *   d' = pivot + scale * (d - pivot) + offset,   p' = p * d' / d
 * with d the depth (z, or the distance from the centre for 360 panoramas). scale is the "3D intensity"
 * (1 = as captured; < 1 flatter, > 1 deeper) about the pivot, the near end of the scene (10th percentile
 * of its depths), as img3dviewer scales depth from its near plane: so nothing is pulled towards the
 * camera. offset moves everything nearer or farther (metres). d' never drops below 15 cm (or d if
 * nearer), so nothing crosses the camera's near plane.
 *
 * Applied in the vertex shader of meshes (MPI / MSI layers, hybrid meshes, depth layers) and as a Spark
 * objectModifier on splats (centres moved, sizes scaled by d'/d). Splats are regenerated when the values
 * change, so they update a little behind the meshes while adjusting.
 */
export function createDepthWarp(THREE, dyno, SplatMesh) {
  const u = { dwScale: { value: 1 }, dwOffset: { value: 0 }, dwPivot: { value: 1 }, dwRadial: { value: 0 } };
  const d = { scale: dyno.dynoFloat(1), offset: dyno.dynoFloat(0), pivot: dyno.dynoFloat(1), radial: dyno.dynoFloat(0) };
  const GLSL = `uniform float dwScale; uniform float dwOffset; uniform float dwPivot; uniform float dwRadial;
vec3 depthWarp(vec3 p) {
  float d = mix(p.z, length(p), dwRadial);
  float d2 = max(min(abs(d), 0.15), dwPivot + dwScale * (d - dwPivot) + dwOffset);
  return p * (d2 / max(abs(d), 1e-6));
}
`;
  const splats = new Set();

  function splatModifier() {
    const { dynoBlock, Gsplat, splitGsplat, combineGsplat, split, length, mix, min, max, mul, add, sub, div, abs, dynoConst } = dyno;
    return dynoBlock({ gsplat: Gsplat }, { gsplat: Gsplat }, ({ gsplat }) => {
      let { center, scales } = splitGsplat(gsplat).outputs;
      const { z } = split(center).outputs;
      const depth = mix(z, length(center), d.radial);
      const ad = abs(depth);
      const floor = min(ad, dynoConst("float", 0.15));
      const warped = max(floor, add(add(d.pivot, mul(d.scale, sub(depth, d.pivot))), d.offset));
      const f = div(warped, max(ad, dynoConst("float", 1e-6)));
      center = mul(center, f);
      scales = mul(scales, f);
      return { gsplat: combineGsplat({ gsplat, center, scales }) };
    });
  }

  function patchMaterial(mat) {
    if (!mat || mat.userData.depthWarp) return;
    mat.userData.depthWarp = true;
    if (mat.isShaderMaterial) {
      Object.assign(mat.uniforms, u);                 // the same uniform objects: one update moves everything
      mat.vertexShader = GLSL + mat.vertexShader.replace('vec4(position, 1.0)', 'vec4(depthWarp(position), 1.0)');
    } else {
      const prev = mat.onBeforeCompile;
      mat.onBeforeCompile = (shader, renderer) => {
        prev?.call(mat, shader, renderer);
        Object.assign(shader.uniforms, u);
        shader.vertexShader = GLSL + shader.vertexShader.replace('#include <begin_vertex>',
                                                                  'vec3 transformed = depthWarp(position);');
      };
      mat.customProgramCacheKey = () => 'depthWarp';
    }
    mat.needsUpdate = true;
  }

  return {
    /** Warp everything in `obj` (call once per loaded scene, before it is shown). */
    attach(obj) {
      walk(SplatMesh, obj, (c) => {
        c.objectModifier = splatModifier();
        c.updateGenerator();
        splats.add(c);
      }, (c) => {
        (Array.isArray(c.material) ? c.material : [c.material]).forEach(patchMaterial);
        c.frustumCulled = false;                      // bounds no longer match the warped geometry
      });
    },
    /** Forget splats of content that was removed. */
    detach(obj) { walk(SplatMesh, obj, (c) => splats.delete(c), () => {}); },
    /** { scale, offset, pivot, radial } (any subset). */
    set(v) {
      if (v.scale !== undefined) { u.dwScale.value = v.scale; d.scale.value = v.scale; }
      if (v.offset !== undefined) { u.dwOffset.value = v.offset; d.offset.value = v.offset; }
      if (v.pivot !== undefined) { u.dwPivot.value = v.pivot; d.pivot.value = v.pivot; }
      if (v.radial !== undefined) { u.dwRadial.value = v.radial ? 1 : 0; d.radial.value = v.radial ? 1 : 0; }
      for (const s of splats) s.updateVersion();      // regenerate the splats with the new values
    },
    get() { return { scale: u.dwScale.value, offset: u.dwOffset.value, pivot: u.dwPivot.value, radial: !!u.dwRadial.value }; },
  };
}

/** Visit splat meshes and plain meshes with vertex positions, not descending into a SplatMesh (its
 *  internal objects are Spark's own). */
function walk(SplatMesh, obj, onSplats, onMesh) {
  if (obj instanceof SplatMesh) { onSplats(obj); return; }
  if (obj.isMesh && obj.geometry?.attributes?.position) onMesh(obj);
  for (const c of obj.children) walk(SplatMesh, c, onSplats, onMesh);
}

/**
 * The near end of the content (10th percentile of z, or of the distance for panoramas) in its own
 * coordinates: the pivot of the depth scale. Samples mesh vertices and splat centres (at most ~50k each).
 */
export async function estimatePivot(THREE, SplatMesh, obj, radial) {
  const ds = [];
  const take = (x, y, z) => { const v = radial ? Math.hypot(x, y, z) : z; if (v > 0.05) ds.push(v); };
  const meshes = [], splats = [];
  walk(SplatMesh, obj, (c) => splats.push(c), (c) => meshes.push(c));
  for (const m of meshes) {
    const pos = m.geometry.attributes.position;
    const step = Math.max(1, Math.floor(pos.count / 50000));
    for (let i = 0; i < pos.count; i += step) take(pos.getX(i), pos.getY(i), pos.getZ(i));
  }
  for (const s of splats) {
    await s.initialized;
    const step = Math.max(1, Math.ceil((s.numSplats || 0) / 50000));
    s.forEachSplat((i, c, scales, q, opacity) => { if (i % step === 0 && opacity > 0.5) take(c.x, c.y, c.z); });
  }
  if (ds.length < 10) return 1;
  ds.sort((a, b) => a - b);
  return ds[Math.floor(ds.length * 0.1)];
}
