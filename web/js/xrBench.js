/**
 * Self-running XR benchmarks (?bench=<name>): run all variants in one XR
 * session, 10 s per phase (first 2 s skipped), log each phase's average FPS,
 * then end the session.
 *
 * bench=corners -- non-instanced splats (web/js/splatFlat.js) as quads
 * (4 vertices) vs single triangles (3 vertices). Phases: quad, tri, quad, tri.
 * Both should look identical.
 */

/**
 * Returns { step(now) } to call every frame (before rendering), or null if
 * the benchmark can't run. flatSplats is the viewer's non-instanced renderer;
 * report(fields) sends a log entry.
 */
export function setupBench(name, { renderer, flatSplats, report }) {
  if (name !== "corners" || !flatSplats) return null;
  const restoreCorners = flatSplats.corners;
  const phases = ["quad", "tri", "quad", "tri"];
  const st = { phase: -1, presentingSince: 0, start: 0, frames: 0, measureStart: 0 };

  function step(now) {
    if (!renderer.xr.isPresenting) {
      if (st.phase >= 0) flatSplats.setCorners(restoreCorners);
      st.phase = -1;
      st.presentingSince = 0;
      return;
    }
    if (st.phase < 0) {
      // Let the session settle (and the scene auto-centre) for 1 s first.
      if (!st.presentingSince) st.presentingSince = now;
      if (now - st.presentingSince < 1000) return;
      Object.assign(st, { phase: 0, start: now, frames: 0 });
    }
    flatSplats.setCorners(phases[st.phase] === "tri" ? 3 : 4);
    const t = now - st.start;
    if (t >= 2000) {
      if (st.frames === 0) st.measureStart = now;
      st.frames++;
    }
    if (t < 10000) return;
    const fps = (st.frames - 1) * 1000 / (now - st.measureStart);
    report({ test: `bench-${name}-${phases[st.phase]}`, phase: st.phase, fps: +fps.toFixed(1) });
    st.phase++;
    if (st.phase >= phases.length) {
      st.phase = -1;
      flatSplats.setCorners(restoreCorners);
      renderer.xr.getSession()?.end();
      return;
    }
    Object.assign(st, { start: now, frames: 0 });
  }
  return { step };
}
