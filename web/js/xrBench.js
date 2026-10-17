/**
 * Self-running XR benchmarks (?bench=<name>): run all variants in one XR
 * session, log each phase's average FPS, then end the session.
 *
 * bench=corners -- non-instanced splats (web/js/splatFlat.js) as quads
 * (4 vertices) vs single triangles (3 vertices). Phases: quad, tri, quad, tri,
 * 10 s each (first 2 s skipped). Both should look identical.
 *
 * bench=views -- view modes and texture variants, each phase a (scene, lod,
 * tex) loaded in place: Original splats, Hybrid, Hybrid-MPI, Hybrid-MPI with
 * compressed textures, MPI-only, MPI-only compressed, for each scene of
 * ?benchscenes=a,b (default: 20260914_113455_56657, 20200927_161631).
 * Per phase: load, re-centre the scene in front, 2 s settle, 8 s measured.
 * Hold still, facing the scene, for comparable numbers.
 */

const VIEW_PHASES = [
  { lod: "full", tex: "png" }, { lod: "hybrid", tex: "png" },
  { lod: "mpi", tex: "png" }, { lod: "mpi", tex: "ktx2" },
  { lod: "mpionly", tex: "png" }, { lod: "mpionly", tex: "ktx2" },
];
const DEFAULT_SCENES = ["20260914_113455_56657", "20200927_161631"];

/**
 * Returns { step(now) } to call every frame (before rendering), or null if
 * the benchmark can't run. flatSplats is the viewer's non-instanced renderer;
 * report(fields) sends a log entry. For bench=views: loadView({scene, lod, tex})
 * returns a promise resolved once that content is shown; onPhase(text) shows
 * a label in the headset; scenes is the list of scenes.
 */
export function setupBench(name, { renderer, flatSplats, report, loadView, onPhase, scenes }) {
  if (name === "views" && loadView) return viewsBench(renderer, report, loadView, onPhase, scenes);
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

function viewsBench(renderer, report, loadView, onPhase, scenes) {
  const list = (scenes && scenes.length ? scenes : DEFAULT_SCENES)
    .flatMap((scene) => VIEW_PHASES.map((p) => ({ scene, ...p })));
  const label = (p) => `${{ full: "Original splats", hybrid: "Hybrid", mpi: "Hybrid-MPI", mpionly: "MPI-only" }[p.lod]}`
    + (p.tex === "ktx2" ? " · compressed" : "");
  // state: idle -> loading -> settle (2 s) -> measure (8 s) -> next phase
  const st = { i: -1, state: "idle", presentingSince: 0, t0: 0, frames: 0, measureStart: 0 };

  function start(i, now) {
    st.i = i;
    st.state = "loading";
    const p = list[i];
    onPhase?.(`Bench ${i + 1}/${list.length} · ${label(p)} · loading…`, 600);
    loadView(p).then(() => {
      st.state = "settle";
      st.t0 = performance.now();
      onPhase?.(`Bench ${i + 1}/${list.length} · ${label(p)}`, 10);
    }).catch((e) => {
      console.error("bench: load failed", p, e);
      report({ test: "bench-views", phase: i, error: String(e), scene: p.scene, lod: p.lod, tex: p.tex });
      st.state = "next";
    });
  }

  function step(now) {
    if (!renderer.xr.isPresenting) {
      st.i = -1; st.state = "idle"; st.presentingSince = 0;
      return;
    }
    if (st.state === "idle") {
      if (!st.presentingSince) st.presentingSince = now;
      if (now - st.presentingSince < 1500) return;
      start(0, now);
      return;
    }
    if (st.state === "loading") return;
    if (st.state === "settle") {
      if (now - st.t0 < 2000) return;
      Object.assign(st, { state: "measure", t0: now, frames: 0, measureStart: now });
      return;
    }
    if (st.state === "measure") {
      st.frames++;
      if (now - st.t0 < 8000) return;
      const fps = (st.frames - 1) * 1000 / (now - st.measureStart);
      const p = list[st.i];
      report({ test: "bench-views", phase: st.i, fps: +fps.toFixed(1), scene: p.scene, lod: p.lod, tex: p.tex });
      st.state = "next";
    }
    if (st.state === "next") {
      if (st.i + 1 >= list.length) {
        st.state = "done";
        onPhase?.("Benchmark done", 3);
        renderer.xr.getSession()?.end();
        return;
      }
      start(st.i + 1, now);
    }
  }
  return { step };
}
