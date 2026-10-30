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
 *
 * bench=ldi -- MPI-only vs its layered-depth-image bake (mpivar=ldi), compressed textures, alternating
 * (A B A B) per scene of ?benchscenes (default: the four test scenes), same timing as bench=views.
 *
 * bench=profile -- where a scene's frame time goes (default scene: the one opened). Phases on one scene: Original
 * splats; Mesh hybrid (all regions, ?lod=meshall) whole, mesh only, splats only, splats only with the splat size
 * capped at 32 px (fill-bound if this is much faster), whole with maximum foveation (fill again); Photo detail
 * (MPI) for reference; Mesh hybrid whole again (drift). Same timing as bench=views; each phase also logs frame-time
 * percentiles, GPU time (EXT_disjoint_timer_query_webgl2, if the browser has it), triangles and draw calls per
 * frame, CPU time and Spark sorts / rebuilds.
 *
 * bench=backdrop -- Mesh hybrid (meshall) vs Mesh hybrid + backdrop (meshback, its big background splats meshed),
 * alternating twice, then the backdrop model's mesh only and splats only; same logging as bench=profile.
 *
 * bench=bgmpi -- Mesh hybrid (meshall) vs Mesh + background MPI (meshbg: meshall's mesh, its background splats as a
 * sparse MPI, its foreground splats), alternating twice, then meshbg without splats and splats only; Photo detail for
 * reference. Same logging as bench=profile.
 *
 * bench=meshmpi -- Mesh-MPI (meshall's mesh + all its leftover splats as a sparse MPI, no splats) vs Mesh hybrid
 * (meshall) and Mesh + background MPI (meshbg), Mesh-MPI twice; Photo detail for reference. Same logging as bench=profile.
 *
 * bench=vars -- MPI-only variant bakes of ?benchvars=a,b,c (compressed; "mpionly" = the plain MPI-only bake),
 * run twice in turn (a b c a b c) on ?benchscenes (default: the scene opened), same timing as bench=views.
 */

const VIEW_PHASES = [
  { lod: "full", tex: "png" }, { lod: "hybrid", tex: "png" },
  { lod: "mpi", tex: "png" }, { lod: "mpi", tex: "ktx2" },
  { lod: "mpionly", tex: "png" }, { lod: "mpionly", tex: "ktx2" },
];
const PROFILE_PHASES = [
  { lod: "full" },
  { lod: "meshall" }, { lod: "meshall", show: "mesh" }, { lod: "meshall", show: "splats" },
  { lod: "meshall", show: "splats", maxPx: 32 }, { lod: "meshall", foveation: 1 },
  { lod: "photo", tex: "ktx2" }, { lod: "meshall" },
];
const BACKDROP_PHASES = [
  { lod: "meshall" }, { lod: "meshback" }, { lod: "meshall" }, { lod: "meshback" },
  { lod: "meshback", show: "mesh" }, { lod: "meshback", show: "splats" },
];
const BGMPI_PHASES = [
  { lod: "meshall" }, { lod: "meshbg", tex: "ktx2" }, { lod: "meshall" }, { lod: "meshbg", tex: "ktx2" },
  { lod: "meshbg", tex: "ktx2", show: "mesh" }, { lod: "meshbg", tex: "ktx2", show: "splats" },
  { lod: "photo", tex: "ktx2" },
];
const MESHMPI_PHASES = [
  { lod: "meshall" }, { lod: "meshmpi", tex: "ktx2" }, { lod: "meshbg", tex: "ktx2" }, { lod: "meshmpi", tex: "ktx2" },
  { lod: "photo", tex: "ktx2" },
];
const DEFAULT_SCENES = ["20260914_113455_56657", "20200927_161631"];
const LDI_PHASES = [
  { lod: "mpionly", tex: "ktx2" }, { lod: "mpionly", tex: "ktx2", mpivar: "ldi" },
  { lod: "mpionly", tex: "ktx2" }, { lod: "mpionly", tex: "ktx2", mpivar: "ldi" },
];
const LDI_SCENES = ["20250810_134957_16671", "20260914_113455_56657", "me_small", "20200927_161631"];

/**
 * Returns { step(now) } to call every frame (before rendering), or null if
 * the benchmark can't run. flatSplats is the viewer's non-instanced renderer;
 * report(fields) sends a log entry. For bench=views: loadView({scene, lod, tex})
 * returns a promise resolved once that content is shown; onPhase(text) shows
 * a label in the headset; scenes is the list of scenes.
 */
export function setupBench(name, { renderer, flatSplats, report, loadView, onPhase, scenes, currentScene, stats }) {
  if (name === "profile" && loadView) return viewsBench(renderer, report, loadView, onPhase,
                                                        scenes && scenes.length ? scenes : [currentScene],
                                                        PROFILE_PHASES, "profile", stats);
  if (name === "backdrop" && loadView) return viewsBench(renderer, report, loadView, onPhase,
                                                         scenes && scenes.length ? scenes : [currentScene],
                                                         BACKDROP_PHASES, "backdrop", stats);
  if (name === "bgmpi" && loadView) return viewsBench(renderer, report, loadView, onPhase,
                                                      scenes && scenes.length ? scenes : [currentScene],
                                                      BGMPI_PHASES, "bgmpi", stats);
  if (name === "meshmpi" && loadView) return viewsBench(renderer, report, loadView, onPhase,
                                                        scenes && scenes.length ? scenes : [currentScene],
                                                        MESHMPI_PHASES, "meshmpi", stats);
  if (name === "views" && loadView) return viewsBench(renderer, report, loadView, onPhase, scenes);
  if (name === "ldi" && loadView) return viewsBench(renderer, report, loadView, onPhase,
                                                    scenes && scenes.length ? scenes : LDI_SCENES, LDI_PHASES, "ldi");
  if (name === "vars" && loadView) {
    const vars = (new URLSearchParams(location.search).get("benchvars") || "mpionly").split(",").filter(Boolean)
      .map((v) => v.replace(/[^a-z0-9]/gi, ""));
    const one = vars.map((v) => ({ lod: "mpionly", tex: "ktx2", mpivar: v === "mpionly" ? "" : v }));
    return viewsBench(renderer, report, loadView, onPhase, scenes && scenes.length ? scenes : [currentScene],
                      [...one, ...one], "vars");
  }
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

function viewsBench(renderer, report, loadView, onPhase, scenes, phases = VIEW_PHASES, test = "views", stats = null) {
  const list = (scenes && scenes.length ? scenes : DEFAULT_SCENES)
    .flatMap((scene) => phases.map((p) => ({ scene, ...p })));
  const label = (p) => `${{ full: "Original splats", hybrid: "Hybrid", mpi: "Hybrid-MPI", mpionly: "MPI-only",
                            meshall: "Mesh hybrid", meshback: "Mesh + backdrop", meshbg: "Mesh + bg MPI", meshmpi: "Mesh-MPI", photo: "Photo detail" }[p.lod] ?? p.lod}`
    + (p.mpivar ? ` ${p.mpivar}` : "") + (p.tex === "ktx2" ? " · compressed" : "")
    + (p.show ? ` · ${p.show} only` : "") + (p.maxPx ? ` · max ${p.maxPx} px` : "")
    + (p.foveation !== undefined ? ` · foveation ${p.foveation}` : "");
  const pct = (a, q) => a.length ? +a[Math.min(a.length - 1, Math.floor(q * a.length))].toFixed(2) : null;
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
      report({ test: `bench-${test}`, phase: i, error: String(e), scene: p.scene, lod: p.lod, tex: p.tex,
               mpivar: p.mpivar || null });
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
      Object.assign(st, { state: "measure", t0: now, frames: 0, measureStart: now, last: now, dts: [] });
      stats?.(true);   // reset the viewer's counters
      return;
    }
    if (st.state === "measure") {
      st.frames++;
      st.dts.push(now - st.last);
      st.last = now;
      if (now - st.t0 < 8000) return;
      const fps = (st.frames - 1) * 1000 / (now - st.measureStart);
      const p = list[st.i];
      const dts = st.dts.slice(1).sort((a, b) => a - b);
      report({ test: `bench-${test}`, phase: st.i, fps: +fps.toFixed(1), scene: p.scene, lod: p.lod, tex: p.tex,
               mpivar: p.mpivar || null, show: p.show || "both", maxPx: p.maxPx ?? null,
               foveation: p.foveation ?? null, label: label(p),
               dtMs: { p10: pct(dts, 0.1), p50: pct(dts, 0.5), p90: pct(dts, 0.9), p99: pct(dts, 0.99) },
               ...(stats?.(false) ?? {}) });
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
