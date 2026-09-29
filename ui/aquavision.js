// AquaVision front end. Streamlit calls the default export again every time Python sends new
// data, on the same root, so the app is created once and later calls only feed it updates.

const ICON = {
  upload: '<path d="M12 15V3"/><path d="m7 8 5-5 5 5"/><path d="M20 15v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-4"/>',
  arrow: '<path d="M5 12h14"/><path d="m13 6 6 6-6 6"/>',
  check: '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
  triangle: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
  octagon: '<path d="M12 16h.01"/><path d="M12 8v4"/><path d="M15.31 2a2 2 0 0 1 1.42.59l4.68 4.68A2 2 0 0 1 22 8.69v6.62a2 2 0 0 1-.59 1.42l-4.68 4.68a2 2 0 0 1-1.42.59H8.69a2 2 0 0 1-1.42-.59l-4.68-4.68A2 2 0 0 1 2 15.31V8.69a2 2 0 0 1 .59-1.42l4.68-4.68A2 2 0 0 1 8.69 2z"/>',
  camera: '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/>',
  swap: '<path d="m16 3 4 4-4 4"/><path d="M20 7H4"/><path d="m8 21-4-4 4-4"/><path d="M4 17h16"/>',
};
const icon = (name) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICON[name]}</svg>`;
const LOGO =
  '<svg class="brand-mark" viewBox="0 0 36 36" fill="none" aria-hidden="true"><circle cx="18" cy="18" r="16" stroke="currentColor" stroke-width="2"/><circle cx="18" cy="18" r="9" stroke="currentColor" stroke-width="2" opacity=".55"/><circle cx="21" cy="15" r="3.2" fill="currentColor"/><path d="M18 0v5M18 31v5M0 18h5M31 18h5" stroke="currentColor" stroke-width="2"/></svg>';

const SEV = {
  none: { label: "Healthy", rank: "Lowest of the three levels", band: "No disease seen", icon: "check", pos: 1 / 6 },
  moderate: { label: "Moderate", rank: "Middle of the three levels", band: "Watch closely", icon: "triangle", pos: 3 / 6 },
  critical: { label: "Critical", rank: "Highest of the three levels", band: "Act today", icon: "octagon", pos: 5 / 6 },
};
const MIN_SCAN_MS = 1300; // let the scan read as a scan even when inference takes 10 ms
const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const el = (html) => {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Resize on the phone before upload: a 12 MP photo becomes ~300 KB, and the model only needs 224–300 px.
async function fileToJpeg(file, max = 1600) {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.decoding = "async";
    img.src = url;
    await img.decode();
    const s = Math.min(1, max / Math.max(img.naturalWidth, img.naturalHeight));
    const c = document.createElement("canvas");
    c.width = Math.max(1, Math.round(img.naturalWidth * s));
    c.height = Math.max(1, Math.round(img.naturalHeight * s));
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#fff"; // transparent PNGs would otherwise turn black in JPEG
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.92);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function createApp(root) {
  const app = {
    trigger: null,
    config: null,
    species: null,
    reqId: null,
    startedAt: 0,
    renderedId: null,
    preview: null,
  };
  const $ = (sel) => root.querySelector(sel);
  const $$ = (sel) => [...root.querySelectorAll(sel)];

  function build(config) {
    app.config = config;
    app.species = config.species[0].key;
    const sp = config.species;
    const shell = el(`<div class="av">
      <div class="wrap">
        <header class="nav">
          <div class="brand">${LOGO}<span class="brand-name">AquaVision</span><span class="brand-by">by AquaManage</span></div>
          <nav class="nav-right">
            <button class="nav-link" data-go="how">How it works</button>
            <button class="nav-link" data-go="cam">The model</button>
            <span class="status mono"><span class="dot"></span><span>Models ready</span></span>
          </nav>
        </header>

        <section class="intro">
          <div>
            <p class="eyebrow mono fade-in" style="--d:80ms">Disease check · shrimp &amp; fish</p>
            <h1 class="h1" aria-label="Spot the disease before it spreads">
              <span class="line" aria-hidden="true"><span style="--i:0">Spot the disease</span></span>
              <span class="line" aria-hidden="true"><span style="--i:1" class="accent">before it spreads.</span></span>
            </h1>
          </div>
          <div class="intro-side fade-in" style="--d:520ms">
            <p class="lede">Photograph one animal. AquaVision names the likely condition, rates how serious it is,
            and tells you what to do next — in about a second.</p>
            <div class="stats">
              ${config.stats.map((s) => `<div><div class="stat-v">${esc(s.value)}</div><div class="stat-l mono">${esc(s.label)}</div></div>`).join("")}
            </div>
          </div>
        </section>

        <div class="stage-head mono fade-in" style="--d:700ms"><span>01 — Choose what you're checking</span><span class="stage-hint">Tap the other plate to switch</span></div>
        <section class="stage fade-in" style="--d:760ms" aria-label="Species">
          ${sp.map((s, n) => panelHtml(s, n)).join("")}
        </section>

        <section class="report" aria-live="polite"><div class="report-clip"><div class="sheet"></div></div></section>

        <section class="section" data-anchor="how">
          <p class="eyebrow mono reveal">How it works</p>
          <h2 class="h2 reveal">Three steps at the pond edge.</h2>
          <div class="steps">
            <div class="step" style="--i:0"><span class="step-n mono">01</span><h3>Photograph</h3><p>Net one animal. Daylight, no flash, and let it fill most of the frame. Gills, shell or skin in focus.</p></div>
            <div class="step" style="--i:1"><span class="step-n mono">02</span><h3>Analyse</h3><p>An EfficientNet model trained on thousands of labelled pond photos checks it on the server in milliseconds.</p></div>
            <div class="step" style="--i:2"><span class="step-n mono">03</span><h3>Act</h3><p>You get the likely condition, one of three severity bands, and a plain next step for today.</p></div>
          </div>
        </section>

        <section class="cam" data-anchor="cam">
          <div class="reveal">
            <p class="eyebrow mono">Under the hood</p>
            <h2 class="h2">It shows where it looked.</h2>
            <p>Grad-CAM heat maps mark the part of the photo that pushed the model to its answer — gills for black gill,
            shell spots for white spot. If the heat sits on the background, retake the photo.</p>
            <ul class="spec-list">
              ${sp.map((s) => `<li><span>${esc(s.name)} model</span><span>${esc(s.arch)} · ${esc(s.input)} px · ${esc(s.accuracy)} val. accuracy</span></li>`).join("")}
              <li><span>Runtime</span><span>ONNX Runtime on CPU</span></li>
            </ul>
          </div>
          <figure class="reveal">
            <div class="cam-fig"><img src="${esc(config.gradcam)}" alt="Grad-CAM heat maps over sample fish photos, one per class" loading="lazy"></div>
            <figcaption class="cam-cap mono">Grad-CAM · fish model · one example per class</figcaption>
          </figure>
        </section>

        <footer class="footer">
          <div><b>AquaVision</b> · part of AquaManage, the pond notebook that reads itself.<br>
          A screening tool, not a veterinary diagnosis.</div>
          <div>Built by <b>Mahesh Penubothu</b> · VIT-AP University<br>
          ${config.credits.map((c) => `${esc(c.what)}: <a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.who)}</a>, ${esc(c.license)}`).join("<br>")}</div>
        </footer>
      </div>
      <input class="file" type="file" accept="image/*" hidden>
      <p class="sr" data-live aria-live="polite"></p>
    </div>`);
    root.appendChild(shell);
    wire();
    setSpecies(app.species, true);
  }

  function panelHtml(s, n) {
    const plate = String(n + 1).padStart(2, "0");
    const st = s.stage;
    return `<article class="panel" data-sp="${esc(s.key)}" style="--sx:${st.x};--sy:${st.y};--sw:${st.w};--sx-m:${st.xm};--sy-m:${st.ym};--sw-m:${st.wm};--shadow-y:${st.shadow};--mask:url('${esc(st.mask)}')">
      <div class="tab" role="button" tabindex="0" aria-label="Switch to ${esc(s.name)}">
        <div class="silhouette"></div>
        <div class="tab-top mono"><span>Plate ${plate}</span><span>${s.conditions.length} conditions</span></div>
        <span class="tab-hint mono">Switch</span>
        <div class="tab-name">${esc(s.name)}</div>
        <span class="tab-go">${icon("arrow")}</span>
      </div>
      <div class="plate">
        <div class="plate-inner" data-drop="Drop to check this ${esc(s.name.toLowerCase())}">
          <span class="reg tl"></span><span class="reg tr"></span><span class="reg bl"></span><span class="reg br"></span>
          <div class="plate-bar mono rise" style="--i:0"><span>Plate ${plate} — ${esc(s.name)}</span><span>${esc(s.arch)} · ${esc(s.input)} px · ${esc(s.accuracy)} val. accuracy</span></div>
          <div class="plate-head">
            <h2 class="plate-title rise" style="--i:1">${esc(s.name)}</h2>
            <div class="legend rise" style="--i:2">
              <span class="label mono">Checks for ${s.conditions.length} conditions</span>
              <ul class="conditions">${s.conditions.map((c) => `<li class="chip sev-${c.severity}"><i></i><span>${esc(c.short)}</span><em>${SEV[c.severity].label === "Healthy" ? "None" : SEV[c.severity].label}</em></li>`).join("")}</ul>
            </div>
          </div>
          <div class="specimen rise" data-blend="${esc(st.blend)}" style="--i:1"><img class="specimen-img" src="${esc(st.image)}" alt="" draggable="false"><div class="specimen-shadow"></div></div>
          <p class="plate-caption mono rise" style="--i:3">${esc(st.caption || "")}</p>
          <div class="plate-foot">
            <div class="dock rise" style="--i:4">
              <button class="btn btn-ink" data-act="pick">${icon("upload")}Check a ${esc(s.name.toLowerCase())} photo</button>
              <p class="dock-hint">or drop or paste a photo here · JPG, PNG, WebP</p>
              <div class="samples"><span class="label mono">No photo? Try</span>
                ${s.samples.map((x) => `<button class="sample" data-sample="${esc(x.id)}" data-tip="${esc(x.label)}" aria-label="Try sample: ${esc(x.label)}"><img src="${esc(x.src)}" alt="" loading="lazy"></button>`).join("")}
              </div>
            </div>
          </div>
        </div>
      </div>
    </article>`;
  }

  function wire() {
    const stage = $(".stage");
    const file = $(".file");

    // Plate content keeps its final width while the panel animates, so text never reflows mid-transition.
    const ro = new ResizeObserver(() => {
      const w = stage.clientWidth;
      const n = app.config.species.length;
      stage.style.setProperty("--plate-w", `${(w - 12 * (n - 1)) / (1 + 0.34 * (n - 1))}px`);
    });
    ro.observe(stage);

    $$(".panel").forEach((p) => {
      const tab = p.querySelector(".tab");
      const go = (e) => {
        if (p.classList.contains("is-active")) return;
        const r = p.getBoundingClientRect();
        const pointer = e && e.clientX;
        p.style.setProperty("--ox", pointer ? `${e.clientX - r.left}px` : "50%");
        p.style.setProperty("--oy", pointer ? `${e.clientY - r.top}px` : "50%");
        setSpecies(p.dataset.sp);
      };
      tab.addEventListener("click", go);
      tab.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), go(null)));
      p.addEventListener("pointermove", (e) => {
        const r = p.getBoundingClientRect();
        p.style.setProperty("--mx", `${e.clientX - r.left}px`);
        p.style.setProperty("--my", `${e.clientY - r.top}px`);
      });
      // Drag and drop onto the open plate
      let depth = 0;
      p.addEventListener("dragenter", (e) => { if (!p.classList.contains("is-active")) return; e.preventDefault(); depth++; p.classList.add("is-drop"); });
      p.addEventListener("dragover", (e) => { if (p.classList.contains("is-active")) e.preventDefault(); });
      p.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; p.classList.remove("is-drop"); } });
      p.addEventListener("drop", (e) => {
        e.preventDefault(); depth = 0; p.classList.remove("is-drop");
        const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith("image/"));
        if (f) analyseFile(f);
      });
    });

    root.addEventListener("click", (e) => {
      const t = e.target.closest("[data-act],[data-sample],[data-go]");
      if (!t) return;
      if (t.dataset.act === "pick") file.click();
      if (t.dataset.act === "switch") { setSpecies(t.dataset.sp); $(".stage").scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "center" }); }
      if (t.dataset.act === "top") $(".stage").scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "center" });
      if (t.dataset.sample) analyseSample(t.dataset.sample);
      if (t.dataset.go) root.querySelector(`[data-anchor="${t.dataset.go}"]`)?.scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "start" });
    });

    file.addEventListener("change", () => {
      const f = file.files?.[0];
      file.value = "";
      if (f) analyseFile(f);
    });

    // Paste a screenshot or copied photo straight in (desktop).
    const onPaste = (e) => {
      const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
      if (item) { e.preventDefault(); analyseFile(item.getAsFile()); }
    };
    document.addEventListener("paste", onPaste);

    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("seen"); io.unobserve(en.target); } });
    }, { rootMargin: "0px 0px -12% 0px" });
    $$(".reveal, .step").forEach((n) => io.observe(n));

    app.cleanup = () => { ro.disconnect(); io.disconnect(); document.removeEventListener("paste", onPaste); };
  }

  function setSpecies(key, initial = false) {
    app.species = key;
    $$(".panel").forEach((p) => {
      const on = p.dataset.sp === key;
      const plate = p.querySelector(".plate");
      if (!on && p.classList.contains("is-active")) {
        // Recentre the open iris without animating (it still covers the plate), so closing it reads as a lens shutting.
        plate.style.transition = "none";
        p.style.setProperty("--ox", "50%");
        p.style.setProperty("--oy", "50%");
        plate.style.setProperty("--r", "72%");
        void plate.offsetWidth;
        plate.style.transition = "";
      }
      if (on) plate.style.removeProperty("--r");
      p.classList.toggle("is-active", on);
      p.querySelector(".tab").setAttribute("tabindex", on ? "-1" : "0");
      p.querySelector(".tab").setAttribute("aria-hidden", on ? "true" : "false");
    });
    const other = app.config.species.find((s) => s.key !== key);
    const hint = $(".stage-hint");
    if (hint && other) hint.textContent = `Tap ${other.name.toLowerCase()} to switch`;
    if (!initial) announce(`${speciesCfg().name} selected`);
  }

  const speciesCfg = (key = app.species) => app.config.species.find((s) => s.key === key);
  const announce = (msg) => { const n = $("[data-live]"); if (n) n.textContent = msg; };

  async function analyseFile(f) {
    if (!f.type.startsWith("image/")) return showError("That file isn't a photo. Choose a JPG, PNG or WebP image.");
    let dataUrl;
    try {
      dataUrl = await fileToJpeg(f);
    } catch {
      return showError("This photo couldn't be opened here. If it came from an iPhone, share it as a JPG and try again.");
    }
    start({ image: dataUrl.split(",")[1] }, dataUrl, "Your photo");
  }

  function analyseSample(id) {
    const s = speciesCfg().samples.find((x) => x.id === id);
    if (s) start({ sample: id }, s.src, `Sample · ${s.label}`);
  }

  function start(payload, previewSrc, previewLabel) {
    const id = Math.random().toString(36).slice(2, 10);
    app.reqId = id;
    app.startedAt = performance.now();
    app.preview = { src: previewSrc, label: previewLabel };
    renderScanning();
    app.trigger?.("analyze", { id, species: app.species, ...payload });
  }

  function openReport() {
    const rep = $(".report");
    const wasOpen = rep.classList.contains("is-open");
    rep.classList.add("is-open");
    setTimeout(() => rep.scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "start" }), wasOpen ? 0 : 120);
  }

  function headHtml(sp, stateText) {
    const now = new Date();
    const when = now.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }) + " · " +
      now.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
    return `<header class="sheet-head mono">
      <b>AquaVision report</b><span>Ref AV-${esc((app.reqId || "").toUpperCase().slice(0, 6))}</span>
      <span>${esc(when)}</span><span>${esc(sp.name)} · ${esc(sp.arch)}</span>
      <span class="sheet-state"><span class="dot"></span><span>${stateText}</span></span>
    </header>`;
  }

  function lensHtml(readout) {
    const p = app.preview;
    return `<figure class="lens">
      <img class="lens-bg" src="${esc(p.src)}" alt="">
      <img class="lens-img" src="${esc(p.src)}" alt="${esc(p.label)}">
      <div class="lens-grid"></div><div class="lens-beam"></div>
      <span class="lens-c tl"></span><span class="lens-c tr"></span><span class="lens-c bl"></span><span class="lens-c br"></span>
      <span class="lens-read mono">${readout}</span>
      <figcaption class="lens-cap">${icon("camera")}${esc(p.label)}</figcaption>
    </figure>`;
  }

  function renderScanning() {
    const sp = speciesCfg();
    const sheet = $(".sheet");
    sheet.className = "sheet is-scanning";
    sheet.innerHTML = `${headHtml(sp, "Analysing")}
      <div class="sheet-body">
        ${lensHtml(`Scanning · ${esc(sp.input)} × ${esc(sp.input)}`)}
        <div class="dx"><div class="skel"><i></i><i></i><i></i><i></i><i></i><i></i></div></div>
      </div>`;
    app.renderedId = null;
    announce("Checking photo");
    openReport();
  }

  function showError(msg) {
    const sheet = $(".sheet");
    sheet.className = "sheet is-error";
    sheet.innerHTML = `${headHtml(speciesCfg(), "Not checked")}
      <div class="err"><h3>We couldn't check that one.</h3><p>${esc(msg)}</p></div>
      <div class="sheet-foot"><button class="btn btn-ink" data-act="pick">${icon("upload")}Choose another photo</button></div>`;
    announce(msg);
    openReport();
  }

  async function renderResult(r) {
    app.renderedId = r.id;
    const wait = MIN_SCAN_MS - (performance.now() - app.startedAt);
    if (wait > 0 && !reduceMotion()) await sleep(wait);
    if (app.reqId !== r.id) return; // a newer check started meanwhile
    if (r.error) return showError(r.error);

    const sp = speciesCfg(r.species);
    const sev = SEV[r.severity];
    const conf = r.confidence;
    const rel = conf >= 95 ? ["Very high", "high"] : conf >= 80 ? ["High", "high"] : conf >= 60 ? ["Medium", "medium"] : ["Low", "low"];
    const other = app.config.species.find((s) => s.key !== r.species);
    const sheet = $(".sheet");
    let i = 0;
    const d = () => `style="--i:${i++}"`;

    const oldLens = sheet.querySelector(".lens");
    sheet.className = `sheet is-result sev-${r.severity}`;
    sheet.style.setProperty("--lens-c", `var(--${r.severity === "none" ? "bloom" : r.severity === "moderate" ? "saline" : "alarm"})`);
    sheet.innerHTML = `${headHtml(sp, `Complete · ${Math.max(1, Math.round(r.ms))} ms`)}
      <div class="sheet-body">
        <div data-lens-slot></div>
        <div class="dx">
          ${conf < 60 ? `<div class="warn in" ${d()}>${icon("triangle")}<span>The model isn't sure about this photo. Retake it in daylight, closer to the animal, and check again.</span></div>` : ""}
          <p class="dx-k mono in" ${d()}>How serious it is</p>
          <div class="sev-row in" ${d()}>
            <span class="sev sev-${r.severity}">${icon(sev.icon)}${sev.label}</span>
            <span class="rank sev-${r.severity}">${sev.rank}</span>
          </div>
          <div class="meter in" ${d()} aria-hidden="true">
            <div class="meter-track">${Object.keys(SEV).map((k) => `<span class="m-${k}${k === r.severity ? " on" : ""}"></span>`).join("")}</div>
            <div class="meter-labels mono">${Object.entries(SEV).map(([k, v]) => `<span class="${k === r.severity ? "on" : ""}">${v.band}</span>`).join("")}</div>
            <span class="meter-pin"></span>
          </div>
          <h2 class="dx-name in" ${d()}>${esc(r.status)}</h2>
          <p class="dx-desc in" ${d()}>${esc(r.description)}</p>
          <div class="conf in" ${d()}>
            <div class="conf-row">
              <span class="conf-l mono">How sure the model is <span class="rel ${rel[1]}">${rel[0]}</span></span>
              <span class="conf-num"><span data-count="${conf.toFixed(1)}">0.0</span><small>%</small></span>
            </div>
            <div class="bar"><i style="--w:${conf.toFixed(1)}%;--gd:420ms"></i></div>
            <p class="conf-note">How sure the model is that this is ${esc(r.status)}. It doesn't change how serious the disease is — that comes from the disease itself.</p>
          </div>
          <div class="todo sev-${r.severity} in" ${d()}>
            <span class="label mono">What to do</span>
            <p>${esc(r.recommendation)}</p>
          </div>
        </div>
      </div>
      <div class="all in" ${d()}>
        <div class="all-head mono"><span>All ${r.probs.length} conditions checked</span><span>Model confidence</span></div>
        <ul class="rows">
          ${r.probs.map((p, n) => `<li class="row sev-${p.severity}${n === 0 ? " top" : ""}">
            <span class="row-dot"></span><span class="row-name">${esc(p.status)}</span><span class="row-pct">${p.p.toFixed(1)}%</span>
            <div class="bar"><i style="--w:${Math.max(p.p, 0.4).toFixed(1)}%;--gd:${600 + n * 70}ms"></i></div>
          </li>`).join("")}
        </ul>
      </div>
      <div class="sheet-foot in" ${d()}>
        <button class="btn btn-ink" data-act="pick">${icon("upload")}Check another photo</button>
        ${other ? `<button class="btn btn-ghost" data-act="switch" data-sp="${esc(other.key)}">${icon("swap")}Check a ${esc(other.name.toLowerCase())} instead</button>` : ""}
        <p class="disc">A screening tool, not a lab test. Confirm with your technician or a fisheries officer before treating.</p>
      </div>`;

    // Keep the scanning lens (same photo) so its corners animate into the locked state.
    const readout = `${esc(sp.arch)} · ${Math.max(1, Math.round(r.ms))} ms`;
    const slot = sheet.querySelector("[data-lens-slot]");
    if (oldLens && oldLens.querySelector(".lens-img")?.getAttribute("src") === app.preview.src) {
      slot.replaceWith(oldLens);
      oldLens.querySelector(".lens-read").innerHTML = readout;
    } else {
      slot.replaceWith(el(lensHtml(readout)));
    }

    // Severity pin slides to its band; confidence counts up.
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const pin = sheet.querySelector(".meter-pin");
      if (pin) pin.style.left = `${sev.pos * 100}%`;
    }));
    countUp(sheet.querySelector("[data-count]"));
    announce(`Result: ${r.status}. ${sev.label} severity. ${conf.toFixed(1)} percent confidence.`);
  }

  function countUp(node) {
    if (!node) return;
    const to = parseFloat(node.dataset.count);
    if (reduceMotion()) { node.textContent = to.toFixed(1); return; }
    const t0 = performance.now() + 300, dur = 1200;
    const tick = (t) => {
      const k = Math.min(1, Math.max(0, (t - t0) / dur));
      const e = 1 - Math.pow(2, -10 * k); // expo out
      node.textContent = (to * (k === 1 ? 1 : e)).toFixed(1);
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  app.update = (data) => {
    if (!data) return;
    if (!app.config && data.config) build(data.config);
    const r = data.result;
    if (!r || r.id === app.renderedId) return;
    if (!app.reqId) {
      // Restored after a remount: show the last result without replaying the scan.
      app.reqId = r.id;
      app.startedAt = 0;
      app.preview = { src: r.preview, label: r.label };
    }
    if (r.id === app.reqId) {
      renderResult(r);
      $(".report").classList.add("is-open");
    }
  };

  return app;
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  let app = parentElement.__aquavision;
  if (!app) {
    app = createApp(parentElement);
    parentElement.__aquavision = app;
  }
  app.trigger = setTriggerValue;
  app.update(data);
  // Streamlit runs this on unmount.
  return () => {
    app.cleanup?.();
    delete parentElement.__aquavision;
  };
}
