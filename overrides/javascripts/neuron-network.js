(function () {
  "use strict";

  var canvas = document.getElementById("neuron-canvas");
  if (!canvas) return;

  var ctx = canvas.getContext("2d");
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  var W = 0,
    H = 0;
  var mouse = { x: -9999, y: -9999 };
  var hoveredNode = -1;
  var time = 0;
  var frameId = null;

  var STAR_COUNT = 180;
  var NODE_BASE_RADIUS = 5;
  var NODE_HOVER_RADIUS = 8;
  var GLOW_BASE = 35;
  var HIT_RADIUS = 45;
  var SIGNAL_BASE_SPEED = 0.002;

  /* --- Node movement settings --- */
  var NODE_SPEED_MAX = 0.45;   // max pixels per frame
  var NODE_WANDER = 0.018;     // velocity perturbation per frame
  var NODE_MARGIN = 90;        // soft boundary margin (px from edge)
  var NODE_REPULSE_DIST = 90;  // nodes repel each other inside this radius
  var NODE_REPULSE_FORCE = 0.006;

  var chapters = [
    { label: "Introduction",         url: "chapter_preface/",                  px: 0.5,  py: 0.12 },
    { label: "Coding with AI",       url: "chapter_ai/",                       px: 0.73, py: 0.18 },
    { label: "Optimization",         url: "chapter_optimization/",             px: 0.88, py: 0.34 },
    { label: "Neural Networks",      url: "chapter_neural_networks/",          px: 0.88, py: 0.56 },
    { label: "Language Models Architecture",      url: "chapter_language_model/",           px: 0.75, py: 0.76 },
    { label: "Language Models Training",          url: "chapter_llm_ft/",                   px: 0.55, py: 0.86 },
    { label: "Genomic Foundation Models",   url: "chapter_foundation_model/",         px: 0.35, py: 0.86 },
    { label: "Image Foundation Models",     url: "chapter_image_foundation_model/",   px: 0.18, py: 0.76 },
    { label: "Self-Supervised Learning",      url: "chapter_self_supervised_learning/", px: 0.10, py: 0.56 },
    { label: "Generative Models",    url: "chapter_generative_model/",         px: 0.10, py: 0.34 },
    { label: "Reinforcement Learning", url: "chapter_reinforcement_learning/", px: 0.27, py: 0.18 },
  ];

  var connections = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
    [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
    [10, 0],
    [2, 8], [3, 7], [5, 10], [9, 5], [0, 5], [3, 6], [1, 9],
  ];

  var stars = [];
  var signals = [];
  var nodes = [];
  var bgParticles = [];
  var BG_PARTICLE_COUNT = 60;

  function resize() {
    var rect = canvas.parentElement.getBoundingClientRect();
    W = rect.width;
    H = rect.height;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    /* Preserve existing node positions on resize; initialise fresh on first call */
    if (nodes.length === 0) {
      nodes = chapters.map(function (ch) {
        return {
          label: ch.label,
          url:   ch.url,
          x:  ch.px * W,
          y:  ch.py * H,
          vx: (Math.random() - 0.5) * 0.3,
          vy: (Math.random() - 0.5) * 0.3,
        };
      });
    } else {
      /* Rescale positions proportionally */
      var scaleX = W / (nodes._prevW || W);
      var scaleY = H / (nodes._prevH || H);
      for (var i = 0; i < nodes.length; i++) {
        nodes[i].x *= scaleX;
        nodes[i].y *= scaleY;
      }
    }
    nodes._prevW = W;
    nodes._prevH = H;

    initStars();
    initBgParticles();
  }

  function initStars() {
    stars = [];
    for (var i = 0; i < STAR_COUNT; i++) {
      stars.push({
        x: Math.random() * W,
        y: Math.random() * H,
        r: Math.random() * 1.2 + 0.3,
        alpha: Math.random() * 0.4 + 0.2,
        phase: Math.random() * Math.PI * 2,
        speed: Math.random() * 0.4 + 0.1,
      });
    }
  }

  function initBgParticles() {
    bgParticles = [];
    for (var i = 0; i < BG_PARTICLE_COUNT; i++) {
      bgParticles.push({
        x:  Math.random() * W,
        y:  Math.random() * H,
        vx: (Math.random() - 0.5) * 0.15,
        vy: (Math.random() - 0.5) * 0.15,
        r:  Math.random() * 1.5 + 0.5,
        alpha: Math.random() * 0.15 + 0.05,
      });
    }
  }

  function initSignals() {
    signals = [];
    for (var i = 0; i < connections.length; i++) {
      signals.push({
        ci:       i,
        progress: Math.random(),
        speed:    SIGNAL_BASE_SPEED + Math.random() * 0.0015,
        alpha:    0.4 + Math.random() * 0.4,
      });
      if (Math.random() > 0.5) {
        signals.push({
          ci:       i,
          progress: Math.random(),
          speed:    SIGNAL_BASE_SPEED + Math.random() * 0.001,
          alpha:    0.2 + Math.random() * 0.3,
        });
      }
    }
  }

  /* ---- Node movement ---- */
  function updateNodes() {
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];

      /* Dampen hovered node so it nearly stops under the cursor */
      if (hoveredNode === i) {
        n.vx *= 0.88;
        n.vy *= 0.88;
      } else {
        /* Random wander */
        n.vx += (Math.random() - 0.5) * NODE_WANDER;
        n.vy += (Math.random() - 0.5) * NODE_WANDER;
      }

      /* Soft repulsion from canvas edges */
      var m = NODE_MARGIN;
      if (n.x < m)         n.vx += (m - n.x) * 0.004;
      if (n.x > W - m)     n.vx -= (n.x - (W - m)) * 0.004;
      if (n.y < m)         n.vy += (m - n.y) * 0.004;
      if (n.y > H - m)     n.vy -= (n.y - (H - m)) * 0.004;

      /* Soft repulsion between nodes so they don't pile up */
      for (var j = 0; j < nodes.length; j++) {
        if (j === i) continue;
        var dx = n.x - nodes[j].x;
        var dy = n.y - nodes[j].y;
        var dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > 0 && dist < NODE_REPULSE_DIST) {
          var force = NODE_REPULSE_FORCE * (1 - dist / NODE_REPULSE_DIST);
          n.vx += (dx / dist) * force;
          n.vy += (dy / dist) * force;
        }
      }

      /* Cap speed */
      var speed = Math.sqrt(n.vx * n.vx + n.vy * n.vy);
      if (speed > NODE_SPEED_MAX) {
        n.vx = (n.vx / speed) * NODE_SPEED_MAX;
        n.vy = (n.vy / speed) * NODE_SPEED_MAX;
      }

      /* Apply velocity */
      n.x += n.vx;
      n.y += n.vy;

      /* Hard clamp as safety net */
      n.x = Math.max(m * 0.5, Math.min(W - m * 0.5, n.x));
      n.y = Math.max(m * 0.5, Math.min(H - m * 0.5, n.y));
    }
  }

  /* ---- Drawing helpers ---- */
  function drawBackground() {
    ctx.fillStyle = "#0a0a1a";
    ctx.fillRect(0, 0, W, H);

    var g = ctx.createRadialGradient(W * 0.5, H * 0.4, 0, W * 0.5, H * 0.4, Math.max(W, H) * 0.55);
    g.addColorStop(0, "rgba(15, 25, 60, 0.4)");
    g.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
  }

  function drawStars() {
    for (var i = 0; i < stars.length; i++) {
      var s = stars[i];
      var twinkle = Math.sin(time * 0.01 * s.speed + s.phase) * 0.35 + 0.65;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 255, 255, " + (s.alpha * twinkle).toFixed(3) + ")";
      ctx.fill();
    }
  }

  function drawBgParticles() {
    for (var i = 0; i < bgParticles.length; i++) {
      var p = bgParticles[i];
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = W;
      if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H;
      if (p.y > H) p.y = 0;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(100, 180, 255, " + p.alpha.toFixed(3) + ")";
      ctx.fill();
    }

    for (var i = 0; i < bgParticles.length; i++) {
      for (var j = i + 1; j < bgParticles.length; j++) {
        var a = bgParticles[i];
        var b = bgParticles[j];
        var dx = a.x - b.x;
        var dy = a.y - b.y;
        var dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          var alpha = ((1 - dist / 120) * 0.06).toFixed(4);
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = "rgba(100, 180, 255, " + alpha + ")";
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
  }

  function drawConnections() {
    for (var i = 0; i < connections.length; i++) {
      var pair = connections[i];
      var a = nodes[pair[0]];
      var b = nodes[pair[1]];
      var isHovered = hoveredNode >= 0 && (hoveredNode === pair[0] || hoveredNode === pair[1]);

      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      if (isHovered) {
        ctx.strokeStyle = "rgba(0, 220, 255, 0.35)";
        ctx.lineWidth = 2;
      } else {
        ctx.strokeStyle = "rgba(100, 180, 255, 0.1)";
        ctx.lineWidth = 1;
      }
      ctx.stroke();
    }
  }

  function drawSignals() {
    for (var i = 0; i < signals.length; i++) {
      var sig = signals[i];
      var pair = connections[sig.ci];
      var a = nodes[pair[0]];
      var b = nodes[pair[1]];
      var isHovered = hoveredNode >= 0 && (hoveredNode === pair[0] || hoveredNode === pair[1]);

      sig.progress += sig.speed * (isHovered ? 3 : 1);
      if (sig.progress > 1) sig.progress -= 1;

      var x = a.x + (b.x - a.x) * sig.progress;
      var y = a.y + (b.y - a.y) * sig.progress;

      var r  = isHovered ? 3 : 2;
      var al = sig.alpha * (isHovered ? 1 : 0.4);

      var g = ctx.createRadialGradient(x, y, 0, x, y, r * 4);
      g.addColorStop(0, "rgba(120, 210, 255, " + al.toFixed(3) + ")");
      g.addColorStop(1, "rgba(120, 210, 255, 0)");
      ctx.beginPath();
      ctx.arc(x, y, r * 4, 0, Math.PI * 2);
      ctx.fillStyle = g;
      ctx.fill();
    }
  }

  function drawMouseGlow() {
    if (mouse.x < -999) return;
    var g = ctx.createRadialGradient(mouse.x, mouse.y, 0, mouse.x, mouse.y, 120);
    g.addColorStop(0, "rgba(80, 180, 255, 0.06)");
    g.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.beginPath();
    ctx.arc(mouse.x, mouse.y, 120, 0, Math.PI * 2);
    ctx.fillStyle = g;
    ctx.fill();
  }

  function drawNodes() {
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      var isHovered  = hoveredNode === i;
      var isNeighbor = false;

      if (hoveredNode >= 0 && !isHovered) {
        for (var c = 0; c < connections.length; c++) {
          var pair = connections[c];
          if ((pair[0] === hoveredNode && pair[1] === i) ||
              (pair[1] === hoveredNode && pair[0] === i)) {
            isNeighbor = true;
            break;
          }
        }
      }

      var pulse = Math.sin(time * 0.018 + i * 0.57) * 0.25 + 0.75;
      var r = isHovered ? NODE_HOVER_RADIUS : isNeighbor ? NODE_BASE_RADIUS + 1.5 : NODE_BASE_RADIUS;
      var glowR = isHovered ? GLOW_BASE * 1.8 : isNeighbor ? GLOW_BASE * 1.2 : GLOW_BASE * pulse;

      var outerGlow = ctx.createRadialGradient(n.x, n.y, r, n.x, n.y, glowR);
      if (isHovered) {
        outerGlow.addColorStop(0, "rgba(0, 240, 255, 0.35)");
        outerGlow.addColorStop(0.5, "rgba(0, 180, 255, 0.1)");
        outerGlow.addColorStop(1, "rgba(0, 0, 0, 0)");
      } else if (isNeighbor) {
        outerGlow.addColorStop(0, "rgba(80, 200, 255, 0.2)");
        outerGlow.addColorStop(1, "rgba(0, 0, 0, 0)");
      } else {
        outerGlow.addColorStop(0, "rgba(100, 180, 255, " + (0.15 * pulse).toFixed(3) + ")");
        outerGlow.addColorStop(1, "rgba(0, 0, 0, 0)");
      }
      ctx.beginPath();
      ctx.arc(n.x, n.y, glowR, 0, Math.PI * 2);
      ctx.fillStyle = outerGlow;
      ctx.fill();

      if (isHovered) {
        var ringPulse = Math.sin(time * 0.06) * 0.15 + 0.85;
        ctx.beginPath();
        ctx.arc(n.x, n.y, r + 12 * ringPulse, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(0, 240, 255, " + (0.3 * ringPulse).toFixed(3) + ")";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      var core = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r);
      if (isHovered) {
        core.addColorStop(0, "#ffffff");
        core.addColorStop(1, "rgba(0, 240, 255, 1)");
      } else if (isNeighbor) {
        core.addColorStop(0, "rgba(220, 240, 255, 1)");
        core.addColorStop(1, "rgba(80, 200, 255, 0.9)");
      } else {
        core.addColorStop(0, "rgba(200, 230, 255, 1)");
        core.addColorStop(1, "rgba(100, 180, 255, 0.8)");
      }
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fillStyle = core;
      ctx.fill();

      /* Label — always on the side with more space (top or bottom of node) */
      var labelBelow = n.y < H * 0.78;
      var labelY = labelBelow ? n.y + r + 16 : n.y - r - 10;
      ctx.textAlign = "center";
      ctx.textBaseline = labelBelow ? "top" : "bottom";

      if (isHovered) {
        ctx.font = "bold 13px -apple-system, BlinkMacSystemFont, sans-serif";
        ctx.fillStyle = "rgba(0, 240, 255, 1)";
        ctx.shadowColor = "rgba(0, 200, 255, 0.6)";
        ctx.shadowBlur = 8;
      } else if (isNeighbor) {
        ctx.font = "12px -apple-system, BlinkMacSystemFont, sans-serif";
        ctx.fillStyle = "rgba(180, 220, 255, 0.85)";
        ctx.shadowColor = "transparent";
        ctx.shadowBlur = 0;
      } else {
        ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
        ctx.fillStyle = "rgba(180, 210, 240, 0.6)";
        ctx.shadowColor = "transparent";
        ctx.shadowBlur = 0;
      }
      ctx.fillText(n.label, n.x, labelY);
      ctx.shadowColor = "transparent";
      ctx.shadowBlur = 0;
    }
  }

  function findHoveredNode(mx, my) {
    var best = -1;
    var bestDist = HIT_RADIUS;
    for (var i = 0; i < nodes.length; i++) {
      var dx = nodes[i].x - mx;
      var dy = nodes[i].y - my;
      var d  = Math.sqrt(dx * dx + dy * dy);
      if (d < bestDist) { bestDist = d; best = i; }
    }
    return best;
  }

  function animate() {
    time++;
    updateNodes();
    ctx.clearRect(0, 0, W, H);
    drawBackground();
    drawStars();
    drawBgParticles();
    drawConnections();
    drawSignals();
    drawMouseGlow();
    drawNodes();
    frameId = requestAnimationFrame(animate);
  }

  function onMouseMove(e) {
    var rect = canvas.getBoundingClientRect();
    mouse.x = e.clientX - rect.left;
    mouse.y = e.clientY - rect.top;
    hoveredNode = findHoveredNode(mouse.x, mouse.y);
    canvas.style.cursor = hoveredNode >= 0 ? "pointer" : "default";
  }

  function onClick() {
    if (hoveredNode >= 0) {
      window.location.href = nodes[hoveredNode].url;
    }
  }

  function onMouseLeave() {
    mouse.x = -9999;
    mouse.y = -9999;
    hoveredNode = -1;
    canvas.style.cursor = "default";
  }

  function onTouchStart(e) {
    if (e.touches.length === 1) {
      var t = e.touches[0];
      var rect = canvas.getBoundingClientRect();
      var mx = t.clientX - rect.left;
      var my = t.clientY - rect.top;
      var hit = findHoveredNode(mx, my);
      if (hit >= 0) window.location.href = nodes[hit].url;
    }
  }

  var resizeTimer;
  function onResize() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 150);
  }

  canvas.addEventListener("mousemove",  onMouseMove);
  canvas.addEventListener("click",      onClick);
  canvas.addEventListener("mouseleave", onMouseLeave);
  canvas.addEventListener("touchstart", onTouchStart, { passive: true });
  window.addEventListener("resize",     onResize);

  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      if (frameId) { cancelAnimationFrame(frameId); frameId = null; }
    } else {
      if (!frameId) animate();
    }
  });

  resize();
  initSignals();
  animate();
})();
