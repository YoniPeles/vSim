import * as THREE from 'three';

// Shader materials for the cluster scene. Physical hardware is shaded dark and quiet; everything that
// encodes a quantity (memory fill, the model's layers, traffic) is light. Colours arrive as linear RGB
// and leave through <colorspace_fragment>; the scene renders without tone mapping so identity colours
// on screen match the legend swatches.

const fogUniforms = () => THREE.UniformsUtils.clone(THREE.UniformsLib.fog);

const COMMON = /* glsl */ `
  float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
  }
  // 1 on a line d pixels away with half-width w pixels, anti-aliased.
  float aaLine(float d, float w) {
    return 1.0 - smoothstep(w - 0.5, w + 0.5, d);
  }
  // 1 inside [lo, hi], anti-aliased over aa.
  float inside(float x, float lo, float hi, float aa) {
    return smoothstep(lo - aa, lo + aa, x) * (1.0 - smoothstep(hi - aa, hi + aa, x));
  }
`;

const FOG_FRAG = /* glsl */ `
  #include <fog_pars_fragment>
  float fogAmount() {
  #ifdef USE_FOG
    return smoothstep(fogNear, fogFar, vFogDepth);
  #else
    return 0.0;
  #endif
  }
`;

/** Instance scale (world size of a unit box) from the instance matrix. */
const INSTANCE_SIZE = /* glsl */ `vec3(length(instanceMatrix[0].xyz), length(instanceMatrix[1].xyz), length(instanceMatrix[2].xyz))`;

// ---------------------------------------------------------------------------------------------
// The layer hologram above each die, in two parts that share per-layer textures:
//  - uLayerTex (L × 2): row 0 = attention light, row 1 = FFN light (rgb) for each layer;
//  - uHeatTex (L × 1): step-trace glow (r = attention, g = FFN) of each layer of instance uTraceInst.
// The shell is one box per GPU spanning the whole model; its sides draw each layer's edge as an
// anti-aliased line (bright where this GPU holds the layer and its tensor-parallel share, ghostly
// elsewhere). Wafers are flat quads, one per held layer: their overlapping faces build the glowing
// volume, and MoE layers held as whole experts carry one cell per resident expert.

const HOLO_COMMON = /* glsl */ `
  uniform sampler2D uLayerTex;
  uniform sampler2D uHeatTex;
  uniform float uTraceInst;
  uniform float uLayers;
  const float ATTN = 0.3;
  const vec3 SCAN = vec3(1.0, 0.8, 0.52);
  vec3 layerLight(float layer, bool attn) {
    return texture2D(uLayerTex, vec2((layer + 0.5) / uLayers, attn ? 0.25 : 0.75)).rgb;
  }
  float layerHeat(float layer, float inst, bool attn) {
    if (abs(inst - uTraceInst) > 0.5) return 0.0;
    vec2 h = texture2D(uHeatTex, vec2((layer + 0.5) / uLayers, 0.5)).rg;
    return attn ? h.x : h.y;
  }
`;

const holoUniforms = () => ({
  ...fogUniforms(),
  uLayerTex: { value: null as THREE.DataTexture | null },
  uHeatTex: { value: null as THREE.DataTexture | null },
  uTraceInst: { value: -1 },
  uLayers: { value: 1 },
});

export function stackShellMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: { ...holoUniforms(), uStep: { value: 0.03 }, uY0: { value: 0.14 }, uWaferH: { value: 0.01 } },
    fog: true,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      attribute vec2 aRange;
      attribute vec4 aSlice;
      attribute float aInst;
      varying vec3 vL;
      varying vec3 vN;
      varying vec3 vW;
      varying vec2 vRange;
      varying vec4 vSlice;
      varying float vInst;
      #include <fog_pars_vertex>
      void main() {
        vL = position;
        vN = normal;
        vRange = aRange;
        vSlice = aSlice;
        vInst = aInst;
        vec4 wp = modelMatrix * instanceMatrix * vec4(position, 1.0);
        vW = wp.xyz;
        vec4 mvPosition = viewMatrix * wp;
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uStep;
      uniform float uY0;
      uniform float uWaferH;
      varying vec3 vL;
      varying vec3 vN;
      varying vec3 vW;
      varying vec2 vRange;
      varying vec4 vSlice;
      varying float vInst;
      ${COMMON}
      ${HOLO_COMMON}
      ${FOG_FRAG}
      void main() {
        if (abs(vN.y) > 0.5) discard;
        float ly = (vW.y - uY0) / uStep;
        float li = clamp(floor(ly), 0.0, uLayers - 1.0);
        float fly = max(fwidth(ly), 1e-5);
        // The wafer edge occupies the bottom uWaferH of each layer pitch; once pitches shrink
        // below a few pixels the lines give way to their average, so nothing aliases.
        float fill = uWaferH / uStep;
        float edgeLine = inside(fract(ly), 0.0, fill, fly * 0.6);
        float far = smoothstep(0.2, 0.55, fly);
        float line = mix(edgeLine, fill * 0.55, far);
        // The step trace's scan fills its whole layer pitch when layers are too fine to resolve.
        float scanLine = mix(edgeLine, 0.8, far);
        float held = inside(ly, vRange.x, vRange.y, fly * 0.5);
        bool fb = abs(vN.z) > 0.5;
        float u = fb ? vL.x + 0.5 : (vN.x > 0.0 ? 0.9995 : 0.0005);
        float w = fb ? (vN.z > 0.0 ? 0.0005 : 0.9995) : 0.5 - vL.z;
        bool attn = w < ATTN;
        vec2 lit = attn ? vSlice.xy : vSlice.zw;
        float inS = inside(u, lit.x, lit.y, max(fwidth(u), 1e-4) * 0.7);
        vec3 base = layerLight(li, attn);
        float heat = layerHeat(li, vInst, attn) * held;
        // A thin tensor-parallel share glows harder than a whole layer, so it still reads at a glance.
        float boost = mix(1.35, 1.0, lit.y - lit.x);
        vec3 col = base * line * mix(0.05, mix(0.3, 0.9 * boost, inS), held);
        col += SCAN * heat * scanLine * mix(0.9, 3.0, inS);
        // The hologram's outline: light gathers along the vertical edges of the volume, like a prism's.
        vec2 fc = fb ? vL.xy : vL.zy;
        float de = (0.5 - abs(fc.x)) / max(fwidth(fc.x), 1e-5);
        col += vec3(0.55, 0.7, 0.88) * 0.05 * aaLine(de, 0.6);
        col += base * held * mix(0.04, 0.14, inS) * pow(2.0 * abs(fc.x), 8.0);
        col *= 1.0 - fogAmount();
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
}

export function waferMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      ...holoUniforms(),
      uFace: { value: 0.03 },
      uExperts: { value: null as THREE.DataTexture | null },
      uExpertCount: { value: 1 },
      uExpertOn: { value: 0 },
    },
    fog: true,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      attribute vec4 aSlice;
      attribute vec3 aExp;
      attribute vec2 aLayer;
      varying vec2 vP;
      varying vec2 vSize;
      varying vec4 vSlice;
      varying vec3 vExp;
      varying vec2 vLayer;
      #include <fog_pars_vertex>
      void main() {
        vP = position.xz;
        vec3 s = ${INSTANCE_SIZE};
        vSize = s.xz;
        vSlice = aSlice;
        vExp = aExp;
        vLayer = aLayer;
        vec4 mvPosition = viewMatrix * modelMatrix * instanceMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uExperts;
      uniform float uExpertCount;
      uniform float uExpertOn;
      uniform float uFace;
      varying vec2 vP;
      varying vec2 vSize;
      varying vec4 vSlice;
      varying vec3 vExp; // cells (0 dense, -1 sharded), first expert id, 1 = top wafer of the stack
      varying vec2 vLayer;
      ${COMMON}
      ${HOLO_COMMON}
      ${FOG_FRAG}
      void main() {
        float u = clamp(vP.x + 0.5, 0.0005, 0.9995);
        float w = clamp(0.5 - vP.y, 0.0005, 0.9995); // 0 = front edge, 1 = back edge
        bool attn = w < ATTN;
        bool cap = vExp.z > 0.5;
        vec2 lit = attn ? vSlice.xy : vSlice.zw;
        vec3 base = layerLight(vLayer.x, attn);
        float heat = layerHeat(vLayer.x, vLayer.y, attn);
        float fu = max(fwidth(u), 1e-5);
        float fw = max(fwidth(w), 1e-5);
        float inS = inside(u, lit.x, lit.y, fu * 0.7);
        float boost = mix(1.35, 1.0, lit.y - lit.x);
        // Inner wafers are faint and build the volume; the top wafer is a legible cap.
        float face = cap ? 0.07 : uFace;
        vec3 col = base * face * mix(0.3, boost, inS);
        // Outline of the whole layer, the lit share, and the attention | FFN seam.
        float dEdge = min(min(u, 1.0 - u) / fu, min(w, 1.0 - w) / fw);
        float dSlice = min(abs(u - lit.x), abs(u - lit.y)) / fu;
        float dSeam = abs(w - ATTN) / fw;
        float res = 1.0 - smoothstep(0.004, 0.012, max(fu, fw));
        float lines = 0.05 * aaLine(dEdge, 0.6) + 0.22 * inS * aaLine(dSlice, 0.6) + 0.12 * inS * aaLine(dSeam, 0.5);
        col += base * lines * (cap ? 2.4 : res);
        if (!attn && vExp.x > 0.5) {
          // One cell per resident expert, laid out as a grid over the FFN band.
          float n = vExp.x;
          float cu = (u - lit.x) / max(lit.y - lit.x, 1e-4);
          float cw = (w - ATTN) / (1.0 - ATTN);
          float aspect = (vSize.x * (lit.y - lit.x)) / (vSize.y * (1.0 - ATTN));
          float cols = max(1.0, ceil(sqrt(n * aspect)));
          float rows = ceil(n / cols);
          vec2 g = vec2(cu * cols, cw * rows);
          vec2 cell = floor(g);
          float idx = cell.y * cols + cell.x;
          vec2 q = abs(fract(g) - 0.5) - 0.28;
          float sd = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - 0.08;
          vec2 fg = max(fwidth(g), vec2(1e-5));
          float pxPerCell = 1.0 / max(fg.x, fg.y);
          float mask = 1.0 - smoothstep(-0.5, 0.5, sd * pxPerCell);
          mask = mix(0.3, mask, smoothstep(2.5, 6.0, pxPerCell)) * step(idx, n - 0.5) * inS;
          float load = uExpertOn > 0.5 ? texture2D(uExperts, vec2((vExp.y + idx + 0.5) / uExpertCount, 0.5)).r : 0.0;
          vec3 c = mix(base, SCAN, load);
          col += cap ? c * mask * (0.16 + load * 1.6) : c * uFace * mask * (0.8 + load * 5.0);
        } else if (!attn && vExp.x < -0.5) {
          // Every expert tensor-parallel sharded: a fine hatch over the lit share.
          float s = (u * vSize.x + w * vSize.y) / 0.035;
          float fs = max(fwidth(s), 1e-5);
          float h = aaLine(abs(fract(s) - 0.5) / fs, 0.5) * (1.0 - smoothstep(0.25, 0.6, fs));
          col += base * (cap ? 0.12 : uFace * 1.2) * h * inS;
        }
        col += SCAN * heat * mix(0.1, 0.55, inS);
        col *= 1.0 - fogAmount();
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
}

// ---------------------------------------------------------------------------------------------
// HBM: luminous liquid fill (opaque) in a glass column (additive). The fill rises through
// weights → activations → runtime overhead → KV in use; free KV is hatched empty glass up to the
// gpu-memory-utilization line; above it the glass is empty.

export function hbmFillMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      ...fogUniforms(),
      uBase: { value: 0.12 },
      uH: { value: 1.8 },
      uTime: { value: 0 },
      uShimmer: { value: 0 },
      uColW: { value: new THREE.Color() },
      uColA: { value: new THREE.Color() },
      uColO: { value: new THREE.Color() },
      uColK: { value: new THREE.Color() },
    },
    fog: true,
    vertexShader: /* glsl */ `
      attribute vec4 aBands;
      uniform float uBase;
      uniform float uH;
      varying vec3 vL;
      varying vec3 vN;
      varying float vF;
      varying vec4 vBands;
      #include <fog_pars_vertex>
      void main() {
        vL = position;
        vN = normal;
        vBands = aBands;
        vec4 wp = modelMatrix * instanceMatrix * vec4(position, 1.0);
        vF = (wp.y - uBase) / uH;
        vec4 mvPosition = viewMatrix * wp;
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uColW;
      uniform vec3 uColA;
      uniform vec3 uColO;
      uniform vec3 uColK;
      uniform float uTime;
      uniform float uShimmer;
      uniform float uH;
      varying vec3 vL;
      varying vec3 vN;
      varying float vF;
      varying vec4 vBands;
      ${COMMON}
      ${FOG_FRAG}
      void main() {
        float f = vF;
        bool kv = f >= vBands.z;
        vec3 c = f < vBands.x ? uColW : f < vBands.y ? uColA : !kv ? uColO : uColK;
        float shade;
        float across;
        bool side = vN.y < 0.5;
        if (!side) { shade = 1.08; across = 0.0; }
        else if (abs(vN.z) > 0.5) { shade = vN.z > 0.0 ? 1.0 : 0.7; across = vL.x; }
        else { shade = vN.x > 0.0 ? 0.7 : 0.62; across = vL.z; }
        // A rounded, softly lit liquid: brightest down the middle of each face.
        shade *= 0.66 + 0.34 * (1.0 - 4.0 * across * across);
        float ff = max(fwidth(f), 1e-5);
        vec3 col = c * shade;
        if (side) {
          // Lit from the meniscus down, and layered like the DRAM dies it sits in.
          col *= 0.78 + 0.27 * smoothstep(0.0, max(vBands.w, 0.05), f);
          float g = f * 12.0;
          float fg = max(fwidth(g), 1e-5);
          col *= 1.0 - 0.16 * aaLine(abs(fract(g + 0.5) - 0.5) / fg, 0.5) * (1.0 - smoothstep(0.1, 0.3, fg));
          // Dark seam between categories so thin bands stay countable.
          float dB = min(min(abs(f - vBands.x), abs(f - vBands.y)), abs(f - vBands.z)) / ff;
          col *= 1.0 - 0.6 * aaLine(dB, 0.5) * step(0.004, vBands.x);
          // Paged KV: a block grid that flickers while the simulation allocates and frees.
          if (kv) {
            vec2 g = vec2((across + 0.5) * 5.0, f * uH / 0.03);
            vec2 fg = max(fwidth(g), vec2(1e-5));
            vec2 e = (0.5 - abs(fract(g) - 0.5)) / fg;
            float grid = aaLine(min(e.x, e.y), 0.5) * (1.0 - smoothstep(0.12, 0.3, max(fg.x, fg.y)));
            float h = hash12(floor(g) + floor(uTime * 3.0) * 17.0);
            col *= 1.0 - 0.3 * grid;
            col += c * uShimmer * 0.35 * step(0.82, h);
          }
          // Meniscus: a bright rim where the fill meets the empty glass.
          float dTop = (vBands.w - f) / ff;
          col += c * 0.9 * aaLine(dTop, 1.0);
        } else {
          col += c * 0.12;
        }
        col = mix(col, fogColor, fogAmount());
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
}

export function hbmGlassMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      ...fogUniforms(),
      uBase: { value: 0.12 },
      uH: { value: 1.8 },
      uDies: { value: 12 },
      uGlass: { value: new THREE.Color('#a8c8e8') },
      uColK: { value: new THREE.Color() },
    },
    fog: true,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      attribute vec2 aKv;
      uniform float uBase;
      uniform float uH;
      varying vec3 vL;
      varying vec3 vN;
      varying vec3 vW;
      varying vec3 vSize;
      varying float vF;
      varying vec2 vKv;
      #include <fog_pars_vertex>
      void main() {
        vL = position;
        vN = normal;
        vKv = aKv;
        vSize = ${INSTANCE_SIZE};
        vec4 wp = modelMatrix * instanceMatrix * vec4(position, 1.0);
        vW = wp.xyz;
        vF = (wp.y - uBase) / uH;
        vec4 mvPosition = viewMatrix * wp;
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uGlass;
      uniform vec3 uColK;
      uniform float uDies;
      uniform float uH;
      varying vec3 vL;
      varying vec3 vN;
      varying vec3 vW;
      varying vec3 vSize;
      varying float vF;
      varying vec2 vKv;
      ${COMMON}
      ${FOG_FRAG}
      void main() {
        float f = vF;
        vec2 fc;
        vec2 fs;
        bool side = abs(vN.y) < 0.5;
        if (!side) { fc = vL.xz; fs = vSize.xz; }
        else if (abs(vN.z) > 0.5) { fc = vL.xy; fs = vSize.xy; }
        else { fc = vL.zy; fs = vSize.zy; }
        vec2 fwc = max(fwidth(fc), vec2(1e-5));
        vec2 de = (0.5 - abs(fc)) / fwc;
        // Glass edges recede on small (distant) columns so a large cluster is not a thicket of lines.
        float facePx = 1.0 / max(fwc.x, fwc.y);
        float edge = aaLine(min(de.x, de.y), 0.7) * mix(0.35, 1.0, smoothstep(10.0, 40.0, facePx));
        vec3 V = normalize(cameraPosition - vW);
        float fres = pow(1.0 - abs(dot(vN, V)), 3.0);
        float facing = gl_FrontFacing ? 1.0 : 0.4;
        vec3 col = uGlass * (0.24 * edge + 0.08 * fres) * facing;
        if (side && gl_FrontFacing) {
          // Two soft vertical highlights, the tell of a glass vessel.
          float sx = fc.x;
          col += uGlass * (0.07 * (1.0 - smoothstep(0.0, 0.07, abs(sx + 0.27))) + 0.035 * (1.0 - smoothstep(0.0, 0.03, abs(sx - 0.3))));
        }
        if (side) {
          float ff = max(fwidth(f), 1e-5);
          // DRAM dies of the HBM stack: faint seams at equal shares of capacity.
          float g = f * uDies;
          float fg = max(fwidth(g), 1e-5);
          float die = aaLine(abs(fract(g + 0.5) - 0.5) / fg, 0.5) * (1.0 - smoothstep(0.08, 0.25, fg));
          col += uGlass * 0.045 * die * facing;
          // Free KV: hatched empty glass, like the memory bar in the side panel.
          float inFree = inside(f, vKv.x, vKv.y, ff);
          float s = (fc.x * fs.x + f * uH) / 0.055;
          float fsx = max(fwidth(s), 1e-5);
          float hatch = aaLine(abs(fract(s) - 0.5) / fsx, 0.55) * (1.0 - smoothstep(0.3, 0.7, fsx));
          col += uColK * inFree * (0.05 + 0.3 * hatch * (gl_FrontFacing ? 1.0 : 0.25));
          // The gpu-memory-utilization line: KV can fill the glass up to here.
          col += uColK * 0.85 * aaLine(abs(f - vKv.y) / ff, 0.7) * facing;
        }
        col *= 1.0 - fogAmount();
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
}

// ---------------------------------------------------------------------------------------------
// Silicon die: a die-shot of SM blocks with a faint thin-film sheen. Powered dies glow softly
// under the layer stack they hold; idle dies stay dark.

export function dieMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: { ...fogUniforms(), uLight: { value: new THREE.Vector3(0.35, 1, 0.55).normalize() } },
    fog: true,
    vertexShader: /* glsl */ `
      attribute float aActive;
      varying vec3 vL;
      varying vec3 vN;
      varying vec3 vW;
      varying float vActive;
      #include <fog_pars_vertex>
      void main() {
        vL = position;
        vN = normal;
        vActive = aActive;
        vec4 wp = modelMatrix * instanceMatrix * vec4(position, 1.0);
        vW = wp.xyz;
        vec4 mvPosition = viewMatrix * wp;
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uLight;
      varying vec3 vL;
      varying vec3 vN;
      varying vec3 vW;
      varying float vActive;
      ${COMMON}
      ${FOG_FRAG}
      void main() {
        vec3 V = normalize(cameraPosition - vW);
        vec3 col;
        if (vN.y > 0.5) {
          vec2 g = (vL.xz + 0.5) * vec2(11.0, 12.0);
          vec2 cell = floor(g);
          vec2 fg = max(fwidth(g), vec2(1e-5));
          vec2 e = (0.5 - abs(fract(g) - 0.5)) / fg;
          float gutter = aaLine(min(e.x, e.y), 0.6) * (1.0 - smoothstep(0.15, 0.4, max(fg.x, fg.y)));
          float r = hash12(cell);
          float ndv = clamp(dot(vN, V), 0.0, 1.0);
          vec3 film = 0.5 + 0.5 * cos(6.2832 * (vec3(0.0, 0.33, 0.67) + ndv * 1.4 + r * 0.3));
          col = vec3(0.024, 0.03, 0.04) * (0.85 + 0.3 * r) + film * 0.016;
          col *= 1.0 - 0.35 * gutter;
          vec2 fe = max(fwidth(vL.xz), vec2(1e-5));
          vec2 ee = (0.5 - abs(vL.xz)) / fe;
          col += vec3(0.3, 0.42, 0.55) * 0.18 * aaLine(min(ee.x, ee.y), 0.6);
          // The powered die lights the stack above it.
          vec2 c = vL.xz * 2.0;
          float pool = 1.0 - smoothstep(0.2, 1.1, length(c));
          col += vec3(0.29, 0.38, 0.49) * vActive * (0.05 + 0.05 * r + 0.08 * pool);
          vec3 H = normalize(uLight + V);
          col += vec3(0.5, 0.6, 0.75) * pow(max(dot(vN, H), 0.0), 60.0) * 0.4;
        } else {
          col = vec3(0.025, 0.03, 0.04);
        }
        col = mix(col, fogColor, fogAmount());
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
}

// ---------------------------------------------------------------------------------------------
// Scale-up switch and scale-out spine: dark glass bars with a light filament whose brightness and
// flow follow the traffic through them.

export function busMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: { ...fogUniforms(), uTime: { value: 0 } },
    fog: true,
    vertexShader: /* glsl */ `
      attribute float aLevel;
      varying vec3 vL;
      varying vec3 vN;
      varying vec3 vW;
      varying vec3 vC;
      varying vec3 vSize;
      varying float vLevel;
      #include <fog_pars_vertex>
      void main() {
        vL = position;
        vN = normal;
        vLevel = aLevel;
        vSize = ${INSTANCE_SIZE};
        vec4 wp = modelMatrix * instanceMatrix * vec4(position, 1.0);
        vW = wp.xyz;
        vC = (modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
        vec4 mvPosition = viewMatrix * wp;
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uTime;
      varying vec3 vL;
      varying vec3 vN;
      varying vec3 vW;
      varying vec3 vC;
      varying vec3 vSize;
      varying float vLevel;
      ${COMMON}
      ${FOG_FRAG}
      void main() {
        bool alongX = vSize.x >= vSize.z;
        vec3 axis = alongX ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0);
        float len = alongX ? vSize.x : vSize.z;
        float thick = min(vSize.y, alongX ? vSize.z : vSize.x);
        vec3 V = normalize(vW - cameraPosition);
        // A filament on the bar's axis, seen through dark glass: distance from the view ray to the axis.
        vec3 cr = cross(V, axis);
        float d = abs(dot(vW - vC, cr)) / max(length(cr), 1e-4);
        float along = dot(vW - vC, axis);
        float lvl = clamp(vLevel, 0.0, 1.0);
        float core = exp(-pow(d / (thick * 0.11), 2.0));
        float halo = exp(-pow(d / (thick * 0.4), 2.0));
        float flow = 0.5 + 0.5 * sin(along * 9.0 - uTime * (1.5 + 6.0 * lvl));
        float fres = pow(1.0 - abs(dot(vN, -V)), 2.5);
        vec3 col = vec3(0.012, 0.022, 0.034) + vec3(0.3, 0.45, 0.65) * fres * 0.12;
        vec3 light = vec3(0.7, 0.88, 1.0);
        col += light * (core * (0.22 + lvl * (0.9 + 0.8 * flow)) + halo * (0.03 + 0.12 * lvl));
        // Machined edges catch a little light; the ends stay dark.
        vec2 fc = abs(vN.y) > 0.5 ? vL.xz : abs(vN.z) > 0.5 ? vL.xy : vL.zy;
        vec2 fwc = max(fwidth(fc), vec2(1e-5));
        vec2 de = (0.5 - abs(fc)) / fwc;
        col += vec3(0.35, 0.5, 0.68) * 0.14 * aaLine(min(de.x, de.y), 0.6);
        col *= 1.0 - smoothstep(0.42, 0.5, abs(along) / len) * 0.6;
        col *= 1.0 - fogAmount();
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
}

// ---------------------------------------------------------------------------------------------
// Floor decals: rounded outlines for scale-up domains (racks) and parallel groups, drawn flat.
// Style 0 = domain, 1 = replica, 2 = prefill instance (dashed), 3 = decode instance.

export function decalMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: { ...fogUniforms() },
    fog: true,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      attribute float aStyle;
      attribute vec3 aColor;
      varying vec2 vP;
      varying vec2 vSize;
      varying float vStyle;
      varying vec3 vColor;
      #include <fog_pars_vertex>
      void main() {
        vec3 s = ${INSTANCE_SIZE};
        vSize = s.xz;
        vP = position.xz * s.xz;
        vStyle = aStyle;
        vColor = aColor;
        vec4 mvPosition = viewMatrix * modelMatrix * instanceMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec2 vP;
      varying vec2 vSize;
      varying float vStyle;
      varying vec3 vColor;
      ${COMMON}
      ${FOG_FRAG}
      void main() {
        float r = vStyle < 0.5 ? 0.35 : 0.18;
        vec2 hs = vSize * 0.5;
        vec2 q = abs(vP) - (hs - r);
        float sd = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
        float px = max(fwidth(sd), 1e-5);
        float line = aaLine(abs(sd) / px, 0.6);
        // Corner brackets: brighter near the corners.
        vec2 nearC = step(hs - vec2(0.45), abs(vP));
        float corner = nearC.x * nearC.y;
        float a;
        if (vStyle < 0.5) {
          // Domain: faint floor tint with a dot grid and a soft border.
          vec2 g = vP / 0.2;
          vec2 fg = max(fwidth(g), vec2(1e-5));
          float dots = aaLine(length((fract(g) - 0.5) / fg), 0.7) * (1.0 - smoothstep(0.08, 0.25, fg.x));
          float inner = 1.0 - smoothstep(-px, px, sd);
          a = 0.012 * inner + 0.05 * dots * inner + line * (0.12 + 0.2 * corner);
        } else {
          float dash = 1.0;
          if (vStyle > 1.5 && vStyle < 2.5) {
            float s = abs(q.x) > abs(q.y) ? vP.y : vP.x;
            dash = step(0.45, fract(s / 0.16));
          }
          a = line * dash * (0.32 + 0.55 * corner);
        }
        vec3 col = vColor * a * (1.0 - fogAmount());
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
}

// ---------------------------------------------------------------------------------------------
// Floor: tile seams of a raised data-centre floor under a soft pool of light. Additive over the
// reflective (or plain) floor slab.

export function floorGridMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: { uCenter: { value: new THREE.Vector2() }, uRadius: { value: 10 } },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      varying vec2 vW;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vW = wp.xz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec2 uCenter;
      uniform float uRadius;
      varying vec2 vW;
      ${COMMON}
      void main() {
        vec2 g = vW / 0.71;
        vec2 fg = max(fwidth(g), vec2(1e-5));
        vec2 e = (0.5 - abs(fract(g) - 0.5)) / fg;
        float seam = aaLine(min(e.x, e.y), 0.5) * (1.0 - smoothstep(0.1, 0.35, max(fg.x, fg.y)));
        float d = length(vW - uCenter) / uRadius;
        float pool = exp(-d * d * 1.6);
        vec3 col = vec3(0.32, 0.45, 0.6) * (0.035 * seam * (0.3 + 0.7 * pool)) + vec3(0.05, 0.09, 0.14) * pool * 0.32;
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
}

// ---------------------------------------------------------------------------------------------
// Sky: a dark dome, a touch lighter at the horizon so the floor's edge dissolves into it.

export function skyMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: { uHorizon: { value: new THREE.Color() }, uZenith: { value: new THREE.Color() } },
    side: THREE.BackSide,
    depthWrite: false,
    vertexShader: /* glsl */ `
      varying vec3 vDir;
      void main() {
        vDir = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uHorizon;
      uniform vec3 uZenith;
      varying vec3 vDir;
      void main() {
        float h = clamp(vDir.y, 0.0, 1.0);
        gl_FragColor = vec4(mix(uHorizon, uZenith, pow(h, 0.55)), 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
}

// ---------------------------------------------------------------------------------------------
// Traffic comets: screen-aligned streaks travelling along quadratic Bézier links. A per-link
// intensity texture sets how many of a link's comets are visible, so live utilization changes cost
// one tiny upload rather than a rebuild.

export function cometMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uViewport: { value: new THREE.Vector2(1, 1) },
      uWidth: { value: 0.024 },
      uIntensity: { value: null as THREE.DataTexture | null },
      uLinks: { value: 1 },
      uGain: { value: 2.2 },
    },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      attribute vec3 aFrom;
      attribute vec3 aCtrl;
      attribute vec3 aTo;
      attribute float aPhase;
      attribute float aSpeed;
      attribute float aTrail;
      attribute vec3 aColor;
      attribute float aLink;
      attribute float aRank;
      uniform sampler2D uIntensity;
      uniform float uLinks;
      uniform float uTime;
      uniform float uWidth;
      uniform vec2 uViewport;
      varying vec3 vColor;
      varying vec2 vUv;
      varying float vLenPx;
      varying float vWidth;
      varying float vFade;
      vec3 bez(float t) {
        float s = 1.0 - t;
        return s * s * aFrom + 2.0 * s * t * aCtrl + t * t * aTo;
      }
      void main() {
        float k = texture2D(uIntensity, vec2((aLink + 0.5) / uLinks, 0.5)).r;
        float t = fract(uTime * aSpeed + aPhase);
        vec4 ch = projectionMatrix * viewMatrix * vec4(bez(t), 1.0);
        vec4 ct = projectionMatrix * viewMatrix * vec4(bez(max(0.0, t - aTrail)), 1.0);
        if (aRank >= k || ch.w <= 0.05 || ct.w <= 0.05) {
          gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
          return;
        }
        vec2 half_ = 0.5 * uViewport;
        vec2 h = ch.xy / ch.w * half_;
        vec2 tl = ct.xy / ct.w * half_;
        vec2 d = h - tl;
        float len = length(d);
        vec2 dir = len > 1e-3 ? d / len : vec2(1.0, 0.0);
        vec2 nrm = vec2(-dir.y, dir.x);
        float width = clamp(uWidth * projectionMatrix[1][1] * half_.y / ch.w, 1.0, 5.0);
        float x = position.x;
        vec2 sp = mix(tl, h, x) + dir * (x * 2.0 - 1.0) * width + nrm * position.y * width;
        float z = mix(ct.z / ct.w, ch.z / ch.w, x);
        gl_Position = vec4(sp / half_, z, 1.0);
        vUv = vec2(x, position.y);
        vLenPx = len;
        vWidth = width;
        vColor = aColor;
        vFade = smoothstep(0.0, 0.1, t) * smoothstep(1.0, 0.85, t);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uGain;
      varying vec3 vColor;
      varying vec2 vUv;
      varying float vLenPx;
      varying float vWidth;
      varying float vFade;
      void main() {
        // Position along the streak in pixels, from the tail (0) to the head (vLenPx), caps included.
        float total = vLenPx + 2.0 * vWidth;
        float along = vUv.x * total - vWidth;
        float across = abs(vUv.y);
        float body = (1.0 - smoothstep(0.15, 1.0, across)) * pow(clamp(along / max(vLenPx, 1.0), 0.0, 1.0), 1.6);
        float dh = length(vec2((along - vLenPx) / vWidth, vUv.y));
        float head = exp(-dh * dh * 2.2);
        float i = (body * 0.65 + head * 1.5) * vFade * uGain;
        gl_FragColor = vec4(vColor * i, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
}
