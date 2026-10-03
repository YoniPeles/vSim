import * as THREE from 'three';

// Layer plates: top face split into attention (front third) and FFN (rest); MoE FFNs draw one stripe
// per resident expert, brightened by per-instance heat (expert load).
export function plateMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uLight: { value: new THREE.Vector3(0.4, 1, 0.6).normalize() },
      uExperts: { value: null as THREE.DataTexture | null },
      uExpertCount: { value: 1 },
      uExpertOn: { value: 0 },
    },
    vertexShader: /* glsl */ `
      attribute vec3 aAttn;
      attribute vec3 aFfn;
      attribute float aStripes;
      attribute float aHeat;
      attribute float aExpertLo;
      varying vec3 vN;
      varying float vExpertLo;
      varying vec2 vUv;
      varying vec3 vAttn;
      varying vec3 vFfn;
      varying float vStripes;
      varying float vHeat;
      void main() {
        vUv = uv;
        vAttn = aAttn;
        vFfn = aFfn;
        vStripes = aStripes;
        vHeat = aHeat;
        vExpertLo = aExpertLo;
        mat4 m = modelMatrix * instanceMatrix;
        vN = normalize(mat3(m) * normal);
        gl_Position = projectionMatrix * viewMatrix * m * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uLight;
      uniform sampler2D uExperts;
      uniform float uExpertCount;
      uniform float uExpertOn;
      varying float vExpertLo;
      varying vec3 vN;
      varying vec2 vUv;
      varying vec3 vAttn;
      varying vec3 vFfn;
      varying float vStripes;
      varying float vHeat;
      void main() {
        bool top = vN.y > 0.5;
        bool front = vN.z > 0.5;
        vec3 col = vFfn;
        if (top && vUv.y > 0.66) {
          col = vAttn;
        } else if (top || front) {
          if (vStripes > 0.5) {
            // One stripe per resident expert, anti-aliased and faded out when sub-pixel.
            float s = vUv.x * vStripes;
            float fw = fwidth(s);
            float d = abs(fract(s) - 0.5);
            float line = 1.0 - smoothstep(0.38 - fw, 0.38 + fw, 0.5 - d);
            float vis = 1.0 - smoothstep(0.25, 0.6, fw);
            col = mix(vFfn, vFfn * 0.4, line * vis);
            if (uExpertOn > 0.5) {
              // Live expert load from the simulator's sampled routing.
              float e = vExpertLo + floor(vUv.x * vStripes);
              float load = texture2D(uExperts, vec2((e + 0.5) / uExpertCount, 0.5)).r;
              col = mix(col, vec3(1.0, 0.5, 0.2) * 1.5, load * 0.9);
            }
          }
          if (front) col *= 0.82;
        } else {
          col = mix(vFfn, vAttn, step(0.66, vUv.x)) * 0.72;
        }
        col += vec3(1.0, 0.82, 0.45) * vHeat * 1.8;
        float lambert = 0.58 + 0.42 * max(dot(vN, uLight), 0.0);
        gl_FragColor = vec4(col * lambert, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
}

// Traffic particles travel along quadratic Bézier links; everything happens in the vertex shader.
export function particleMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uSize: { value: 70 },
      uPixelRatio: { value: 1 },
      uIntensity: { value: null as THREE.DataTexture | null },
      uLinks: { value: 1 },
    },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    toneMapped: false,
    vertexShader: /* glsl */ `
      attribute vec3 aFrom;
      attribute vec3 aCtrl;
      attribute vec3 aTo;
      attribute float aPhase;
      attribute float aSpeed;
      attribute vec3 aColor;
      attribute float aLink;
      attribute float aRank;
      uniform sampler2D uIntensity;
      uniform float uLinks;
      uniform float uTime;
      uniform float uSize;
      uniform float uPixelRatio;
      varying vec3 vColor;
      varying float vFade;
      void main() {
        float t = fract(uTime * aSpeed + aPhase);
        float u = 1.0 - t;
        vec3 p = u * u * aFrom + 2.0 * u * t * aCtrl + t * t * aTo;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;
        // Show only the first k share of this link particles.
        float k = texture2D(uIntensity, vec2((aLink + 0.5) / uLinks, 0.5)).r;
        gl_PointSize = aRank < k ? uSize * uPixelRatio / max(1.0, -mv.z) : 0.0;
        vColor = aColor;
        vFade = smoothstep(0.0, 0.08, t) * smoothstep(1.0, 0.85, t);
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec3 vColor;
      varying float vFade;
      void main() {
        vec2 d = gl_PointCoord - 0.5;
        float r = length(d);
        float a = smoothstep(0.5, 0.0, r);
        gl_FragColor = vec4(vColor * 3.0 * a * vFade, a * vFade);
        #include <colorspace_fragment>
      }
    `,
  });
}
