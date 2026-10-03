// Dev helper: print a URL hash for a partial input set, e.g.
// node scripts/hash.ts '{"modelRepo":"deepseek-ai/DeepSeek-V3","clusterPreset":"4x8xh200","flags":{"tp":8,"dp":4,"ep":true}}'
import { CLUSTER_PRESETS } from '../src/core/hardware/clusters.ts';
const p = JSON.parse(process.argv[2] ?? '{}');
if (p.clusterPreset && !p.cluster) p.cluster = CLUSTER_PRESETS.find((c) => c.id === p.clusterPreset)?.build;
console.log(Buffer.from(JSON.stringify(p)).toString('base64').replace(/=+$/, ''));
