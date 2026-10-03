import { calibrationFor } from '../hardware/calib.ts';
import { totalGpus } from '../hardware/clusters.ts';
import type {
  Calibration,
  ClusterSpec,
  Deployment,
  InstanceMemory,
  Issue,
  ModelSpec,
  Placement,
  ResolvedInstance,
} from '../types.ts';
import { fmtBytes, fmtTokens } from '../units.ts';
import { resolveInstance } from '../vllm/resolve.ts';
import { instanceMemory, suggestMaxModelLen } from './memory.ts';
import { placeInstance } from './placement.ts';

export interface InstanceEval {
  inst: ResolvedInstance;
  placement: Placement;
  memory: InstanceMemory;
}

export interface Evaluation {
  model: ModelSpec;
  cluster: ClusterSpec;
  calib: Calibration;
  instances: InstanceEval[];
  usedGpus: number;
  issues: Issue[];
}

export function evaluate(model: ModelSpec, cluster: ClusterSpec, deployment: Deployment, calibOverride?: Partial<Calibration>): Evaluation {
  const calib = { ...calibrationFor(cluster.gpu.arch), ...calibOverride };
  const issues: Issue[] = [];
  const instances: InstanceEval[] = [];
  let used = 0;
  for (const spec of deployment.instances) {
    const inst = resolveInstance(spec, model, cluster);
    const placement = placeInstance(model, inst, cluster);
    const memory = instanceMemory(model, inst, placement, cluster, calib);
    const neg = memory.perRank.find((m) => m.requested - m.weights - m.nonTorch - m.persistent - m.actPeak - m.cudagraph <= 0);
    if (neg) {
      inst.issues.push({
        level: 'error',
        msg: `Weights (${fmtBytes(neg.weights)}) plus overheads exceed the ${fmtBytes(neg.requested)} budget on GPU ${neg.gpu}: no room for KV cache. Increase TP/PP/EP or use a quantized checkpoint.`,
      });
    } else if (!memory.fits) {
      const s = suggestMaxModelLen(model, inst, placement, memory);
      inst.issues.push({
        level: 'error',
        flag: 'maxModelLen',
        msg: `A single ${fmtTokens(inst.maxModelLen)}-token request needs more KV than is available. vLLM refuses to start; try --max-model-len ${s}.`,
      });
    }
    used = Math.max(used, inst.gpuStart + inst.world);
    instances.push({ inst, placement, memory });
  }
  if (used > totalGpus(cluster)) {
    issues.push({ level: 'error', msg: `Deployment needs ${used} GPUs but the cluster has ${totalGpus(cluster)}.` });
  }
  return { model, cluster, calib, instances, usedGpus: used, issues };
}

export function allIssues(e: Evaluation): Issue[] {
  return [...e.issues, ...e.instances.flatMap((i) => i.inst.issues)];
}
