import { invariant } from "./errors.js";
import type {
  ModelDefinition,
  ModelCheckpoint,
  ModelSnapshot,
  Precision,
  RuntimeState,
} from "./types.js";
import { compilePlan } from "./plan.js";
import { validateCheckpoint } from "./runtime.js";
import { validateDefinition } from "./validate.js";

export type RuntimeBufferName =
  | "state"
  | "activation"
  | "previousActivation"
  | "derivative"
  | "eligibilityTrace"
  | "extendedEligibilityTrace"
  | "projectedError"
  | "gatedError"
  | "error";

export interface BufferManifest {
  readonly id: string;
  readonly role: "parameters" | "runtime" | "optimizer";
  readonly name?: string;
  readonly dtype: Precision;
  readonly length: number;
  readonly byteLength: number;
  readonly checksum: string;
}

export interface CheckpointManifest {
  readonly step: number;
  readonly randomSeed: number;
  readonly randomCounter: number;
}

export interface ArtifactManifest {
  readonly format: "synaptic";
  readonly formatVersion: 1;
  readonly definition: ModelDefinition;
  readonly buffers: readonly BufferManifest[];
  readonly checkpoint?: CheckpointManifest;
}

export interface ArtifactBundle {
  readonly manifest: ArtifactManifest;
  readonly buffers: ReadonlyMap<string, Uint8Array>;
}

function checksum(bytes: Uint8Array): string {
  let hash = 0x811c_9dc5;
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 0x0100_0193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function encodeNumbers(
  values: Float32Array | Float64Array,
  precision: Precision,
): Uint8Array {
  const bytesPerElement = precision === "f32" ? 4 : 8;
  const bytes = new Uint8Array(values.length * bytesPerElement);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    invariant(
      value !== undefined && Number.isFinite(value),
      "Artifacts cannot contain non-finite numeric values",
      "NON_FINITE_ARTIFACT_VALUE",
      { index },
    );
    if (precision === "f32") {
      view.setFloat32(index * bytesPerElement, value, true);
    } else {
      view.setFloat64(index * bytesPerElement, value, true);
    }
  }
  return bytes;
}

function decodeNumbers(
  bytes: Uint8Array,
  precision: Precision,
  length: number,
): Float32Array | Float64Array {
  const bytesPerElement = precision === "f32" ? 4 : 8;
  invariant(bytes.byteLength === length * bytesPerElement, "Artifact buffer length does not match its manifest", "BUFFER_SIZE_MISMATCH");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const result = precision === "f32" ? new Float32Array(length) : new Float64Array(length);
  for (let index = 0; index < length; index += 1) {
    result[index] = precision === "f32"
      ? view.getFloat32(index * bytesPerElement, true)
      : view.getFloat64(index * bytesPerElement, true);
  }
  return result;
}

function createBuffer(
  id: string,
  role: BufferManifest["role"],
  values: Float32Array | Float64Array,
  precision: Precision,
  name?: string,
): readonly [BufferManifest, Uint8Array] {
  const bytes = encodeNumbers(values, precision);
  return [{
    id,
    role,
    ...(name === undefined ? {} : { name }),
    dtype: precision,
    length: values.length,
    byteLength: bytes.byteLength,
    checksum: checksum(bytes),
  }, bytes];
}

function decodeBuffer(
  bundle: ArtifactBundle,
  manifest: BufferManifest,
): Float32Array | Float64Array {
  invariant(
    manifest.dtype === bundle.manifest.definition.precision,
    `Artifact buffer ${manifest.id} does not match the model precision`,
    "PRECISION_MISMATCH",
  );
  const bytes = bundle.buffers.get(manifest.id);
  invariant(bytes, "Artifact sidecar is missing", "MISSING_BUFFER", {
    id: manifest.id,
  });
  invariant(
    bytes.byteLength === manifest.byteLength,
    "Artifact byte length does not match",
    "BUFFER_SIZE_MISMATCH",
    { id: manifest.id },
  );
  invariant(
    checksum(bytes) === manifest.checksum,
    "Artifact checksum does not match",
    "BUFFER_CHECKSUM_MISMATCH",
    { id: manifest.id },
  );
  return decodeNumbers(bytes, manifest.dtype, manifest.length);
}

function validateManifest(bundle: ArtifactBundle): void {
  invariant(
    bundle.manifest.format === "synaptic",
    "Unknown artifact format",
    "UNKNOWN_ARTIFACT_FORMAT",
  );
  invariant(
    bundle.manifest.formatVersion === 1,
    "Unsupported artifact version",
    "UNSUPPORTED_ARTIFACT_VERSION",
  );
  validateDefinition(bundle.manifest.definition);
  const ids = new Set<string>();
  for (const buffer of bundle.manifest.buffers) {
    invariant(
      !ids.has(buffer.id),
      `Artifact buffer id ${buffer.id} is duplicated`,
      "DUPLICATE_BUFFER",
    );
    ids.add(buffer.id);
  }
}

export function saveSnapshot(snapshot: ModelSnapshot): ArtifactBundle {
  validateDefinition(snapshot.definition);
  invariant(
    snapshot.parameters.length === snapshot.definition.topology.parameters.length,
    "Parameter count does not match the model definition",
    "PARAMETER_COUNT_MISMATCH",
  );
  const id = "parameters.bin";
  const [buffer, bytes] = createBuffer(
    id,
    "parameters",
    snapshot.parameters,
    snapshot.definition.precision,
  );
  return {
    manifest: {
      format: "synaptic",
      formatVersion: 1,
      definition: snapshot.definition,
      buffers: [buffer],
    },
    buffers: new Map([[id, bytes]]),
  };
}

export function loadSnapshot(bundle: ArtifactBundle): ModelSnapshot {
  validateManifest(bundle);
  const parameterBuffers = bundle.manifest.buffers.filter(
    (buffer) => buffer.role === "parameters",
  );
  invariant(
    parameterBuffers.length === 1,
    "Artifact must contain exactly one parameter buffer",
    "INVALID_PARAMETER_BUFFERS",
  );
  const manifest = parameterBuffers[0];
  invariant(manifest, "Artifact is missing its parameter buffer", "MISSING_PARAMETER_BUFFER");
  const parameters = decodeBuffer(bundle, manifest);
  invariant(
    parameters.length === bundle.manifest.definition.topology.parameters.length,
    "Artifact parameter count does not match its topology",
    "PARAMETER_COUNT_MISMATCH",
  );
  return { definition: bundle.manifest.definition, parameters };
}

const RUNTIME_BUFFER_NAMES = [
  "state",
  "activation",
  "previousActivation",
  "derivative",
  "eligibilityTrace",
  "extendedEligibilityTrace",
  "projectedError",
  "gatedError",
  "error",
] as const satisfies readonly RuntimeBufferName[];

export function saveCheckpoint(checkpoint: ModelCheckpoint): ArtifactBundle {
  const plan = compilePlan(checkpoint.definition);
  validateCheckpoint(plan, checkpoint);
  const snapshotBundle = saveSnapshot(checkpoint);
  const manifests = [...snapshotBundle.manifest.buffers];
  const buffers = new Map(snapshotBundle.buffers);
  for (const name of RUNTIME_BUFFER_NAMES) {
    const id = `runtime/${name}.bin`;
    const [manifest, bytes] = createBuffer(
      id,
      "runtime",
      checkpoint.runtime[name],
      checkpoint.definition.precision,
      name,
    );
    manifests.push(manifest);
    buffers.set(id, bytes);
  }
  for (const [name, values] of Object.entries(checkpoint.optimizer ?? {})) {
    invariant(name.length > 0, "Optimizer buffer names cannot be empty", "INVALID_BUFFER_NAME");
    invariant(
      checkpoint.definition.precision === "f32"
        ? values instanceof Float32Array
        : values instanceof Float64Array,
      `Optimizer buffer ${name} has the wrong precision`,
      "PRECISION_MISMATCH",
    );
    const id = `optimizer/${encodeURIComponent(name)}.bin`;
    const [manifest, bytes] = createBuffer(
      id,
      "optimizer",
      values,
      checkpoint.definition.precision,
      name,
    );
    manifests.push(manifest);
    buffers.set(id, bytes);
  }
  return {
    manifest: {
      ...snapshotBundle.manifest,
      buffers: manifests,
      checkpoint: {
        step: checkpoint.runtime.step,
        randomSeed: checkpoint.runtime.randomSeed,
        randomCounter: checkpoint.runtime.randomCounter,
      },
    },
    buffers,
  };
}

export function loadCheckpoint(bundle: ArtifactBundle): ModelCheckpoint {
  validateManifest(bundle);
  const metadata = bundle.manifest.checkpoint;
  invariant(metadata, "Artifact does not contain checkpoint metadata", "NOT_A_CHECKPOINT");
  const snapshot = loadSnapshot(bundle);
  const runtime = {} as Record<RuntimeBufferName, Float32Array | Float64Array>;
  for (const name of RUNTIME_BUFFER_NAMES) {
    const matches = bundle.manifest.buffers.filter(
      (buffer) => buffer.role === "runtime" && buffer.name === name,
    );
    invariant(
      matches.length === 1,
      `Checkpoint must contain exactly one ${name} runtime buffer`,
      "INVALID_RUNTIME_BUFFERS",
      { name },
    );
    const manifest = matches[0];
    invariant(manifest, "Checkpoint runtime manifest is missing", "MISSING_RUNTIME_BUFFER");
    runtime[name] = decodeBuffer(bundle, manifest);
  }
  const optimizer: Record<string, Float32Array | Float64Array> = {};
  for (const manifest of bundle.manifest.buffers) {
    if (manifest.role !== "optimizer") {
      continue;
    }
    invariant(
      manifest.name !== undefined && manifest.name.length > 0,
      "Optimizer buffers must have names",
      "INVALID_BUFFER_NAME",
    );
    invariant(
      optimizer[manifest.name] === undefined,
      `Optimizer buffer ${manifest.name} is duplicated`,
      "DUPLICATE_BUFFER",
    );
    optimizer[manifest.name] = decodeBuffer(bundle, manifest);
  }
  const checkpoint: ModelCheckpoint = {
    ...snapshot,
    runtime: {
      ...runtime as Pick<RuntimeState, RuntimeBufferName>,
      step: metadata.step,
      randomSeed: metadata.randomSeed,
      randomCounter: metadata.randomCounter,
    },
    ...(Object.keys(optimizer).length === 0 ? {} : { optimizer }),
  };
  validateCheckpoint(compilePlan(checkpoint.definition), checkpoint);
  return checkpoint;
}
