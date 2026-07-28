import { invariant } from "./errors.js";
import type {
  ModelDefinition,
  ModelSnapshot,
  Precision,
} from "./types.js";
import { validateDefinition } from "./validate.js";

export interface BufferManifest {
  readonly id: string;
  readonly role: "parameters";
  readonly dtype: Precision;
  readonly length: number;
  readonly byteLength: number;
  readonly checksum: string;
}

export interface ArtifactManifest {
  readonly format: "synaptic";
  readonly formatVersion: 1;
  readonly definition: ModelDefinition;
  readonly buffers: readonly BufferManifest[];
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

function encodeNumbers(values: Float32Array | Float64Array, precision: Precision): Uint8Array {
  const bytesPerElement = precision === "f32" ? 4 : 8;
  const bytes = new Uint8Array(values.length * bytesPerElement);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    invariant(value !== undefined && Number.isFinite(value), "Artifacts cannot contain non-finite parameters", "NON_FINITE_PARAMETER", {
      index,
    });
    if (precision === "f32") {
      view.setFloat32(index * bytesPerElement, value, true);
    } else {
      view.setFloat64(index * bytesPerElement, value, true);
    }
  }
  return bytes;
}

function decodeNumbers(bytes: Uint8Array, precision: Precision, length: number): Float32Array | Float64Array {
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

export function saveSnapshot(snapshot: ModelSnapshot): ArtifactBundle {
  validateDefinition(snapshot.definition);
  invariant(
    snapshot.parameters.length === snapshot.definition.topology.parameters.length,
    "Parameter count does not match the model definition",
    "PARAMETER_COUNT_MISMATCH",
  );
  const id = "parameters.bin";
  const bytes = encodeNumbers(snapshot.parameters, snapshot.definition.precision);
  const buffer: BufferManifest = {
    id,
    role: "parameters",
    dtype: snapshot.definition.precision,
    length: snapshot.parameters.length,
    byteLength: bytes.byteLength,
    checksum: checksum(bytes),
  };
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
  invariant(bundle.manifest.format === "synaptic", "Unknown artifact format", "UNKNOWN_ARTIFACT_FORMAT");
  invariant(bundle.manifest.formatVersion === 1, "Unsupported artifact version", "UNSUPPORTED_ARTIFACT_VERSION");
  validateDefinition(bundle.manifest.definition);
  const manifest = bundle.manifest.buffers.find((buffer) => buffer.role === "parameters");
  invariant(manifest, "Artifact is missing its parameter buffer", "MISSING_PARAMETER_BUFFER");
  const bytes = bundle.buffers.get(manifest.id);
  invariant(bytes, "Artifact sidecar is missing", "MISSING_BUFFER", { id: manifest.id });
  invariant(bytes.byteLength === manifest.byteLength, "Artifact byte length does not match", "BUFFER_SIZE_MISMATCH");
  invariant(checksum(bytes) === manifest.checksum, "Artifact checksum does not match", "BUFFER_CHECKSUM_MISMATCH");
  const parameters = decodeNumbers(bytes, manifest.dtype, manifest.length);
  invariant(
    parameters.length === bundle.manifest.definition.topology.parameters.length,
    "Artifact parameter count does not match its topology",
    "PARAMETER_COUNT_MISMATCH",
  );
  return { definition: bundle.manifest.definition, parameters };
}
