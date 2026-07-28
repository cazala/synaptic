export type LegacyFormat = "synaptic-v1" | "synaptic2-engine";

export interface LegacyImportWarning {
  readonly code: string;
  readonly message: string;
  readonly source?: Readonly<Record<string, unknown>>;
}

export interface LegacyImportResult<T> {
  readonly value: T;
  readonly warnings: readonly LegacyImportWarning[];
}
