export class SynapticError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly details?: Readonly<Record<string, unknown>>,
  ) {
    super(message);
    this.name = "SynapticError";
  }
}

export function invariant(
  condition: unknown,
  message: string,
  code: string,
  details?: Readonly<Record<string, unknown>>,
): asserts condition {
  if (!condition) {
    throw new SynapticError(message, code, details);
  }
}
