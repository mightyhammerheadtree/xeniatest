/**
 * Dawn-backed WGSL validation for tests.
 *
 * Node has no `navigator.gpu`, so tests cannot construct a real pipeline —
 * but they CAN compile the shader with the real WebGPU compiler (Tint, via
 * the `webgpu` package's Dawn build) and read structured diagnostics from it.
 * This helper does exactly that: any diagnostic of type "error" makes the
 * test fail with the full list and the offending shader.
 *
 * NOTE (empirically established, do not "optimize" into a cached device):
 * dawn.node's null backend is not safe to hold across awaits in module
 * scope — a cached/global GPUDevice intermittently aborts or segfaults the
 * process once other ESM modules are loaded. The only proven-stable shape is
 * a single linear function that creates the device, validates, and drops
 * everything in the same scope. The null backend is cheap enough to do that
 * per call.
 */
import { create, globals } from 'webgpu';

Object.assign(globalThis, globals);

/** Compile `code` with Dawn; throw with the full diagnostic list on any error. */
export async function assertValidWgsl(code) {
  const gpu = create(['backend=null']);
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('Dawn (null backend) returned no adapter');
  const device = await adapter.requestDevice();
  if (!device) throw new Error('Dawn (null backend) returned no device');
  const module = device.createShaderModule({ code });
  const info = await module.getCompilationInfo();
  const errors = (info?.messages ?? []).filter((m) => m.type === 'error');
  try {
    device.destroy();
  } catch {
    // best-effort; the null backend has nothing real to free
  }
  if (errors.length > 0) {
    const diagnostics = errors
      .map((m) => `  :${m.lineNum}:${m.linePos} ${m.type}: ${m.message}`)
      .join('\n');
    throw new Error(`Dawn rejected the generated WGSL:\n${diagnostics}\n\n${code}`);
  }
  return true;
}
