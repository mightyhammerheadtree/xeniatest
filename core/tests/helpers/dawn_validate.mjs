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

/**
 * Validate a render pipeline with the given color/depth attachment formats on
 * Dawn's null backend. Used to prove that the attachment formats this port
 * chooses (e.g. depth32float for the guest's kD24FS8 float depth) are legal
 * WebGPU, and that the depth-stencil descriptor we build for them validates.
 *
 * @returns {Promise<string|null>} null when the pipeline validates, else the
 *   full diagnostic message.
 */
export async function checkAttachmentFormats({ colorFormat = 'bgra8unorm', depthFormat, stencil = false }) {
  const gpu = create(['backend=null']);
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('Dawn (null backend) returned no adapter');
  const device = await adapter.requestDevice();
  device.pushErrorScope('validation');
  const vsModule = device.createShaderModule({
    code: '@vertex fn vs_main() -> @builtin(position) vec4f { return vec4f(0.0); }',
  });
  const fsModule = device.createShaderModule({
    code: '@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }',
  });
  const depthStencil = depthFormat
    ? {
      format: depthFormat,
      depthWriteEnabled: true,
      depthCompare: 'less',
      ...(stencil
        ? {
          stencilFront: { compare: 'always', failOp: 'keep', depthFailOp: 'keep', passOp: 'keep' },
          stencilBack: { compare: 'always', failOp: 'keep', depthFailOp: 'keep', passOp: 'keep' },
          stencilReadMask: 0xff,
          stencilWriteMask: 0xff,
        }
        : {}),
    }
    : undefined;
  device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: vsModule, entryPoint: 'vs_main' },
    fragment: { module: fsModule, entryPoint: 'fs_main', targets: [{ format: colorFormat }] },
    depthStencil,
    primitive: { topology: 'triangle-list' },
  });
  const err = await device.popErrorScope();
  try {
    device.destroy();
  } catch {
    // best-effort
  }
  return err ? String(err.message ?? err) : null;
}

/**
 * Create a render pipeline on the null backend from two candidate modules and
 * capture any interface/layout validation diagnostic via an error scope.
 * Fresh device, destroyed in the same linear scope — never held across calls.
 *
 * @returns {Promise<string|null>} null when the pipeline validates, else the
 *   full diagnostic message.
 */
export async function checkProgramLinks(vsCode, fsCode, vertexBuffers = [], format = 'bgra8unorm') {
  const gpu = create(['backend=null']);
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('Dawn (null backend) returned no adapter');
  const device = await adapter.requestDevice();
  device.pushErrorScope('validation');
  const vsModule = device.createShaderModule({ code: vsCode });
  const fsModule = device.createShaderModule({ code: fsCode });
  device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: vsModule, entryPoint: 'vs_main', buffers: vertexBuffers },
    fragment: { module: fsModule, entryPoint: 'fs_main', targets: [{ format }] },
    // A fragment stage writing frag_depth is rejected without depth-stencil state (Dawn validation).
    depthStencil: /@builtin\(frag_depth\)/.test(fsCode)
      ? { format: 'depth24plus-stencil8', depthWriteEnabled: true, depthCompare: 'always' }
      : undefined,
    primitive: { topology: 'triangle-list' },
  });
  const err = await device.popErrorScope();
  try {
    device.destroy();
  } catch {
    // best-effort
  }
  return err ? String(err.message ?? err) : null;
}

