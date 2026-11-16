/**
 * Xenia Canary - Web: static file server.
 *
 * Serves public/ on 0.0.0.0:$PORT (default 3000) with the headers a modern
 * emulator front-end needs:
 *   - Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy so the page
 *     is "crossOriginIsolated" (required for SharedArrayBuffer, which the
 *     threading milestone of the port will need),
 *   - Cross-Origin-Resource-Policy: same-origin,
 *   - a tight Content-Security-Policy (self + inline styles only),
 *   - long-lived caching for fingerprinted core assets, no-store for HTML.
 *
 * Nothing here buffers file bodies in memory beyond what compression does
 * per-response; large assets stream from disk.
 */
import express from 'express';
import compression from 'compression';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.join(__dirname, '..', 'public');
const PORT = Number(process.env.PORT || 3000);
const HOST = process.env.HOST || '0.0.0.0';

const app = express();
app.disable('x-powered-by');
app.use(compression());

// Security headers (applied to everything).
app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  res.setHeader(
    'Content-Security-Policy',
    [
      "default-src 'self'",
      "script-src 'self' 'wasm-unsafe-eval'",
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data:",
      "font-src 'self'",
      "connect-src 'self'",
      "worker-src 'self'",
      "manifest-src 'self'",
      "object-src 'none'",
      "base-uri 'self'",
      "form-action 'none'",
    ].join('; ')
  );
  next();
});

// Static assets. index.html is never cached; hashed assets get a long TTL.
const MIME = {
  '.wasm': 'application/wasm',
  '.webmanifest': 'application/manifest+json',
  '.mjs': 'text/javascript',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.json': 'application/json',
  '.ico': 'image/x-icon',
};

app.use(
  express.static(PUBLIC_DIR, {
    index: 'index.html',
    etag: true,
    setHeaders(res, filePath) {
      const ext = path.extname(filePath).toLowerCase();
      if (MIME[ext]) res.setHeader('Content-Type', MIME[ext]);
      if (ext === '.html') {
        res.setHeader('Cache-Control', 'no-store');
      } else if (filePath.includes(`${path.sep}core${path.sep}`)) {
        // The WASM core is content-hashed by release; cache hard.
        res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
      } else {
        res.setHeader('Cache-Control', 'public, max-age=3600');
      }
    },
  })
);

// SPA fallback (single-page app): anything unknown serves the shell.
app.get('*', (req, res) => {
  res.setHeader('Cache-Control', 'no-store');
  res.sendFile(path.join(PUBLIC_DIR, 'index.html'));
});

const server = app.listen(PORT, HOST, () => {
  console.log(`xenia-canary-web listening on http://${HOST}:${PORT}`);
  console.log(`crossOriginIsolated headers: COOP=same-origin COEP=require-corp`);
});

function shutdown(signal) {
  console.log(`\n${signal} received, shutting down gracefully`);
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(1), 5000).unref();
}
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));
