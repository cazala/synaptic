export const tests = Object.freeze([
  {
    description: "Train a 2→3→1 network and watch all four truth-table predictions converge.",
    kind: "Interactive demo",
    name: "Learn XOR",
    path: "/packages/backend-webgpu/test/xor.html",
  },
  {
    description: "Train on MNIST, draw a digit, and inspect the network's ten output activations.",
    kind: "Interactive demo",
    name: "Draw a digit",
    path: "/packages/backend-webgpu/test/mnist.html",
  },
  {
    description: "Train direct and network MNIST rules, transfer both exactly, and compare them as Neural CAs.",
    kind: "Interactive demo",
    name: "MNIST to Neural CA",
    path: "/packages/backend-webgpu/test/mnist-automata.html",
  },
  {
    description: "Train growth, persistence, and repair through a 64–96-generation tape, then stream the artifact into Automata.",
    kind: "Interactive demo",
    name: "Growing Neural CA",
    path: "/packages/backend-webgpu/test/growing-neural-ca.html",
  },
  {
    description: "Train a four-cell LSTM, then animate random length-10 recall challenges.",
    kind: "Interactive demo",
    name: "Sequence recall",
    path: "/packages/backend-webgpu/test/dsr.html",
  },
  {
    description: "Continuously train a coordinate-to-RGB network to reconstruct a portrait.",
    kind: "Interactive demo",
    name: "Learn to paint",
    path: "/packages/backend-webgpu/test/learn-to-paint.html",
  },
  {
    description: "Compile the WGSL module and report browser validation messages.",
    kind: "Hardware check",
    name: "Shader compilation",
    path: "/packages/backend-webgpu/test/shader-smoke.html",
  },
  {
    description: "Run a minimal recurrent train, forward, snapshot, and checkpoint lifecycle.",
    kind: "Hardware check",
    name: "Forward and state smoke test",
    path: "/packages/backend-webgpu/test/forward-smoke.html",
  },
  {
    description: "Compare recurrent training, restoration, traces, and shared parameters with CPU.",
    kind: "Hardware check",
    name: "Training parity",
    path: "/packages/backend-webgpu/test/training-parity.html",
  },
  {
    description: "Verify Growing NCA loss descent and that Automata accepts the exported artifact.",
    kind: "Hardware check",
    name: "Growing NCA trainer",
    path: "/packages/backend-webgpu/test/growing-neural-ca-smoke.html",
  },
  {
    description: "Learn XOR, MNIST, and DSR on hardware and report accuracy and elapsed time.",
    kind: "Automated regression",
    name: "Learning workloads",
    path: "/packages/backend-webgpu/test/learning-workloads.html",
  },
]);

export function indexPage() {
  const links = tests.map((test) => `
    <article class="panel">
      <span>${test.kind}</span>
      <h2><a href="${test.path}">${test.name}</a></h2>
      <p>${test.description}</p>
      <code>${test.path}</code>
    </article>
  `).join("");
  return `<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synaptic WebGPU tests</title>
<link rel="stylesheet" href="/packages/backend-webgpu/test/demo.css">
<style>
  main { max-width: 74rem; }
  h1 {
    max-width: 11ch;
    margin: 0;
    font-size: clamp(3rem, 8vw, 6rem);
    letter-spacing: -.07em;
    line-height: .88;
  }
  .intro { max-width: 42rem; margin: 1.5rem 0 3rem; color: #9cabc0; line-height: 1.65; }
  .cards {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  article { padding: 1.35rem; }
  article span {
    color: #ff809f;
    font: 650 .6rem ui-monospace, monospace;
    letter-spacing: .09em;
    text-transform: uppercase;
  }
  h2 { margin: .55rem 0; font-size: 1.2rem; }
  h2 a { color: #edf4ff; text-decoration-color: #526983; text-underline-offset: .22em; }
  article p { min-height: 3.2em; margin: .65rem 0 1rem; color: #8999ae; line-height: 1.55; }
  code { color: #64758b; font-size: .68rem; overflow-wrap: anywhere; }
  @media (max-width: 680px) { .cards { grid-template-columns: 1fr; } }
</style>
<main>
  <h1>WebGPU, in motion.</h1>
  <p class="intro">
    Interactive learning demos and no-fallback hardware checks for Synaptic.
    Every page below compiles directly to WebGPU and must report
    <code>webgpu</code> as its backend.
  </p>
  <section class="cards">${links}</section>
</main>`;
}
