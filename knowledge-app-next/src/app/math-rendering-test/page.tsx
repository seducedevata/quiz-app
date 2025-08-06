'use client';

import React, { useEffect, useState } from 'react';
import { MathJax } from 'better-react-mathjax';

// KaTeX CSS - Assuming it's loaded globally or via a link tag in _document.js or similar
// For this component, we'll just ensure the link is mentioned for clarity.
// <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

// MathJax configuration
const mathJaxConfig = {
  tex: {
    inlineMath: [['$', '$'], ['\(', '\)']],
    displayMath: [['$$', '$$'], ['\[', '\]']],
    processEscapes: true,
    processEnvironments: true,
    packages: {'[+]': ['noerrors', 'textmacros', 'ams', 'newcommand', 'mathtools']},
    tags: 'none',
    macros: {
      degree: '^{\circ}',
      angstrom: '\text{Å}',
      celsius: '^{\circ}\text{C}',
      kelvin: '\text{K}',
      joule: '\text{J}',
      kilojoule: '\text{kJ}',
      mol: '\text{mol}',
      electronvolt: '\text{eV}',
      meter: '\text{m}',
      nanometer: '\text{nm}',
      picometer: '\text{pm}',
      bohr: 'a_0'
    },
    formatError: function (jax: any, error: any) {
      console.warn('MathJax error:', error.message);
      return ['span', {style: 'font-family: monospace; color: #333;'}, jax.math];
    }
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
    ignoreHtmlClass: 'tex2jax_ignore',
    processHtmlClass: 'tex2jax_process'
  },
  startup: {
    typeset: false,
    ready() {
      console.log('✅ MathJax loaded');
      (window as any).MathJax.startup.defaultReady();
      (window as any).mathJaxReady = true;
    }
  }
};

const MathRenderingTestPage: React.FC = () => {
  const [status, setStatus] = useState<string>('');

  const showStatus = (message: string, type: 'info' | 'success' | 'error' = 'info') => {
    setStatus(`<div class="${type === 'info' ? 'text-blue-700 bg-blue-50 border-blue-200' : type === 'success' ? 'text-green-700 bg-green-50 border-green-200' : 'text-red-700 bg-red-50 border-red-200'} p-3 rounded-md border">${message}</div>`);
    console.log(message);
  };

  const testMathJax = () => {
    showStatus('🔬 Testing MathJax rendering...', 'info');
    // In a real scenario, you'd trigger MathJax typesetting here if not auto-typesetting
    // For better-react-mathjax, it handles typesetting automatically within MathJax components.
    showStatus('✅ MathJax rendering simulated. Check rendered expressions.', 'success');
  };

  const testKaTeX = () => {
    showStatus('⚡ Testing KaTeX rendering...', 'info');
    // KaTeX rendering is typically done via auto-render or manual renderToString
    // For this replication, we assume auto-render is set up or will be triggered.
    if ((window as any).renderMathInElement) {
      try {
        (window as any).renderMathInElement(document.body, {
          delimiters: [
            {left: "$$", right: "$$", display: true},
            {left: "$", right: "$", display: false},
            {left: "\[", right: "\]", display: true},
            {left: "\(", right: "\)", display: false}
          ],
          throwOnError: false,
          errorColor: '#cc0000',
          strict: false,
          trust: true,
          macros: {
            "\degree": "^{\circ}",
            "\celsius": "^{\circ}\text{C}",
            "\kelvin": "\text{K}",
            "\joule": "\text{J}",
            "\kilojoule": "\text{kJ}",
            "\mol": "\text{mol}",
            "\electronvolt": "\text{eV}",
            "\meter": "\text{m}",
            "\nanometer": "\text{nm}",
            "\picometer": "\text{pm}"
          }
        });
        showStatus('✅ KaTeX rendering completed successfully!', 'success');
      } catch (err: any) {
        showStatus(`❌ KaTeX rendering failed: ${err.message}`, 'error');
        console.error('KaTeX error:', err);
      }
    } else {
      showStatus('❌ KaTeX auto-render not available. Ensure KaTeX scripts are loaded.', 'error');
    }
  };

  const testBoth = () => {
    showStatus('🚀 Testing both MathJax and KaTeX...', 'info');
    testMathJax();
    setTimeout(() => testKaTeX(), 500);
  };

  const clearAll = () => {
    // In a real app, this might clear rendered math or reload the component state
    // For this test, we'll just clear the status message.
    setStatus('');
    console.log('🧹 Cleared all test statuses.');
  };

  useEffect(() => {
    // Auto-test when page loads
    setTimeout(() => {
      showStatus('🔍 Page loaded, checking math libraries...', 'info');
      const checks = [];
      if ((window as any).MathJax) checks.push('✅ MathJax loaded');
      else checks.push('❌ MathJax not loaded');
      
      if ((window as any).katex) checks.push('✅ KaTeX loaded');
      else checks.push('❌ KaTeX not loaded');
      
      showStatus(checks.join('<br>'), 'info');
      
      // Auto-render with the available library
      if ((window as any).katex && (window as any).renderMathInElement) {
          testKaTeX();
      } else if ((window as any).MathJax && (window as any).MathJax.typesetPromise) {
          testMathJax();
      }
    }, 1000);
  }, []);

  return (
    <div className="font-sans max-w-screen-lg mx-auto p-5 leading-relaxed">
      <h1 className="text-3xl font-bold mb-4 text-gray-800">🧮 Math Rendering Test: MathJax vs KaTeX</h1>
      <p className="mb-6 text-gray-700">Testing mathematical expression rendering with both MathJax and KaTeX libraries.</p>

      <div className="mb-6">
        <button onClick={testMathJax} className="bg-blue-600 text-white px-4 py-2 m-1 rounded-md cursor-pointer hover:bg-blue-700 transition-colors">🔬 Test MathJax</button>
        <button onClick={testKaTeX} className="bg-blue-600 text-white px-4 py-2 m-1 rounded-md cursor-pointer hover:bg-blue-700 transition-colors">⚡ Test KaTeX</button>
        <button onClick={testBoth} className="bg-blue-600 text-white px-4 py-2 m-1 rounded-md cursor-pointer hover:bg-blue-700 transition-colors">🚀 Test Both</button>
        <button onClick={clearAll} className="bg-blue-600 text-white px-4 py-2 m-1 rounded-md cursor-pointer hover:bg-blue-700 transition-colors">🧹 Clear All</button>
      </div>

      <div className="bg-gray-50 p-5 my-4 border-2 border-gray-200 rounded-lg">
        <h3 className="text-xl font-semibold mb-3 text-blue-700 border-b-2 border-blue-700 pb-2">📐 Complex Physics Expression (from screenshot)</h3>
        <div className="bg-white p-4 my-2 border-l-4 border-green-500 rounded-md text-lg">
          <MathJax>
            Consider a hydrogen atom in its ground state, with principal quantum number n = 1 and orbital angular momentum quantum number l = 0. Calculate the expectation value of the radial probability density for an electron in this atom using the formula: ⟨r⟩ = $$rac{2}{a_0} 
int_0^{\infty} r^4 e^{-2r/a_0} dr$$, where a₀ is the Bohr radius (approximately $$5.29 \times 10^{-11}$$ m).
          </MathJax>
        </div>
      </div>

      <div className="bg-gray-50 p-5 my-4 border-2 border-gray-200 rounded-lg">
        <h3 className="text-xl font-semibold mb-3 text-blue-700 border-b-2 border-blue-700 pb-2">🌊 Wave Function</h3>
        <div className="bg-white p-4 my-2 border-l-4 border-green-500 rounded-md text-lg">
          <MathJax>
            $$\psi(r) = \frac{1}{\sqrt{\pi}} \left(\frac{a_0}{r}\right)^{3/2} e^{-r/a_0}$$
          </MathJax>
        </div>
      </div>

      <div className="bg-gray-50 p-5 my-4 border-2 border-gray-200 rounded-lg">
        <h3 className="text-xl font-semibold mb-3 text-blue-700 border-b-2 border-blue-700 pb-2">🔢 Complex Integral</h3>
        <div className="bg-white p-4 my-2 border-l-4 border-green-500 rounded-md text-lg">
          <MathJax>
            $$\int r |\psi(r)|^2 d^3r = \int_0^{\infty} r \cdot r^2 \left|\frac{1}{\sqrt{\pi}} \left(\frac{a_0}{r}\right)^{3/2} e^{-r/a_0}\right|^2 dr$$
          </MathJax>
        </div>
      </div>

      <div className="bg-gray-50 p-5 my-4 border-2 border-gray-200 rounded-lg">
        <h3 className="text-xl font-semibold mb-3 text-blue-700 border-b-2 border-blue-700 pb-2">⚗️ Chemistry Expression</h3>
        <div className="bg-white p-4 my-2 border-l-4 border-green-500 rounded-md text-lg">
          <MathJax>
            The reaction $$	ext{H}_2 + 	ext{O}_2 ightarrow 	ext{H}_2	ext{O}$$ releases $$463 	ext{ kJ/mol}$$ at $$T = 298 	ext{ K}$$. The enthalpy change is $$\Delta H = -463 	ext{ kJ/mol}$$.
          </MathJax>
        </div>
      </div>

      <div className="bg-gray-50 p-5 my-4 border-2 border-gray-200 rounded-lg">
        <h3 className="text-xl font-semibold mb-3 text-blue-700 border-b-2 border-blue-700 pb-2">🔬 Mixed Mathematical Expressions</h3>
        <div className="bg-white p-4 my-2 border-l-4 border-green-500 rounded-md text-lg">
          <MathJax>
            $$\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}$$, 
            $$e^{i\pi} + 1 = 0$$,
            $$\nabla^2 \psi + \frac{2m}{\hbar^2}(E - V)\psi = 0$$
          </MathJax>
        </div>
      </div>

      <div className="mt-6" dangerouslySetInnerHTML={{ __html: status }} />
    </div>
  );
};

export default MathRenderingTestPage;