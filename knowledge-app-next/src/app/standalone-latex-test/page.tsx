'use client';

import React, { useState, useEffect } from 'react';
import { MathJax, MathJaxContext } from 'better-react-mathjax';

// MathJax configuration (same as in math-rendering-test/page.tsx)
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
  svg: {
    fontCache: 'global',
    displayAlign: 'left',
    displayIndent: '0'
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
    ignoreHtmlClass: 'tex2jax_ignore',
    processHtmlClass: 'tex2jax_process'
  },
  startup: {
    typeset: false,
    ready() {
      console.log('✅ MathJax loaded and ready');
      (window as any).MathJax.startup.defaultReady();
      (window as any).mathJaxReady = true;
    }
  }
};

const StandaloneLatexTestPage: React.FC = () => {
  const [status, setStatus] = useState<string>('⏳ Waiting for MathJax to load...');
  const [physicsResult, setPhysicsResult] = useState<string>('<strong>Processed (Click Test to see):</strong>');
  const [waveResult, setWaveResult] = useState<string>('<strong>Processed:</strong>');
  const [integralResult, setIntegralResult] = useState<string>('<strong>Processed:</strong>');
  const [chemistryResult, setChemistryResult] = useState<string>('<strong>Processed:</strong>');

  const updateStatus = (message: string, type: 'success' | 'error' = 'success') => {
    setStatus(`<div class="status ${type}">${message}</div>`);
  };

  // Our enhanced LaTeX processor function (same as in the app)
  const processLatexText = (text: string): string => {
    if (!text) return text;

    console.log("🔧 Processing LaTeX for:", text.substring(0, Math.min(text.length, 100)) + "...");

    // Protect existing LaTeX expressions
    const protectedLatex: { placeholder: string; content: string }[] = [];
    let protectedIndex = 0;
    
    text = text.replace(/\$\$([^$]+)\$\$/g, (match, content) => {
        const placeholder = `__PROTECTED_DISPLAY_${protectedIndex++}__`;
        protectedLatex.push({ placeholder, content: `$$${content}$$` });
        return placeholder;
    });
    
    text = text.replace(/\$([^$]+)\$/g, (match, content) => {
        const placeholder = `__PROTECTED_INLINE_${protectedIndex++}__`;
        protectedLatex.push({ placeholder, content: `$${content}$` });
        return placeholder;
    });

    // Handle complex fractions with parentheses and exponents
    text = text.replace(/\(([a-zA-Z_0-9]+)\s*\/\s*([a-zA-Z_0-9]+)\)\s*\^\s*\(([^)]+)\)/g, '$$\\left(\\frac{$1}{$2}\\right)^{$3}$$');
    
    // Handle integrals
    text = text.replace(/∫\s*([^d]+)\s*d\s*([a-zA-Z])/g, '$$\\int $1 \, d$2$$');
    text = text.replace(/∫₀\^∞/g, '$$\\int_0^{\\infty}$$');
    text = text.replace(/∫₀\^\\infty/g, '$$\\int_0^{\\infty}$$');
    
    // Handle exponentials
    text = text.replace(/e\^(-([^)]+))/g, '$$e^{-$1}$$');
    text = text.replace(/e\^([^,\s\)]+)/g, '$$e^{$1}$$');
    
    // Handle fractions
    text = text.replace(/(\d+(?:\.\d+)?)\s*\/\s*([a-zA-Z_0-9]+)/g, '$$\\frac{$1}{$2}$$');
    text = text.replace(/([a-zA-Z_0-9]+)\s*\/\s*([a-zA-Z_0-9]+)/g, '$$\\frac{$1}{$2}$$');
    
    // Handle subscripts and superscripts
    text = text.replace(/([a-zA-Z])₀/g, '$1_0');
    text = text.replace(/([a-zA-Z])₁/g, '$1_1');
    text = text.replace(/([a-zA-Z])₂/g, '$1_2');
    text = text.replace(/([a-zA-Z])₃/g, '$1_3');
    text = text.replace(/([a-zA-Z])₄/g, '$1_4');
    
    text = text.replace(/([a-zA-Z0-9])² /g, '$1^2');
    text = text.replace(/([a-zA-Z0-9])³ /g, '$1^3');
    text = text.replace(/([a-zA-Z0-9])⁴ /g, '$1^4');
    
    // Handle variables with subscripts
    text = text.replace(/\b([a-zA-Z])_([a-zA-Z0-9]+)\b/g, '$$\\ $1_{$2}$$');
    
    // Handle Greek letters
    text = text.replace(/ψ/g, '$$\\psi$$');
    text = text.replace(/π/g, '$$\\pi$$');
    text = text.replace(/√/g, '$$\\sqrt$$');
    
    // Handle mathematical operators
    text = text.replace(/×/g, '$$\\times$$');
    text = text.replace(/∞/g, '$$\\infty$$');
    
    // Handle scientific notation
    text = text.replace(/(\d+(?:\.\d+)?)\s*[×xX]\s*10\^([+-]?\d+)/g, '$$\\ $1 \\times 10^{$2}$$');
    
    // Handle units
    text = text.replace(/(\d+(?:\.\d+)?)\s*kJ\/mol/g, '$$\\ $1 \\text{ kJ/mol}$$');
    text = text.replace(/(\d+(?:\.\d+)?)\s*K\b/g, '$$\\ $1 \\text{ K}$$');
    text = text.replace(/(\d+(?:\.\d+)?)\s*m\b/g, '$$\\ $1 \\text{ m}$$');
    
    // Handle chemical formulas
    text = text.replace(/\bH2O\b/g, '$$\\text{H}_2\\text{O}$$');
    text = text.replace(/\bH2\b/g, '$$\\text{H}_2$$');
    text = text.replace(/\bO2\b/g, '$$\\text{O}_2$$');
    
    // Clean up
    text = text.replace(/\$\$\s*\$\$\s*/g, '');
    text = text.replace(/\$\$([^$]*)\$\$\s*\$\$([^$]*)\$\$\s*/g, '$$\\ $1 $2$$');
    text = text.replace(/\$\$\s*\$\$\s*/g, '');
    text = text.replace(/\$\s*\$\s*/g, '');
    
    // Restore protected LaTeX
    protectedLatex.forEach(({ placeholder, content }) => {
        text = text.replace(placeholder, content);
    });

    console.log("✅ LaTeX processing completed");
    return text;
  };

  const testLaTeX = () => {
    if (!(window as any).MathJax || !(window as any).mathJaxReady) {
      updateStatus('❌ MathJax not ready yet, please wait...', 'error');
      return;
    }

    updateStatus('🧪 Testing LaTeX processing...', 'success');

    const testCases = [
      {
        id: 'physics-result',
        text: "Consider a hydrogen atom in its ground state, with principal quantum number n = 1 and orbital angular momentum quantum number l = 0. Calculate the expectation value of the radial probability density for an electron in this atom using the formula: ⟨r⟩ = 2/a₀ ∫₀^∞ r⁴ e^(-2r/a₀) dr, where a0 is the Bohr radius (approximately 5.29 × 10^{-11} m)."
      },
      {
        id: 'wave-result',
        text: "ψ(r) = 1/√π (a₀/r)^(3/2) e^(-r/a₀)"
      },
      {
        id: 'integral-result',
        text: "∫ r |ψ(r)|^2 d³r"
      },
      {
        id: 'chemistry-result',
        text: "The reaction H2 + O2 → H2O releases 463 kJ/mol at T = 298 K"
      }
    ];

    testCases.forEach(({ id, text }) => {
      const processed = processLatexText(text);
      if (id === 'physics-result') setPhysicsResult(`<strong>Processed:</strong> <MathJax>${processed}</MathJax>`);
      if (id === 'wave-result') setWaveResult(`<strong>Processed:</strong> <MathJax>${processed}</MathJax>`);
      if (id === 'integral-result') setIntegralResult(`<strong>Processed:</strong> <MathJax>${processed}</MathJax>`);
      if (id === 'chemistry-result') setChemistryResult(`<strong>Processed:</strong> <MathJax>${processed}</MathJax>`);
    });

    // Trigger MathJax typesetting after content is updated
    if ((window as any).MathJax && (window as any).MathJax.typesetPromise) {
      (window as any).MathJax.typesetPromise().then(() => {
        updateStatus('✅ LaTeX rendering completed successfully!', 'success');
      }).catch((err: any) => {
        updateStatus(`❌ MathJax rendering failed: ${err.message}`, 'error');
      });
    }
  };

  const clearTests = () => {
    location.reload();
  };

  useEffect(() => {
    // Check MathJax loading
    setTimeout(() => {
      if ((window as any).MathJax && (window as any).mathJaxReady) {
        updateStatus('✅ MathJax loaded and ready for testing!', 'success');
      } else {
        updateStatus('⏳ Still loading MathJax...', 'error');
        // Keep checking
        const checkInterval = setInterval(() => {
          if ((window as any).MathJax && (window as any).mathJaxReady) {
            updateStatus('✅ MathJax loaded and ready for testing!', 'success');
            clearInterval(checkInterval);
          }
        }, 500);
      }
    }, 1000);
  }, []);

  return (
    <MathJaxContext config={mathJaxConfig}>
      <div className="standalone-latex-body">
        <h1>🧮 LaTeX Rendering Test - Knowledge App Style</h1>
        <p>Testing the same MathJax configuration used in the Knowledge App with complex mathematical expressions.</p>
        
        <div id="status" className="status" dangerouslySetInnerHTML={{ __html: status }} />
        
        <button className="test-button" onClick={testLaTeX}>🧪 Test LaTeX Rendering</button>
        <button className="test-button" onClick={clearTests}>🧹 Clear All</button>
        
        <div className="test-case">
            <h3>🔬 Physics Expression (from your screenshot)</h3>
            <div className="original">
                <strong>Original:</strong> Consider a hydrogen atom in its ground state, with principal quantum number n = 1 and orbital angular momentum quantum number l = 0. Calculate the expectation value of the radial probability density for an electron in this atom using the formula: ⟨r⟩ = 2/a₀ ∫₀^∞ r⁴ e^(-2r/a₀) dr, where a0 is the Bohr radius (approximately 5.29 × 10^{-11} m).
            </div>
            <div className="processed" id="physics-result" dangerouslySetInnerHTML={{ __html: physicsResult }} />
        </div>
        
        <div className="test-case">
            <h3>🌊 Wave Function</h3>
            <div className="original">
                <strong>Original:</strong> ψ(r) = 1/√π (a₀/r)^(3/2) e^(-r/a₀)
            </div>
            <div className="processed" id="wave-result" dangerouslySetInnerHTML={{ __html: waveResult }} />
        </div>
        
        <div className="test-case">
            <h3>🔢 Complex Integral</h3>
            <div className="original">
                <strong>Original:</strong> ∫ r |ψ(r)|^2 d³r
            </div>
            <div className="processed" id="integral-result" dangerouslySetInnerHTML={{ __html: integralResult }} />
        </div>
        
        <div className="test-case">
            <h3>⚗️ Chemistry</h3>
            <div className="original">
                <strong>Original:</strong> The reaction H2 + O2 → H2O releases 463 kJ/mol at T = 298 K
            </div>
            <div className="processed" id="chemistry-result" dangerouslySetInnerHTML={{ __html: chemistryResult }} />
        </div>
      </div>
    </MathJaxContext>
  );
};

export default StandaloneLatexTestPage;