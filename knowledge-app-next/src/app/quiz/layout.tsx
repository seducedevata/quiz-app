import { MathJaxContext } from 'better-react-mathjax';

const mathJaxConfig = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true,
    packages: {'[+]': ['noerrors', 'textmacros', 'ams', 'newcommand', 'mathtools']},
    tags: 'none',
    macros: {
      degree: '^{\\circ}',
      angstrom: '\\text{Å}',
      celsius: '^{\\circ}\\text{C}',
      kelvin: '\\text{K}',
      joule: '\\text{J}',
      kilojoule: '\\text{kJ}',
      mol: '\\text{mol}',
      electronvolt: '\\text{eV}',
      meter: '\\text{m}',
      nanometer: '\\text{nm}',
      picometer: '\\text{pm}',
      bohr: 'a_0'
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
    typeset: false
  }
};

export default function QuizLayout({ children }: { children: React.ReactNode }) {
  return (
    <MathJaxContext config={mathJaxConfig}>
      {children}
    </MathJaxContext>
  );
}
