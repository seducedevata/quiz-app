#!/usr/bin/env python3
"""
Get the full PhD-level question from DeepSeek R1
"""

import requests
import json

def get_full_phd_question():
    """Get the complete PhD-level question from DeepSeek R1"""
    
    prompt = """Generate a truly PhD-level question about quantum mechanics and atomic structure that would challenge a graduate student or postdoc. 

The question should involve:
- Advanced quantum mechanical concepts (not basic energy level calculations)
- Multi-step reasoning requiring deep understanding
- Complex interactions between quantum phenomena
- Research-level knowledge

Examples of PhD-level topics:
- Quantum field theory applications to atomic physics
- Many-body quantum systems
- Advanced spectroscopy and selection rules
- Quantum entanglement in atomic systems
- Relativistic quantum mechanics effects
- Advanced perturbation theory
- Quantum phase transitions in atomic systems

Generate a calculation-based question that requires graduate-level knowledge.

Respond with JSON format:
{
  "question": "Your PhD-level question here",
  "options": ["option A", "option B", "option C", "option D"],
  "correct_answer": "A",
  "explanation": "Detailed explanation with advanced concepts"
}"""

    try:
        print("🧠 Getting full PhD-level question from DeepSeek R1...")
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-r1:14b",
                "prompt": prompt,
                "stream": False,  # No streaming to get complete response
                "format": "json",
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 3000
                }
            },
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            
            print(f"✅ Generated response ({len(generated_text)} chars)")
            print("=" * 80)
            print("📝 FULL RESPONSE:")
            print(generated_text)
            print("=" * 80)
            
            # Try to parse as JSON
            try:
                question_data = json.loads(generated_text)
                
                question = question_data.get('question', '')
                options = question_data.get('options', [])
                explanation = question_data.get('explanation', '')
                
                print(f"\n🎓 EXTRACTED PhD-LEVEL QUESTION:")
                print(f"📝 Question: {question}")
                print(f"🔢 Options: {options}")
                print(f"📚 Explanation: {explanation}")
                
                # Analyze PhD-level content
                phd_indicators = [
                    'perturbation theory', 'many-body', 'field theory', 'relativistic',
                    'entanglement', 'decoherence', 'quantum field', 'second quantization',
                    'green\'s function', 'feynman diagram', 'renormalization', 'symmetry breaking',
                    'phase transition', 'critical point', 'correlation function', 'scattering matrix',
                    'fine structure', 'hyperfine', 'zeeman', 'stark', 'lamb shift', 'casimir',
                    'dirac equation', 'klein-gordon', 'pauli matrices', 'spinor', 'gauge theory'
                ]
                
                full_text = f"{question} {explanation}".lower()
                phd_found = [term for term in phd_indicators if term in full_text]
                
                print(f"\n🔍 PhD-LEVEL ANALYSIS:")
                print(f"   Advanced terms found: {phd_found}")
                print(f"   PhD-level score: {len(phd_found)}/25 indicators")
                
                if len(phd_found) >= 3:
                    print("🎉 SUCCESS: This is genuinely PhD-level!")
                    return question_data
                elif len(phd_found) >= 1:
                    print("⚠️ PARTIAL: Some advanced concepts but could be better")
                    return question_data
                else:
                    print("❌ FAILED: Still undergraduate level")
                    return None
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print("💡 Raw response might contain thinking tokens")
                return None
        else:
            print(f"❌ Request failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    question = get_full_phd_question()
    
    if question:
        print("\n✅ Successfully generated PhD-level question!")
        print("🎯 Ready to integrate into the app")
    else:
        print("\n❌ Failed to generate PhD-level question")
        print("🔄 Need to try again or adjust prompt")
