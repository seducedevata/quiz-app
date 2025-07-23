#!/usr/bin/env python3
"""
Integration test for offline quiz generation bug fixes

This script performs end-to-end testing of the offline quiz generation
system to ensure all bug fixes work together properly.

Run with: python tests/integration_test_offline_fixes.py
"""

import sys
import os
import time
import json
import asyncio
from unittest.mock import Mock, patch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_gpu_memory_management():
    """Test GPU memory management integration"""
    print("🔧 Testing GPU Memory Management...")
    
    try:
        from knowledge_app.core.offline_mcq_generator import OfflineMCQGenerator
        
        # Mock torch to simulate GPU operations
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('torch.cuda.memory_allocated', return_value=1024*1024*1000), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            mock_device = Mock()
            mock_device.total_memory = 1024*1024*8000  # 8GB
            mock_device.name = "Test GPU"
            mock_props.return_value = mock_device
            
            generator = OfflineMCQGenerator()
            
            # Test GPU memory context
            with generator._gpu_memory_context():
                pass
            
            # Verify cleanup was called
            assert mock_empty_cache.call_count >= 2
            print("   ✅ GPU memory context manager working")
            
            # Test memory availability check
            result = generator._check_gpu_memory_availability(required_mb=500.0)
            assert result is True
            print("   ✅ GPU memory availability check working")
            
            print("✅ GPU Memory Management: PASSED")
            return True
            
    except Exception as e:
        print(f"❌ GPU Memory Management: FAILED - {e}")
        return False


def test_timeout_and_cancellation():
    """Test timeout and cancellation integration"""
    print("🛑 Testing Timeout and Cancellation...")
    
    try:
        from knowledge_app.core.offline_mcq_generator import OfflineMCQGenerator
        
        with patch('knowledge_app.core.offline_mcq_generator.OllamaModelInference'):
            generator = OfflineMCQGenerator()
            
            # Test cancellation token
            assert hasattr(generator, '_cancellation_token')
            assert not generator.is_cancelled()
            print("   ✅ Cancellation token initialized")
            
            # Test cancellation
            generator.cancel_generation()
            assert generator.is_cancelled()
            print("   ✅ Generation cancellation working")
            
            # Test reset
            generator.reset_cancellation()
            assert not generator.is_cancelled()
            print("   ✅ Cancellation reset working")
            
            print("✅ Timeout and Cancellation: PASSED")
            return True
            
    except Exception as e:
        print(f"❌ Timeout and Cancellation: FAILED - {e}")
        return False


def test_error_handling():
    """Test enhanced error handling integration"""
    print("🔧 Testing Enhanced Error Handling...")
    
    try:
        from knowledge_app.core.model_manager_service import ModelManagerService
        
        model_manager = ModelManagerService()
        
        # Test error callback
        callback_messages = []
        def error_callback(message):
            callback_messages.append(message)
        
        model_manager.set_error_callback(error_callback)
        model_manager._report_error("Test error message")
        
        assert len(callback_messages) == 1
        print("   ✅ Error callback system working")
        
        # Test retry logic
        call_count = 0
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "Success"
        
        result = model_manager._retry_with_exponential_backoff(failing_function, max_retries=3, base_delay=0.01)
        assert result == "Success"
        print("   ✅ Retry logic working")
        
        print("✅ Enhanced Error Handling: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Error Handling: FAILED - {e}")
        return False


def test_prompt_optimization():
    """Test prompt optimization integration"""
    print("🧠 Testing Prompt Optimization...")
    
    try:
        from knowledge_app.core.offline_mcq_generator import OfflineMCQGenerator
        
        with patch('knowledge_app.core.offline_mcq_generator.OllamaModelInference'):
            generator = OfflineMCQGenerator()
            
            # Test model capabilities detection
            caps_deepseek = generator._get_model_capabilities("deepseek-r1")
            caps_llama = generator._get_model_capabilities("llama-7b")
            
            assert caps_deepseek['max_context'] > caps_llama['max_context']
            assert caps_deepseek['supports_reasoning'] and not caps_llama['supports_reasoning']
            print("   ✅ Model capabilities detection working")
            
            # Test context length checking
            short_prompt = "Generate a question"
            long_prompt = "x" * 10000
            
            assert generator._check_context_length(short_prompt, "llama-7b")
            assert not generator._check_context_length(long_prompt, "llama-7b")
            print("   ✅ Context length validation working")
            
            # Test model-optimized prompts
            prompt_simple = generator._create_model_optimized_prompt("math", "", 1, "multiple_choice", "llama-7b")
            prompt_advanced = generator._create_model_optimized_prompt("math", "", 1, "multiple_choice", "deepseek-r1")
            
            assert len(prompt_advanced) > len(prompt_simple)
            print("   ✅ Model-optimized prompts working")
            
            print("✅ Prompt Optimization: PASSED")
            return True
            
    except Exception as e:
        print(f"❌ Prompt Optimization: FAILED - {e}")
        return False


def test_content_validation():
    """Test content validation integration"""
    print("🔍 Testing Content Validation...")
    
    try:
        from knowledge_app.core.offline_mcq_generator import OfflineMCQGenerator
        
        with patch('knowledge_app.core.offline_mcq_generator.OllamaModelInference'):
            generator = OfflineMCQGenerator()
            
            # Test valid MCQ
            valid_mcq = {
                "question": "What is the capital of France?",
                "options": {"A": "London", "B": "Berlin", "C": "Paris", "D": "Madrid"},
                "correct": "C",
                "explanation": "Paris is the capital and largest city of France."
            }
            
            result = generator._validate_mcq_structure(valid_mcq)
            assert result['is_valid']
            assert result['quality_score'] > 60
            print("   ✅ Valid MCQ validation working")
            
            # Test invalid MCQ
            invalid_mcq = {
                "question": "What?",
                "options": {"A": "Yes", "B": "No"},  # Missing C and D
                "correct": "A"
                # Missing explanation
            }
            
            result = generator._validate_mcq_structure(invalid_mcq)
            assert not result['is_valid']
            assert len(result['errors']) > 0
            print("   ✅ Invalid MCQ detection working")
            
            print("✅ Content Validation: PASSED")
            return True
            
    except Exception as e:
        print(f"❌ Content Validation: FAILED - {e}")
        return False


def test_ui_integration():
    """Test UI integration"""
    print("🖥️ Testing UI Integration...")
    
    try:
        # Test GPU stats functionality
        from knowledge_app.webengine_app import WebEngineApp
        
        with patch('knowledge_app.webengine_app.QWebEngineView'), \
             patch('knowledge_app.webengine_app.QApplication'):
            
            app = WebEngineApp()
            app.page = Mock()
            app.page.return_value.runJavaScript = Mock()
            
            # Test GPU stats with mocked torch
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.get_device_properties') as mock_props, \
                 patch('torch.cuda.memory_allocated', return_value=1024*1024*1000):
                
                mock_device = Mock()
                mock_device.total_memory = 1024*1024*8000
                mock_device.name = "Test GPU"
                mock_props.return_value = mock_device
                
                app.getGPUStats()
                
                # Verify JavaScript was called
                app.page().runJavaScript.assert_called_once()
                call_args = app.page().runJavaScript.call_args[0][0]
                assert "updateGPUStatsDisplay" in call_args
                print("   ✅ GPU stats UI integration working")
        
        print("✅ UI Integration: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ UI Integration: FAILED - {e}")
        return False


def main():
    """Run all integration tests"""
    print("🚀 Starting Offline Quiz Generation Bug Fix Integration Tests")
    print("=" * 70)
    
    tests = [
        test_gpu_memory_management,
        test_timeout_and_cancellation,
        test_error_handling,
        test_prompt_optimization,
        test_content_validation,
        test_ui_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 70)
    print(f"📊 Integration Test Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {(passed / (passed + failed)) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All integration tests passed! Bug fixes are working correctly.")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please review the fixes.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
