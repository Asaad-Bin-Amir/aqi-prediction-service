"""Check models in registry"""
from src.model_registry import ModelRegistry

with ModelRegistry() as reg:
    models = reg.list_models()
    
    if not models:
        print('❌ No models in registry!')
        print('Run: python train_optimized.py')
    else:
        print('📋 Models in Registry:')
        for m in models:
            print(f"  {m['model_name']} {m['version']} - {m['stage']}")