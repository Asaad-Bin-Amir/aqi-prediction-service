"""Promote latest models to production"""
from src.model_registry import ModelRegistry

# Latest versions from your registry
VERSIONS = {
    '24h': 'v20260217_0114',
    '48h': 'v20260217_0117',
    '72h': 'v20260217_0120'
}

with ModelRegistry() as reg:
    print("Promoting latest models to production...\n")
    
    for horizon, version in VERSIONS.items():
        success = reg.promote_to_production(f'aqi_forecast_{horizon}', version)
        if success:
            print(f"✅ {horizon} promoted successfully")
        else:
            print(f"❌ {horizon} promotion failed")
    
    print('\n' + '='*50)
    print('Verifying production models:')
    print('='*50 + '\n')
    
    for horizon in ['24h', '48h', '72h']:
        model, meta = reg.get_production_model(f'aqi_forecast_{horizon}')
        if model and meta:
            print(f"✅ {horizon}: {meta['version']}")
            print(f"   R²: {meta['metrics']['r2']:.3f} | MAE: {meta['metrics']['mae']:.3f}")
        else:
            print(f"❌ {horizon}: No production model found")