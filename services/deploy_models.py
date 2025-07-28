#!/usr/bin/env python3
"""
Script to deploy ONNX models to Triton model repository.
This can be run as an init container or standalone script.
"""
import os
import shutil
import sys
from pathlib import Path


def deploy_model_to_triton(legacy_model_name: str, triton_model_name: str, model_repository_path: Path, source_models_path: Path):
    """Deploy a single model from source directory to Triton model repository."""
    
    print(f"Deploying {legacy_model_name} -> {triton_model_name}")
    
    # Create Triton model directory structure
    triton_model_dir = model_repository_path / triton_model_name / "1"
    triton_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for model files in source directory
    # TODO: Replace with fsspec access to actual storage
    source_model_dir = source_models_path / legacy_model_name.lower()
    
    if not source_model_dir.exists():
        print(f"  ⚠️ Source model directory not found: {source_model_dir}")
        print(f"  📝 Creating placeholder for manual deployment")
        
        # Create placeholder file to indicate where model should be placed
        placeholder = triton_model_dir / "model.onnx.placeholder"
        with open(placeholder, 'w') as f:
            f.write(f"# Placeholder for {legacy_model_name} ONNX model\n")
            f.write(f"# Copy the actual model.onnx file here and remove this placeholder\n")
            f.write(f"# Expected source: storage system PREDICT/{legacy_model_name.lower()}/\n")
            f.write(f"# Target: {triton_model_dir}/model.onnx\n")
        
        print(f"  📝 Created placeholder: {placeholder}")
        return False
    
    try:
        # Find ONNX files in the source directory
        onnx_files = list(source_model_dir.glob("*.onnx"))
        
        if not onnx_files:
            print(f"  ❌ No ONNX files found in {source_model_dir}")
            return False
            
        if len(onnx_files) > 1:
            print(f"  ⚠️ Multiple ONNX files found, using the first one: {onnx_files[0].name}")
        
        onnx_file = onnx_files[0]
        destination = triton_model_dir / "model.onnx"
        
        # Copy the ONNX model file
        shutil.copy2(onnx_file, destination)
        print(f"  ✅ Copied {onnx_file.name} -> {destination}")
        
        # Verify the copy was successful
        if destination.exists() and destination.stat().st_size > 0:
            file_size = destination.stat().st_size / (1024 * 1024)  # MB
            print(f"  📊 Model size: {file_size:.2f} MB")
            return True
        else:
            print(f"  ❌ Copy failed or resulted in empty file")
            return False
            
    except Exception as e:
        print(f"  ❌ Error deploying {legacy_model_name}: {e}")
        return False


def main():
    """Main deployment function."""
    
    print("🚀 Deploying Phase 1 models to Triton model repository")
    print("=" * 60)
    
    # Get paths
    script_dir = Path(__file__).parent
    model_repository_path = script_dir / "model-repository"
    
    # Source models path - can be overridden via environment variable
    source_models_path = Path(os.getenv("SOURCE_MODELS_PATH", "/app/models"))
    
    print(f"📁 Model repository: {model_repository_path}")
    print(f"📁 Source models: {source_models_path}")
    
    if not model_repository_path.exists():
        print(f"❌ Model repository not found at {model_repository_path}")
        sys.exit(1)
    
    # Models to deploy (legacy_name -> triton_model_name)
    models_to_deploy = [
        ("ProtT5Conservation", "conservation_model"),
        ("ProtT5SecondaryStructure", "secondary_structure_model"),
        ("SETH", "disorder_onnx"),  # ONNX component of ensemble
        ("BindEmbed", "binding_sites_model"),
        ("LightAttentionMembrane", "membrane_localization_model"),
        ("LightAttentionSubcellularLocalization", "subcellular_localization_model"),
        ("TMbed", "tmbed_ensemble"),
        ("VespaG", "variant_effects_model"),
    ]
    
    print(f"📋 Deploying {len(models_to_deploy)} models")
    
    deployed_successfully = 0
    total_models = len(models_to_deploy)
    
    for legacy_name, triton_name in models_to_deploy:
        success = deploy_model_to_triton(legacy_name, triton_name, model_repository_path, source_models_path)
        if success:
            deployed_successfully += 1
        print()
    
    print("=" * 60)
    print(f"📈 Deployment Summary:")
    print(f"   Successfully deployed: {deployed_successfully}/{total_models} models")
    
    if deployed_successfully == total_models:
        print(f"✅ All models deployed successfully!")
        print("\n🎯 Next steps:")
        print("   1. Start the services: docker-compose -f docker-compose.services.yml up")
        print("   2. Verify Triton models are loaded: curl http://localhost:8000/v2/models")
        print("   3. Check model status: curl http://localhost:8002/models/list")
        print("   4. Test predictions: curl -X POST http://localhost:8002/predictions/conservation -d '{\"sequences\":[\"MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF\"]}'")
    else:
        print("⚠️ Models require manual deployment. Check placeholder files created above.")
        print("\n🎯 Manual deployment steps:")
        print("   1. Copy your ONNX model files to the appropriate directories")
        print("   2. Remove the .placeholder files")
        print("   3. Start the services")
        print(f"\n📋 Expected model locations:")
        for legacy_name, triton_name in models_to_deploy:
            triton_dir = model_repository_path / triton_name / "1" 
            print(f"   {legacy_name}: {triton_dir}/model.onnx")
        
        # Don't exit with error for manual deployment case
        sys.exit(0)


if __name__ == "__main__":
    main()