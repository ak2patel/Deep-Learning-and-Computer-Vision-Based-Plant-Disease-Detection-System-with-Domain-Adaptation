"""
Test the plant disease model with the PlantDoc test dataset images
"""
import sys
import os
from pathlib import Path
from PIL import Image
import json

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from models.disease_model_pytorch import DiseaseModel

# Mapping: PlantDoc folder names -> exact PlantVillage folder names
PLANTDOC_TO_PV_MAP = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot"
}

def get_expected_label(img_path):
    """Extract expected label based on parent folder name"""
    folder_name = img_path.parent.name
    return PLANTDOC_TO_PV_MAP.get(folder_name, None)

def test_model():
    """Test model on PlantDoc test dataset images"""
    print("=" * 80)
    print("TESTING PLANT DISEASE MODEL (AGRINET)")
    print("=" * 80)
    
    # Initialize model
    print("\n[INFO] Loading model...")
    model = DiseaseModel()
    
    # Get test images from PlantDoc test set
    test_dir = Path(current_dir) / ".." / "ml-training" / "PlantDoc-Dataset" / "test"
    test_dir = test_dir.resolve()
    
    if not test_dir.exists():
        print(f"[ERROR] Test directory not found: {test_dir}")
        return
    
    test_files = set(test_dir.glob("**/*"))
    test_images = sorted([p for p in test_files if p.suffix.lower() in ('.jpg', '.jpeg', '.png') and p.is_file()])
    print(f"[OK] Found {len(test_images)} unique test images in {test_dir}\n")
    
    # Test each image
    results = []
    correct = 0
    total = 0
    skipped = 0
    
    # Iterate and print progress
    for img_path in sorted(test_images):
        filename = img_path.name
        folder_name = img_path.parent.name
        expected = get_expected_label(img_path)
        
        if expected is None:
            skipped += 1
            continue
        
        # Load and predict
        try:
            image = Image.open(img_path).convert("RGB")
            prediction = model.predict(image)
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")
            continue
        
        # Check if correct
        is_correct = prediction['predicted_class'] == expected
        correct += is_correct
        total += 1
        
        # Format expected label
        exp_plant, exp_disease = model.format_disease_name(expected)
        
        # Print status for progress update (limit print output spam for 200+ files)
        status = "[OK]" if is_correct else "[X]"
        if total % 10 == 0 or not is_correct:
            print(f"[{total}/{len(test_images) - skipped}] {status} {folder_name}/{filename}")
            print(f"   Expected: {exp_plant} - {exp_disease}")
            print(f"   Predicted: {prediction['plant_name']} - {prediction['disease_name']} ({prediction['confidence']:.2f}%)")
            print()
            
        # Store result
        results.append({
            'filename': filename,
            'folder': folder_name,
            'expected': expected,
            'predicted': prediction['predicted_class'],
            'confidence': prediction['confidence'],
            'correct': is_correct
        })
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"Total Images Checked: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Incorrect Predictions: {total - correct}")
    print(f"Skipped (unmapped classes): {skipped}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Group by disease type
    print("\n[STATS] Results by Disease Type:")
    disease_stats = {}
    for result in results:
        expected = result['expected']
        if expected not in disease_stats:
            disease_stats[expected] = {'correct': 0, 'total': 0}
        disease_stats[expected]['total'] += 1
        if result['correct']:
            disease_stats[expected]['correct'] += 1
    
    for disease, stats in sorted(disease_stats.items()):
        plant, disease_name = model.format_disease_name(disease)
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {plant} - {disease_name}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    # Save results to JSON
    output_file = Path(current_dir) / "test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total': total,
                'correct': correct,
                'accuracy': accuracy,
                'skipped': skipped
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n[INFO] Detailed results saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    test_model()
