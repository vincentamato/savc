import os
import json
import logging
import torch
from pathlib import Path
import pandas as pd
from artwork_music_matcher import AdvancedArtworkMusicMatcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_paths(midi_csv_path, image_dir, image_list_path):
    """Verify all required paths exist and are accessible"""
    errors = []
    
    # Check MIDI CSV
    if not os.path.exists(midi_csv_path):
        errors.append(f"MIDI features CSV not found: {midi_csv_path}")
    else:
        try:
            df = pd.read_csv(midi_csv_path)
            if 'extracted_path' not in df.columns:
                errors.append("CSV missing required 'extracted_path' column")
        except Exception as e:
            errors.append(f"Error reading MIDI CSV: {str(e)}")
    
    # Check image directory
    if not os.path.isdir(image_dir):
        errors.append(f"Image directory not found: {image_dir}")
    
    # Check image list
    if not os.path.exists(image_list_path):
        errors.append(f"Image list file not found: {image_list_path}")
    
    return errors

def test_single_image(
    midi_csv_path: str,
    image_dir: str,
    image_list_path: str,
    output_dir: str,
    num_matches: int = 3
):
    """
    Test the artwork-music matcher with a single image
    
    Args:
        midi_csv_path: Path to CSV containing MIDI features
        image_dir: Base directory containing artwork images
        image_list_path: Path to file containing image paths
        output_dir: Directory to save results
        num_matches: Number of top matches to return
    """
    logger.info("Starting single image test")
    
    # Verify paths
    errors = verify_paths(midi_csv_path, image_dir, image_list_path)
    if errors:
        for error in errors:
            logger.error(error)
        return
    
    # Create output directory if needed
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load first valid image path from the list
    with open(image_list_path, 'r') as f:
        image_paths = []
        for line in f:
            path = line.strip()
            full_path = os.path.join(image_dir, path)
            if os.path.exists(full_path):
                image_paths = [path]
                break
    
    if not image_paths:
        logger.error("No valid image paths found!")
        return
    
    test_image = image_paths[0]
    logger.info(f"Testing with image: {test_image}")
    
    try:
        # Initialize matcher
        matcher = AdvancedArtworkMusicMatcher(
            midi_features_path=midi_csv_path,
            image_base_dir=image_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Find matches
        matches = matcher.find_matching_music(test_image, top_k=num_matches)
        
        if matches:
            # Prepare detailed results
            results = {
                "image_path": test_image,
                "matches": [
                    {
                        "midi_path": midi_path,
                        "score": float(score),
                        "exists": os.path.exists(midi_path)
                    }
                    for midi_path, score in matches
                ]
            }
            
            # Save results
            output_path = output_dir / "test_match.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Successfully saved matches to {output_path}")
            logger.info("\nMatch details:")
            logger.info(json.dumps(results, indent=2))
            
            # Additional analysis
            logger.info("\nMatch Analysis:")
            for idx, match in enumerate(results["matches"], 1):
                logger.info(f"\nMatch {idx}:")
                logger.info(f"MIDI: {os.path.basename(match['midi_path'])}")
                logger.info(f"Score: {match['score']:.3f}")
                logger.info(f"MIDI file exists: {match['exists']}")
                
        else:
            logger.error("No matches found!")
            
    except Exception as e:
        logger.error(f"Error during matching: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Configuration
    config = {
        "midi_csv_path": "data/midi_features.csv",
        "image_dir": "data/artbench",
        "image_list_path": "data/artbench/paths.txt",
        "output_dir": "test_results",
        "num_matches": 3
    }
    
    test_single_image(**config)